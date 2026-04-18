"""Qwen3.5-Omni-Plus 全模态视频理解模块

通过 DashScope OpenAI 兼容 API 调用 qwen3.5-omni-plus 模型，
同时传入音频 + 关键帧图片 + 结构化 prompt，实现像人一样的视频理解：
  - 听到谁在说话、说了什么、什么情感
  - 看到画面中有什么角色、在做什么、什么场景
  - 自然而然地关联音频与画面

核心能力:
  - 说话人识别 (根据音色区分)
  - 逐句转录 + 时间戳 + 情感
  - 视觉场景描述 + 角色识别
  - 音频与画面的自然关联

流程:
  1. 将视频按时段切分为 ~2min 片段 (按场景边界)
  2. 每个片段: 音频切片 + 2~5张关键帧 → qwen3.5-omni-plus
  3. 解析结构化 JSON 输出 (说话人/台词/情感/视觉描述)
  4. 映射回各场景
"""

import base64
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from vl.core.logging import get_logger

logger = get_logger()


@dataclass
class OmniSegment:
    """单个转写片段 (含说话人 + 情感)"""
    segment_id: str
    text: str
    start_time: float
    end_time: float
    speaker: str = ""
    emotion: str = ""
    language: str = ""
    scene_id: str = ""
    confidence: float = 1.0

    def to_dict(self):
        return {
            "segment_id": self.segment_id,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "speaker": self.speaker,
            "emotion": self.emotion,
            "language": self.language,
            "scene_id": self.scene_id,
            "confidence": self.confidence,
        }


@dataclass
class OmniSceneDescription:
    """片段的视觉场景描述"""
    time_of_day: str = ""
    space: str = ""
    subspace: str = ""
    scene: str = ""
    characters: list = field(default_factory=list)
    main_actions: str = ""
    interactions: str = ""
    emotion: str = ""
    plot_state: str = ""

    def to_dict(self):
        return {
            "time_of_day": self.time_of_day,
            "space": self.space,
            "subspace": self.subspace,
            "scene": self.scene,
            "characters": self.characters,
            "main_actions": self.main_actions,
            "interactions": self.interactions,
            "emotion": self.emotion,
            "plot_state": self.plot_state,
        }


@dataclass
class OmniChunkResult:
    """单个片段的完整理解结果"""
    chunk_index: int
    time_start: float
    time_end: float
    segments: list[OmniSegment] = field(default_factory=list)
    speakers: list[dict] = field(default_factory=list)
    scene_description: Optional[OmniSceneDescription] = None
    raw_text: str = ""
    content_type: str = "main"  # "opening" | "main" | "ending"


# 结构化 Prompt —— 引导模型输出 JSON
STRUCTURED_PROMPT = """\
请对这段音视频片段进行详细分析。你将同时听到音频和看到关键帧画面。

请严格按以下 JSON 格式输出（不要输出其他内容）：
{
  "content_type": "main",
  "speakers": [
    {"id": "说话人A", "description": "音色/语气特征", "possible_identity": "如果能从画面判断角色身份则标注"}
  ],
  "transcript": [
    {"start": 0.0, "end": 3.5, "speaker": "说话人A", "emotion": "平静", "text": "台词内容"}
  ],
  "visual": {
    "time_of_day": "白天或晚上",
    "space": "具体环境名称，如实验室、草地、舞台",
    "subspace": "更细的子区域，如沙发上、树下",
    "scene": "室内或室外",
    "characters": ["角色1", "角色2"],
    "main_actions": "正在发生的主要动作",
    "interactions": "角色间的互动",
    "emotion": "画面氛围和情感基调",
    "plot_state": "剧情推进状态"
  }
}

规则：
1. 根据音色区分不同说话人，命名为"说话人A"、"说话人B"等
2. 结合画面判断说话人可能的角色身份（如看到灰色狼 → 可能是灰太狼）
3. start/end 为相对于该片段开始的秒数，保留一位小数
4. emotion 从以下选择：平静、愉快、悲伤、愤怒、惊讶、恐惧、厌恶
5. 如果没有语音内容，transcript 为空数组
6. content_type 判断这段内容属于视频的哪个部分：
   - "opening": 片头曲/主题曲。特征：完整主题歌曲播放、快速蒙太奇剪辑、标题画面、角色展示阵列、通常没有推进剧情的对话。
   - "main": 正片/主要故事内容。特征：角色之间的叙事对话、情节推进、场景转换有叙事意义。注意：正片中也会有背景音乐和快速镜头切换，关键区别是内容是否服务于故事叙述。
   - "ending": 片尾曲/片尾画面。特征：片尾音乐、滚动字幕、静止或循环画面、总结性画面。
   如果不确定，默认为 "main"。
7. 只输出 JSON，不要输出任何其他内容"""


class QwenOmni:
    """基于 Qwen3.5-Omni-Plus 的全模态视频理解器"""

    def __init__(
        self,
        api_key: str = "",
        model: str = "qwen3.5-omni-plus",
        language: str = "zh",
    ):
        self.api_key = api_key
        self.model = model
        self.language = language

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def understand_chunk(
        self,
        audio_path: str,
        image_paths: list[str],
        time_offset: float,
        chunk_index: int = 0,
        time_end: float = 0.0,
    ) -> OmniChunkResult:
        """
        对单个视频片段进行全模态理解。

        Args:
            audio_path: 该片段的音频文件路径
            image_paths: 该片段的关键帧图片路径列表
            time_offset: 该片段在原始视频中的起始时间 (秒)
            chunk_index: 片段序号
            time_end: 该片段的结束时间 (秒)
        """
        logger.info(f"全模态理解片段 {chunk_index + 1}: "
                     f"[{time_offset:.1f}s - {time_end:.1f}s], "
                     f"{len(image_paths)} 张关键帧")

        # 构建请求内容
        content = []

        # 1. 音频 (base64)
        audio_data_uri = self._encode_file(audio_path, "audio/wav")
        content.append({
            "type": "input_audio",
            "input_audio": {"data": audio_data_uri, "format": "wav"},
        })

        # 2. 关键帧图片 (base64)
        for img_path in image_paths:
            img_data_uri = self._encode_file(img_path, "image/jpeg")
            content.append({
                "type": "image_url",
                "image_url": {"url": img_data_uri},
            })

        # 3. 结构化 Prompt
        content.append({"type": "text", "text": STRUCTURED_PROMPT})

        # 调用 API (必须 stream=True)
        raw_text = self._call_api(content)

        # 解析结果
        result = self._parse_response(raw_text, time_offset, chunk_index, time_end)
        result.raw_text = raw_text

        logger.info(f"  片段 {chunk_index + 1}: "
                     f"{len(result.segments)} 句台词, "
                     f"{len(result.speakers)} 位说话人")
        return result

    def _encode_file(self, file_path: str, mime_type: str) -> str:
        """将本地文件编码为 base64 Data URL"""
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        return f"data:{mime_type};base64,{b64}"

    def _call_api(self, content: list) -> str:
        """调用 Qwen3.5-Omni-Plus API (流式)"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            modalities=["text"],
            stream=True,
            stream_options={"include_usage": True},
        )

        full_text = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                full_text += chunk.choices[0].delta.content

        return full_text

    def _parse_response(
        self,
        raw: str,
        time_offset: float,
        chunk_index: int,
        time_end: float,
    ) -> OmniChunkResult:
        """解析模型输出的 JSON"""
        result = OmniChunkResult(
            chunk_index=chunk_index,
            time_start=time_offset,
            time_end=time_end,
        )

        parsed = self._extract_json(raw)
        if not parsed or not isinstance(parsed, dict):
            # JSON 解析失败，作为纯文本处理
            if raw.strip():
                result.segments.append(OmniSegment(
                    segment_id=uuid.uuid4().hex[:8],
                    text=raw.strip(),
                    start_time=time_offset,
                    end_time=time_end,
                ))
            return result

        # 解析 speakers
        result.speakers = parsed.get("speakers", [])

        # 解析 content_type
        raw_ct = parsed.get("content_type", "main").strip().lower()
        result.content_type = raw_ct if raw_ct in ("opening", "main", "ending") else "main"

        # 解析 transcript
        for item in parsed.get("transcript", []):
            if not isinstance(item, dict):
                continue
            text = item.get("text", "").strip()
            if not text:
                continue

            local_start = float(item.get("start", 0))
            local_end = float(item.get("end", local_start + 1))

            result.segments.append(OmniSegment(
                segment_id=uuid.uuid4().hex[:8],
                text=text,
                start_time=time_offset + local_start,
                end_time=time_offset + local_end,
                speaker=item.get("speaker", ""),
                emotion=item.get("emotion", ""),
                language=self.language,
            ))

        # 解析 visual
        visual = parsed.get("visual", {})
        if visual:
            result.scene_description = OmniSceneDescription(
                time_of_day=visual.get("time_of_day", ""),
                space=visual.get("space", ""),
                subspace=visual.get("subspace", ""),
                scene=visual.get("scene", ""),
                characters=visual.get("characters", visual.get("characters_visible", [])),
                main_actions=visual.get("main_actions", visual.get("actions", "")),
                interactions=visual.get("interactions", ""),
                emotion=visual.get("emotion", visual.get("atmosphere", "")),
                plot_state=visual.get("plot_state", ""),
            )

        return result

    @staticmethod
    def _extract_json(text: str):
        """从模型输出中提取 JSON"""
        import regex

        text = text.strip()

        # 尝试直接解析
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # 提取 JSON 代码块
        match = regex.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, regex.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 提取裸 JSON 对象
        match = regex.search(
            r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}",
            text, regex.DOTALL,
        )
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def assign_to_scenes(
        segments: list[OmniSegment],
        scenes: list,
    ) -> list[OmniSegment]:
        """将转写片段分配到对应的视频场景"""
        for seg in segments:
            mid = (seg.start_time + seg.end_time) / 2
            for scene in scenes:
                if scene.start_time <= mid <= scene.end_time:
                    seg.scene_id = scene.scene_id
                    break
        return segments
