"""说话人识别 - 使用 pyannote-audio 识别音频中的不同说话人"""

import os
import tempfile
import shutil

import torch

from vl.core.config import get_config
from vl.core.logging import get_logger
from vl.core.models.transcript import DiarizationSegment

logger = get_logger()


def _ensure_hf_env():
    """确保 HuggingFace 环境变量已设置"""
    from dotenv import load_dotenv
    from pathlib import Path

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            try:
                content = (parent / "pyproject.toml").read_text(encoding="utf-8")
                if 'name = "videolens"' in content:
                    env_path = parent / ".env"
                    if env_path.is_file():
                        load_dotenv(str(env_path), override=False)
                    break
            except (OSError, UnicodeDecodeError):
                pass


class Diarizer:
    """说话人识别器"""

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        hf_token: str = "",
    ):
        _ensure_hf_env()

        if not hf_token:
            hf_token = os.getenv("HF_TOKEN", "")

        if not hf_token:
            raise ValueError(
                "缺少 HF_TOKEN。请在 .env 中配置，"
                "并到 https://huggingface.co/pyannote/speaker-diarization-3.1 接受模型使用条款"
            )

        logger.info(f"加载说话人识别模型: {model_name}")

        # pyannote diarization 是 gated model，hf-mirror 不支持认证
        # 临时移除 mirror 端点，直接从 huggingface.co 下载
        _saved_endpoint = os.environ.pop("HF_ENDPOINT", None)
        _saved_hub_url = os.environ.pop("HUGGINGFACE_HUB_URL", None)
        try:
            from pyannote.audio import Pipeline
            self.pipeline = Pipeline.from_pretrained(
                model_name,
                token=hf_token,
            )
        finally:
            # 恢复 mirror 设置
            if _saved_endpoint:
                os.environ["HF_ENDPOINT"] = _saved_endpoint
            if _saved_hub_url:
                os.environ["HUGGINGFACE_HUB_URL"] = _saved_hub_url

        logger.info("说话人识别模型加载完成")

    def diarize(self, audio_path: str) -> list[DiarizationSegment]:
        """识别音频中的说话人片段"""
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        logger.info(f"开始说话人识别: {audio_path}")

        # pyannote 4.x 依赖 torchcodec 解码音频，Windows 上不可用
        # 使用 soundfile 预加载为 waveform dict 绕过
        import soundfile as sf
        import numpy as np
        waveform_np, sample_rate = sf.read(audio_path, dtype="float32")
        # soundfile 返回 (samples, channels)，pyannote 需要 (channels, samples)
        if waveform_np.ndim == 1:
            waveform_np = waveform_np[np.newaxis, :]
        else:
            waveform_np = waveform_np.T
        waveform = torch.from_numpy(waveform_np)

        audio_dict = {"waveform": waveform, "sample_rate": sample_rate}
        diarization = self.pipeline(audio_dict)

        results = []
        # pyannote 4.x 返回 DiarizeOutput 对象
        timeline = diarization.speaker_diarization
        for segment, _, speaker in timeline.itertracks(yield_label=True):
            results.append(DiarizationSegment(
                speaker=speaker,
                start_time=segment.start,
                end_time=segment.end,
            ))

        speakers = set(r.speaker for r in results)
        logger.info(f"说话人识别完成: {len(results)} 个片段, {len(speakers)} 个说话人")
        return results
