# VideoLens — AI 项目上下文

> 影视视频内容分析与检索系统。输入一个视频文件，输出结构化知识库 + FAISS 语义检索索引，支持基于知识库的问答。

---

## 1. 技术栈

- **语言**: Python 3.12.10
- **构建**: PDM (`pyproject.toml`), 依赖安装用 `uv`
- **CLI**: Typer (`vl/app/cli.py`)
- **配置**: `.env` (密钥) + `config/pipeline.yaml` (参数/定价) + `config/prompts.yaml` (提示词模板)
- **LLM API**: 阿里 DashScope (OpenAI 兼容端点 + `MultiModalConversation` SDK)
- **向量检索**: `faiss-cpu` (`IndexFlatIP`, 内积 = 归一化余弦)
- **视觉编码**: `sentence-transformers/clip-ViT-B-32` (512 维)
- **场景检测**: `scenedetect[opencv]` (`ContentDetector`)
- **视频/音频处理**: OpenCV (`cv2`) + ffmpeg (subprocess)
- **声纹识别**: 科大讯飞声纹 1:N 搜索 API (HMAC-SHA256 鉴权)
- **日志**: `coloredlogs` + 文件 `data/output/videolens.log`

---

## 2. 目录树

```
VideoLens/
│
├── .env                                  # DASHSCOPE_API_KEY, HF_TOKEN, XFYUN_*
├── .env.example
├── .gitignore
├── pyproject.toml                        # PDM 构建, poe 任务定义
├── requirements.txt
├── uv.lock
├── README.md
├── PROJECT_CONTEXT_FOR_AI.md
│
├── config/
│   ├── pipeline.yaml                     # 模型名, 阈值, 路径, 定价
│   └── prompts.yaml                      # 所有 LLM 提示词模板 (9 个 key)
│
├── data/
│   ├── videos/
│   │   ├── 052 鸟蛋之争.mp4
│   │   ├── 053 懒羊羊的歌声.mp4
│   │   └── 054 羊毛节.mp4
│   └── output/                           # 流水线运行时生成
│       ├── videolens.log
│       ├── checkpoints/{video_id}.json
│       ├── stage1_scenes/{video_id}/
│       │   ├── scenes.json               # list[Scene.to_dict()]
│       │   ├── metadata.json             # enriched scenes (含 vlm_caption)
│       │   └── keyframes/                # scene_NNNN_NN.jpg
│       ├── stage2_features/
│       │   ├── preprocessing/{video_id}.wav
│       │   └── {video_id}/
│       │       ├── transcript.json       # list[TranscriptSegment.to_dict()]
│       │       ├── characters.json       # list[Character.to_dict()]
│       │       ├── clip_vectors.npy
│       │       └── voiceprint_result.json
│       ├── stage3_captions/{video_id}/
│       │   ├── captions.json
│       │   ├── index.faiss
│       │   ├── doc_store.json
│       │   └── index_id_map.json
│       ├── stage4_events/{video_id}/
│       │   └── events.json
│       └── stage5_knowledge/{video_id}/
│           └── knowledge_base.json
│
├── scripts/
│   └── test_kb_qa.py                     # 批量 QA 测试脚本
│
├── tests/
│   └── qa/
│       ├── 052 鸟蛋之争.json              # QA 测试集
│       └── report_052_鸟蛋之争_20260423.md
│
└── vl/                                   # ========== 源码包 ==========
    ├── __init__.py
    │
    ├── app/                              # CLI 层
    │   ├── cli.py                        # Typer app: index/search/qa/analyze/test-stage
    │   └── main.py                       # python -m vl.app.main 入口
    │
    ├── core/                             # 核心基础设施
    │   ├── config.py                     # AppConfig, load_config(), get_config()
    │   ├── cost.py                       # CostTracker, CallRecord, ModelPricing
    │   ├── logging.py                    # setup_logger(), get_logger()
    │   ├── paths.py                      # PathManager — 全部输出路径属性
    │   ├── helpers/
    │   │   ├── ffmpeg.py                 # extract_audio()
    │   │   ├── json_utils.py             # save_json(), load_json()
    │   │   ├── prompt_loader.py          # load_prompt(config, key) -> (user, system)
    │   │   ├── scene_utils.py            # assign_segments_to_scenes()
    │   │   └── text_utils.py             # extract_json(), extract_json_obj()
    │   ├── llm/
    │   │   ├── base_client.py            # BaseLLMClient (OpenAI 兼容端点)
    │   │   ├── qwen_text.py              # QwenTextClient.generate / generate_json
    │   │   └── qwen_vl.py               # QwenVLClient.analyze_images (滑动窗口)
    │   └── models/
    │       ├── __init__.py               # 统一导出 7 个模型类
    │       ├── scene.py                  # Scene
    │       ├── transcript.py             # TranscriptSegment, Word
    │       ├── character.py              # Character
    │       └── omni.py                   # OmniSegment, OmniSceneDescription, OmniChunkResult
    │
    ├── pipeline/                         # 流水线阶段
    │   ├── orchestrator.py               # PipelineOrchestrator, Checkpoint
    │   ├── stage1_ingestion.py           # run_stage1()  场景分割 + 关键帧
    │   ├── stage2_analysis.py            # run_stage2()  全模态理解 + CLIP + 角色
    │   ├── stage3_understanding.py       # run_stage3()  VLM caption + FAISS 索引
    │   ├── stage4_event_builder.py       # run_stage4()  事件提取
    │   └── stage5_knowledge.py           # run_stage5()  知识库增量生成
    │
    ├── scene/                            # 场景检测
    │   ├── detector.py                   # SceneDetector.detect_scenes()
    │   └── frame_sampler.py             # FrameSampler.sample_keyframes()
    │
    ├── asr/                              # 语音识别
    │   └── qwen_omni.py                 # QwenOmni.understand_chunk()
    │
    ├── vision/                           # 视觉编码
    │   └── clip_encoder.py              # CLIPEncoder.encode_images / encode_texts
    │
    ├── store/                            # 向量存储
    │   └── vector_store.py              # VectorStore (FAISS IndexFlatIP)
    │
    ├── voiceprint/                       # 声纹识别 (讯飞)
    │   ├── client.py                    # VoiceprintClient (1:N search API)
    │   └── matcher.py                   # match_per_segment()
    │
    ├── qa/                               # 知识库问答
    │   └── knowledge_qa.py             # answer_question(), batch_answer()
    │
    └── services/                         # 业务服务层
        ├── search.py                    # search_scenes() — CLIP + FAISS + LLM 重排
        ├── qa.py                        # answer_question() — 检索增强 QA
        ├── analysis.py                  # analyze_video() — 摘要/角色/时间线
        └── test_stage.py               # run_test_stage() — 单阶段测试
```

## 3. 关键文件清单

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `config/pipeline.yaml` | — | 模型名、场景检测阈值、ASR 参数、声纹配置、CLIP 维度、检索参数、路径、定价 |
| `config/prompts.yaml` | — | 9 个 prompt 模板: `stage2_omni`, `stage2_character_extract`, `scene_caption`, `event_builder`, `knowledge_summary`, `query_expand`, `rerank`, `qa_answer`, `kb_qa` |
| `vl/core/config.py` | `AppConfig`, `load_config()`, `get_config()` | 三层配置加载 (.env + pipeline.yaml + prompts.yaml)，frozen dataclass 全局单例 |
| `vl/core/paths.py` | `PathManager` | 30+ 个路径属性: `scenes_json_path`, `transcript_json_path`, `faiss_index_path`, `knowledge_base_path` 等 |
| `vl/core/cost.py` | `CostTracker`, `ModelPricing`, `CallRecord` | LLM 调用计费，按模型/阶段汇总，`report()` 输出格式化表格 |
| `vl/core/logging.py` | `setup_logger()`, `get_logger()` | coloredlogs 控制台 + 文件双输出，logger 名 `"videolens"` |
| `vl/core/models/scene.py` | `Scene` | 14 字段 dataclass, `from_dict()`, `to_dict()`, `get_normalized_caption()` |
| `vl/core/models/transcript.py` | `TranscriptSegment`, `Word` | 台词片段 + 词级详情 |
| `vl/core/models/omni.py` | `OmniChunkResult`, `OmniSegment`, `OmniSceneDescription` | Qwen-Omni 输出结构 |
| `vl/core/helpers/json_utils.py` | `save_json()`, `load_json()` | JSON 读写 + 自动建目录 |
| `vl/core/helpers/prompt_loader.py` | `load_prompt(config, key)` | 返回 `(user_template, system_prompt)` 元组 |
| `vl/core/helpers/text_utils.py` | `extract_json()`, `extract_json_obj()` | 3 策略 JSON 提取: 直接解析 → code block → 递归正则 |
| `vl/core/helpers/ffmpeg.py` | `extract_audio()` | subprocess ffmpeg 提取 16kHz mono WAV |
| `vl/core/helpers/scene_utils.py` | `assign_segments_to_scenes()` | 中点法将 segment 分配到 scene |
| `vl/core/llm/base_client.py` | `BaseLLMClient` | 封装 `openai.OpenAI` → DashScope 兼容端点, `chat()` 自动上报 cost |
| `vl/core/llm/qwen_text.py` | `QwenTextClient(BaseLLMClient)` | `generate()` / `generate_json()` |
| `vl/core/llm/qwen_vl.py` | `QwenVLClient` | 封装 `dashscope.MultiModalConversation.call()`, 滑动窗口多图分析 |
| `vl/asr/qwen_omni.py` | `QwenOmni` | OpenAI streaming 调用, base64 编码音频+图片, 解析到 `OmniChunkResult` |
| `vl/scene/detector.py` | `SceneDetector` | PySceneDetect `ContentDetector` |
| `vl/scene/frame_sampler.py` | `FrameSampler` | OpenCV `cv2.VideoCapture` 均匀采样, `cv2.imencode()` 处理中文路径 |
| `vl/vision/clip_encoder.py` | `CLIPEncoder` | SentenceTransformer 加载 CLIP ViT-B-32, 512 维 |
| `vl/store/vector_store.py` | `VectorStore` | FAISS `IndexFlatIP`, `save()/load()` 用 temp file 解决中文路径 |
| `vl/voiceprint/client.py` | `VoiceprintClient` | 讯飞声纹 API, HMAC-SHA256 鉴权, `search()` 1:N 识别 |
| `vl/voiceprint/matcher.py` | `match_per_segment()` | 逐段切音 → 1:N 搜索 → 替换 speaker_id |
| `vl/pipeline/orchestrator.py` | `PipelineOrchestrator`, `Checkpoint` | 6 阶段编排 + 断点续跑 + 代价报告 |
| `vl/pipeline/stage1_ingestion.py` | `run_stage1()` | SceneDetector + FrameSampler |
| `vl/pipeline/stage2_analysis.py` | `run_stage2()` | 音频分片 → QwenOmni → CLIP → 角色提取 LLM |
| `vl/pipeline/stage3_understanding.py` | `run_stage3()` | QwenVLClient 补全 caption → VectorStore 建 FAISS |
| `vl/pipeline/stage4_event_builder.py` | `run_stage4()` | 连续帧分组 → LLM 提取事件 (大事件+小事件) |
| `vl/pipeline/stage5_knowledge.py` | `run_stage5()` | 逐帧 LLM 增量更新知识库 JSON |
| `vl/services/search.py` | `search_scenes()` | CLIP 编码 → FAISS 检索 → 可选 LLM 重排 |
| `vl/services/qa.py` | `answer_question()` | search_scenes → LLM 生成答案 |
| `vl/services/analysis.py` | `analyze_video()` | 摘要/角色/时间线分析 |
| `vl/qa/knowledge_qa.py` | `answer_question()`, `batch_answer()` | 直接读取 knowledge_base.json → LLM 问答 |
| `vl/app/cli.py` | `index()`, `search()`, `qa()`, `analyze()`, `test_stage()` | Typer CLI 命令 |
| `vl/app/main.py` | `app()` | `python -m vl.app.main` 入口 |
| `scripts/test_kb_qa.py` | — | 独立批量 QA 测试, 加载 tests/qa/{video_id}.json |
| `pyproject.toml` | — | PDM 构建, poe 任务 (sync/index/search/qa/analyze/test-stage) |

---

## 4. 数据模型 (全部在 `vl/core/models/`)

| 类 | 文件 | 关键字段 |
|---|---|---|
| `Scene` | `scene.py` | `scene_id, video_id, index, start_time, end_time, start_frame, end_frame, keyframe_paths, transition_type, clip_embedding, vlm_caption, structured_caption, content_type, confidence` |
| `TranscriptSegment` | `transcript.py` | `segment_id, scene_id, speaker_id, text, start_time, end_time, words, confidence` |
| `Word` | `transcript.py` | `text, start, end, confidence` |
| `Character` | `character.py` | `character_id, label, appearance_scenes` |
| `OmniSegment` | `omni.py` | `segment_id, text, start_time, end_time, speaker, emotion, language, scene_id, confidence` |
| `OmniSceneDescription` | `omni.py` | `time_of_day, space, subspace, scene, characters, main_actions, interactions, emotion, plot_state` |
| `OmniChunkResult` | `omni.py` | `chunk_index, time_start, time_end, segments, speakers, scene_description, raw_text, content_type` |

所有模型都有 `to_dict()` 和 `from_dict(data)` (classmethod)。`Scene` 额外有 `duration` (property) 和 `get_normalized_caption()`。

导出集中在 `vl/core/models/__init__.py`：
```python
from vl.core.models.scene import Scene
from vl.core.models.transcript import TranscriptSegment, Word
from vl.core.models.character import Character
from vl.core.models.omni import OmniSegment, OmniSceneDescription, OmniChunkResult
```

---

## 5. 配置系统

**入口**: `vl/core/config.py`

**三层配置加载** (`load_config()`):
1. `.env` → `os.getenv()` 拿 `DASHSCOPE_API_KEY`, `XFYUN_APP_ID`, `XFYUN_API_KEY`, `XFYUN_API_SECRET`, `HF_TOKEN`, `HF_ENDPOINT`
2. `config/pipeline.yaml` → 各子节 (`models:`, `scene_detection:`, `asr:`, `voiceprint:`, `clip:`, `retrieval:`, `paths:`, `pricing:`)
3. `config/prompts.yaml` → 整个 dict 存入 `AppConfig.prompts`

**全局单例**: `get_config()` 返回 `AppConfig` frozen dataclass。

**AppConfig 关键字段** (31 个):

| 分组 | 字段 | 默认值 |
|------|------|--------|
| API | `dashscope_api_key` | `""` (从 .env) |
| 模型 | `model_vlm` | `"qwen-vl-max"` |
| | `model_text` | `"qwen-plus"` |
| | `model_omni` | `"qwen3.5-omni-plus"` |
| | `model_clip` | `"sentence-transformers/clip-ViT-B-32"` |
| 场景 | `content_threshold` | `27.0` |
| | `min_scene_len` | `1.0` |
| | `samples_per_scene` | `8` |
| VLM | `vlm_window_size` | `4` |
| | `vlm_stride` | `2` |
| ASR | `asr_chunk_duration` | `120` |
| | `asr_max_keyframes_per_chunk` | `5` |
| 声纹 | `voiceprint_enabled` | `False` |
| | `voiceprint_score_threshold` | `0.3` |
| | `voiceprint_min_duration` | `3.0` |
| | `voiceprint_name_mapping` | `{}` (pipeline.yaml 里的拼音→中文映射) |
| 检索 | `retrieval_top_k` | `10` |
| | `retrieval_rerank` | `True` |

**提示词模板加载**: `vl/core/helpers/prompt_loader.py` — `load_prompt(config, key) -> (user_template, system_prompt)`。模板变量用 Python `str.format()` 语法 (双花括号 `{{` 转义 JSON)。

---

## 6. 流水线架构

**编排器**: `vl/pipeline/orchestrator.py` — `PipelineOrchestrator`

### 6 个阶段 (含断点续跑)

| 阶段 | 函数 | 输入 | 输出文件 | 核心逻辑 |
|------|------|------|----------|----------|
| **stage1** 场景分割 | `run_stage1()` in `stage1_ingestion.py` | 视频文件 | `scenes.json`, `keyframes/*.jpg` | `SceneDetector.detect_scenes()` → `FrameSampler.sample_keyframes()` |
| **stage2** 特征提取 | `run_stage2()` in `stage2_analysis.py` | scenes + 音频 | `transcript.json`, `characters.json`, `clip_vectors.npy` | 音频分片 → `QwenOmni.understand_chunk()` → `CLIPEncoder.encode_images()` → 角色提取 LLM |
| **stage2.5** 声纹识别 | `_stage2_5_voiceprint()` in `orchestrator.py` | transcript.json + 音频 | `voiceprint_result.json` (覆写 transcript.json) | `match_per_segment()` → 讯飞 1:N 搜索 → 替换 speaker_id |
| **stage3** 帧描述 | `run_stage3()` in `stage3_understanding.py` | enriched scenes | `captions.json`, `index.faiss`, `doc_store.json` | `QwenVLClient.analyze_images()` 滑动窗口补全缺失 caption → `VectorStore` 建 FAISS 索引 |
| **stage4** 事件提取 | `run_stage4()` in `stage4_event_builder.py` | enriched scenes + transcripts | `events.json` | 连续帧分组 → `QwenTextClient.generate_json()` 提取事件 |
| **stage5** 知识库 | `run_stage5()` in `stage5_knowledge.py` | enriched scenes + transcripts | `knowledge_base.json` | 逐帧增量: `QwenTextClient.generate()` 逐帧更新知识库 JSON |

### 数据流

```
video.mp4
  │
  ├─ stage1 ─→ scenes.json (list[Scene]) + keyframes/*.jpg
  │
  ├─ extract_audio() ─→ .wav
  │
  ├─ stage2 ─→ transcript.json (list[TranscriptSegment])
  │            characters.json (list[Character])
  │            clip_vectors.npy
  │
  ├─ stage2.5 ─→ voiceprint_result.json (覆写 transcript.json 的 speaker_id)
  │
  ├─ stage3 ─→ captions.json, index.faiss, doc_store.json
  │            (补全 Scene.vlm_caption / structured_caption, 建 FAISS)
  │
  ├─ stage4 ─→ events.json (大事件 + 小事件)
  │
  └─ stage5 ─→ knowledge_base.json (层级结构: 阶段 > 事件 > summary)
```

### Checkpoint 机制

`Checkpoint` dataclass (`orchestrator.py`) 存储 `stages: dict[str, str]`，值为 `"pending"` / `"running"` / `"done"` / `"failed"`。

持久化到 `data/output/checkpoints/{video_id}.json`。`--resume` (默认) 时跳过 `done` 阶段，从断点恢复 `scene_transcripts` 等内存状态。

---

## 7. LLM 客户端

### BaseLLMClient (`vl/core/llm/base_client.py`)
- 底层: `openai.OpenAI` 指向 `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `chat(messages, model, temperature, stage)` → 统一调用 + 自动上报 `CostTracker`

### QwenTextClient (`vl/core/llm/qwen_text.py`) 继承 BaseLLMClient
- `generate(prompt, system, model, temperature, stage) -> str | None`
- `generate_json(prompt, system, model, stage) -> str | None` — 提取 JSON 块

### QwenVLClient (`vl/core/llm/qwen_vl.py`) 独立实现
- 底层: `dashscope.MultiModalConversation.call()` (不支持 OpenAI 兼容)
- `analyze_images(image_paths, prompt, window_size=4, stride=2)` — 滑动窗口多图分析
- 图片以 `file://{path}` 本地路径传入 (DashScope 自动上传)

### QwenOmni (`vl/asr/qwen_omni.py`) 独立实现
- 底层: `openai.OpenAI` streaming chat completions
- `understand_chunk(audio_path, image_paths, time_offset, chunk_index, time_end)` — 音频+图片多模态理解
- 音频 base64 编码为 `data:audio/wav;base64,...`，图片为 `data:image/jpeg;base64,...`
- 解析响应到 `OmniChunkResult` (含 `OmniSegment` 列表 + `OmniSceneDescription`)
- 默认 prompt 存在模块级常量 `STRUCTURED_PROMPT`，可被 `prompts.yaml` 的 `stage2_omni` 覆盖

---

## 8. 声纹识别 (`vl/voiceprint/`)

### VoiceprintClient (`client.py`)
- 鉴权: HMAC-SHA256 签名 (`_build_auth_url()`)
- API 端点: `https://api.xf-yun.com/v1/private/s1aa729d0`
- 音频要求: 16kHz/16bit/mono WAV, base64 编码, ≤ 4MB, 推荐 3-5 秒
- `search(file_path, top_k=1)` — 1:N 声纹搜索，返回最匹配的 `featureId` 和 `score`
- `search_score(feature_id, file_path)` — 1:1 声纹比对

### match_per_segment (`matcher.py`)
- 逐段匹配 (Method C): 过滤 < `min_duration` 的段 → `cut_audio_segment()` ffmpeg 切片 → `client.search()` → 映射 `featureId` → 替换 `TranscriptSegment.speaker_id`
- `featureId` 命名规则: `{pinyin}_{nn}` (如 `xiyangyang_01`)，去掉后缀后查 `name_mapping` 转中文
- 返回 `(segments, report_dict)`，report 含 `matched/skipped/failed` 统计

---

## 9. 提示词系统

所有模板在 `config/prompts.yaml`，通过 `load_prompt(config, key)` 加载。

| Key | 用途 | 模板变量 |
|-----|------|----------|
| `stage2_omni` | Qwen-Omni 音频理解 (system + user) | — |
| `stage2_character_extract` | 从台词提取角色名 (user only) | `{transcript_summary}`, `{existing_names}` |
| `scene_caption` | VLM 场景描述 (user only) | `{audiotext}`, `{caption}` |
| `event_builder` | 事件提取 (system + user) | `{frame_descriptions}` |
| `knowledge_summary` | 知识库增量更新 (system + user) | `{video_title}`, `{caption}`, `{audiotext}`, `{summary}` |
| `query_expand` | 搜索查询扩展 (system + user) | `{query}` |
| `rerank` | 搜索结果重排 (system + user) | `{query}`, `{scene_caption}`, `{transcript}` |
| `qa_answer` | 检索增强 QA (system + user) | `{question}`, `{context}` |
| `kb_qa` | 知识库 QA (system + user) | `{video_title}`, `{knowledge_base}`, `{question}` |

---

## 10. 向量检索 (`vl/store/vector_store.py` + `vl/services/search.py`)

**VectorStore**: 封装 `faiss.IndexFlatIP`
- `add(scene_id, embedding)` / `add_batch(scene_ids, embeddings)`
- `search(query_embedding, top_k) -> list[tuple[scene_id, score]]`
- `save(path)` / `load(path)` — 持久化 `.faiss` + `index_id_map.json`
- 中文路径 workaround: FAISS C++ 不支持中文路径，使用 `tempfile.mkstemp()` 中转

**search_scenes() 流程**:
1. CLIP 编码查询 → 可选 LLM 查询扩展 → 平均向量
2. 遍历 `stage3_captions/{video}/` 目录加载 FAISS 索引
3. 合并所有结果按 score 排序
4. 可选 LLM 重排 (每个结果打 1-10 分)

**doc_store.json 结构**: `[{scene_id, video_id, index, start_time, end_time, keyframe_paths, vlm_caption, transcript, structured_caption}]`

---

## 11. 代价追踪 (`vl/core/cost.py`)

**CostTracker** (全局单例 `get_cost_tracker()`):
- 每次 LLM 调用自动 `record(model, input_tokens, output_tokens, latency, stage)`
- 定价查找: 精确匹配 → 前缀匹配 (如 `qwen-plus-latest` 匹配 `qwen-plus`)
- 默认定价 (CNY/百万 tokens): qwen-plus 0.8/2.0, qwen-vl-max 3.0/9.0, qwen3.5-omni-plus 2.0/6.0
- 自定义覆盖: `pipeline.yaml` → `pricing:` 节 → `configure_pricing()`
- 报告: `report()` 格式化文本, `summary_by_model()`, `summary_by_stage()`

---

## 12. CLI 命令

入口: `python -m vl.app.main` (Typer app)

| 命令 | 函数 | 参数 | 说明 |
|------|------|------|------|
| `index <video>` | `index()` | `--genre/-g`, `--resume/--no-resume` | 跑完整流水线 |
| `search <query>` | `search()` | `--video/-v`, `--top-k/-k` | 语义搜索场景 |
| `qa <question>` | `qa()` | `--video/-v`, `--top-k/-k` | 视频问答 |
| `analyze <video>` | `analyze()` | `--type/-t` (summary/characters/timeline) | 视频分析 |
| `test-stage <video_id>` | `test_stage()` | `--stage/-s` (1-5) | 单独跑某个阶段 |

Poe 任务: `poe index`, `poe search`, `poe qa`, `poe analyze`, `poe test-stage`

---

## 13. 外部 API 依赖

| 服务 | 用途 | 认证 | 配置位置 |
|------|------|------|----------|
| 阿里 DashScope (Qwen) | 文本/视觉/多模态 LLM | `DASHSCOPE_API_KEY` | `.env` |
| 科大讯飞声纹 | 说话人识别 1:N | `XFYUN_APP_ID` + `API_KEY` + `API_SECRET` | `.env` |
| HuggingFace | CLIP 模型下载 | `HF_TOKEN`, `HF_ENDPOINT` (镜像) | `.env` |
| ffmpeg (系统) | 音频提取/切片 | — | PATH |

---

## 14. 关键实现细节

### 中文路径处理
- **FAISS**: `VectorStore.save()/load()` 用 `tempfile` 中转 (FAISS C++ 不支持中文路径)
- **OpenCV**: `FrameSampler` 用 `cv2.imencode() + tofile()` 替代 `cv2.imwrite()` (后者不支持中文路径)

### 滑动窗口 VLM (`QwenVLClient.analyze_images`)
- 窗口大小 `window_size=4`，步长 `stride=2`
- 每个窗口携带上一窗口的分析结果，指令"增量更新，修正错误，补充新信息"

### 知识库增量生成 (`stage5_knowledge.py`)
- 逐帧调用 LLM，将当前帧 caption + 台词 + 已有知识库传给 LLM
- LLM 返回完整更新后的 JSON (非增量片段)
- `content_type` 为 `"opening"` 或 `"ending"` 的帧跳过
- 输出层级: `{video_title}` → `阶段N: 名称` → `EN: {characters, action, goal, summary}`

### Stage2 音频分片
- 音频按 `asr_chunk_duration=120s` 分片
- 每个分片选最多 `asr_max_keyframes_per_chunk=5` 张均匀分布的关键帧
- Qwen-Omni 同时处理音频 + 图片，输出台词 + 说话人 + 场景描述

### Content Type 分类
- Qwen-Omni 为每个分片标注 `content_type`: `"opening"` / `"main"` / `"ending"`
- stage3/stage5 对非 `"main"` 内容有特殊处理

---

## 15. 输出文件格式

### `scenes.json` — `list[Scene.to_dict()]`
```json
[{"scene_id": "052 鸟蛋之争_s0", "video_id": "052 鸟蛋之争", "index": 0, "start_time": 0.0, "end_time": 5.2, ...}]
```

### `transcript.json` — `list[TranscriptSegment.to_dict()]`
```json
[{"segment_id": "seg_0", "scene_id": "052 鸟蛋之争_s0", "speaker_id": "喜羊羊", "text": "...", "start_time": 0.5, "end_time": 2.3}]
```

### `events.json`
```json
[{"event_id": "E1", "type": "major", "characters": ["灰太狼"], "action": "chase", "goal": "抓住喜羊羊", "start_scene": 1, "end_scene": 5, "summary": "...", "sub_events": [...]}]
```

### `knowledge_base.json`
```json
{"052 鸟蛋之争": {"阶段1: 灰太狼抓羊": {"E1": {"characters": [...], "action": "", "goal": "", "summary": ""}}}}
```

### `index.faiss` + `doc_store.json` + `index_id_map.json`
FAISS 索引 + 并行文档存储，用于 `search_scenes()` 语义检索。
