# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库工作时提供指引。

**重要约定：与用户的所有交流一律使用中文回复，不要切换到英文。**

## 项目概述

VideoLens 是一个面向情景类剧集的多模态 AI 流水线。视频依次经过三个阶段（音频、视觉、知识库），每个阶段产出的 JSON 制品会被下一阶段消费。

流水线主要面向中文剧集（如《家有儿女》《爱情公寓》），依赖阿里云 DashScope（通义千问）和讯飞声纹 API。

## 常用命令

项目使用 `uv` 管理依赖，Python 版本要求 `3.12.x`（必须 `<3.13`）。所有模块都通过项目虚拟环境调用。

```bash
# 环境初始化
uv venv .venv
.venv\Scripts\activate           # Windows
uv pip install -r requirements.txt

# 运行完整流水线
python -m src.app.main run "052 鸟蛋之争"
python -m src.app.main run "家有儿女/第一季/第01集"

# 只运行单个阶段（1=音频，2=视觉，3=知识库）
python -m src.app.main run "052 鸟蛋之争" --stage 2

# 常用参数
--skip-theme        # 跳过片头/片尾曲检测
--chunk 90          # Omni 切块时长（秒）
--vp-threshold 0.4  # 声纹置信度阈值，低于此值标为「路人」

# 用 Ground Truth 评估 Stage 1 的声纹归属
python -m tests.eval_audio --video "052 鸟蛋之争" --save

# 批量测试 OCR（含 omni-plus 参考 ASR，无 GT 时的软评估）
python -m tests.batch_test_ocr --episodes "S1E1,S1E2" --workers 3
python -m tests.batch_test_ocr --only-eval            # 只重算评估（前面都跑过）
```

`tests/` 目录下是评估脚本（目前没有 pytest 单元测试）。流水线产物落到 `data/output/{video_dir}/`，评估报告落到 `tests/results/` 与 `data/output/_batch_reports/`，实验记录落到 `experiments/`。

## 架构

### 三阶段流水线（`src/pipeline/`）

`orchestrator.py` 是唯一入口 —— `cli.py` 调用 `run_pipeline()`，后者依据 `--stage` 参数条件性触发各阶段：

1. **`stage1_audio.py`**（约 700 行，单体文件）—— 通过 ffmpeg 提取 WAV，用 `qwen3.5-omni-plus` 检测片头/片尾曲，做静音裁剪与智能切块，调用 `qwen3.5-omni-flash` 完成 ASR + 说话人轮次 + 情感 + 时间戳，最后用讯飞声纹 1:N 匹配说话人。产出 `audio.json`，其中 `segments[]` 携带 `speaker_pred`、`vp_score`、`emotion`、`text`。
2. **`stage2_visual.py`** —— 以 Stage 1 的声纹 segments 为驱动生成 speaker anchors（midpoint / switch / silence 三类），每锚点抽 1 帧，调用 `qwen3-vl-plus` 做 tiered 2-phase OCR（Phase 1 中心帧 → Phase 2 仅对 miss 抽 4 个邻域帧），可选地调用 `qwen-vl-max` 生成视觉描述（caption，已默认冻结不重跑）。当无 Stage 1 数据时，回退到 `src/scene/detector` (PySceneDetect) 或 `src/scene/transnet_detector` (TransNetV2) 做场景检测。产出 `scenes.json`、`keyframes/`、`visual.json`。
3. **`stage3_knowledge.py`** —— 占位文件；结构化知识库生成仍在设计中。

Stage 2 强依赖 Stage 1 的 `audio.json` 做声纹锚点生成，因此阶段顺序有意义——单独运行 Stage 2 时若没有 `audio.json` 会回退到 SBD 全扫描路径（成本高、且无 speaker_pred 上下文）。

### Speaker Anchor 抽帧策略（Stage 2 核心）

`src/scene/speaker_anchor.py` 把声纹 segments 转成抽帧锚点：
- **midpoint**：每段中点，作为该 utterance 字幕的主采样点（最小段长 0.3s）
- **switch**：说话人切换处 + 0.3s 偏移，捕获反应镜头（默认开启，但当前 OCR 默认让 switch 继承同 segment_id 的 midpoint 结果，不单独采样）
- **silence**：>3s 静默段中点，做兜底（默认直接标为「无字幕」，零成本）
- 去重最小间隔 0.5s，优先级 midpoint > switch > silence

这是替代「TransNetV2 8 帧/场景 → 过滤 → 取 1 帧」的方案，理由：每帧都绑定 speaker_pred + 台词文本，下游 VLM 描述有强上下文；总采样量从 4000+ 降到 400 左右，4× 速度提升 + 4× 成本下降。

### 配置流转

- `src/core/config.py` 把 `config/pipeline.yaml` 和 `config/prompts.yaml` 加载到一个 frozen 的 `AppConfig` 数据类。项目根目录通过向上扫描 `pyproject.toml`（要求包含 `name = "videolens"`）自动定位。
- `get_config()` 是模块级单例，禁止直接构造 `AppConfig`。
- **声纹组按 `data/videos/` 下的影视作品子目录名作为键**（如 `喜羊羊与灰太狼`、`家有儿女`）。`orchestrator.py::get_show_name()` 从视频目录参数推断作品名，`load_voiceprint_config()` 再从 `pipeline.yaml` 的 `voiceprint_groups:` 中取 `group_id` 和 `name_mapping`。若没有匹配项，声纹识别会被静默跳过。

### 视频路径解析

`resolve_video_path()` 支持三类视频格式（`.mp4 / .mkv / .mov / .avi`），按优先级匹配：直接子路径 → 一级子目录 → 二级子目录（用于 `家有儿女/第一季/第01集.mkv` 这类季结构）。流水线写入 `data/output/{video_dir}/`——传入的 `video_dir` 字符串需要和输入时一致。

### LLM 客户端（`src/core/llm/`）

所有通义千问 API 调用都经由 `QwenVLClient`（视觉）或 `QwenTextClient`（文本），二者都构建在 `base_client.py` 之上。它们要求 `DASHSCOPE_API_KEY` 作为**系统环境变量**（不写入 `.env`）。讯飞凭证（`XFYUN_*`）则放在 `.env` 中。

### 成本追踪

`src/core/cost.py` 按 `pipeline.yaml` 的 `pricing:` 块计价，累计每次调用的 token 使用量。各阶段在调用 LLM 时传入 `stage=` 标签，便于把成本归属回具体阶段。

**Omni 系列按输入模态分档计价**（文本/音频/图片各有独立费率，输出按文本输出计）。`BaseLLMClient.chat()` 和 stage1 的两处 omni stream 调用都会从 DashScope 响应里的 `usage.prompt_tokens_details.{text,audio}_tokens` 拆分出各模态 token 后上报。新增 omni/多模态调用时，不要把 `prompt_tokens` 当作单一输入价计算 —— 必须走 `CostTracker.record()` 的分模态参数，或直接用 `src.core.llm.base_client._report_usage()`。`qwen3.5-omni-{plus,flash}` 在 DEFAULT_PRICING 中已有正确的官方价（中国内地，元/百万 Token）：text_input 7/2.2，audio_input 53/18，image_input 40/13.3，text_output 213/72。`qwen3-vl-{plus,flash}` 在 DEFAULT_PRICING 中也已配置（text/image input 同价，plus 1.4 / flash 0.6）。

### 评估工具（`src/eval/`）

OCR 评估因为缺人工 GT，采用 omni-plus 全集 ASR 作为软 Ground Truth：

- **`reference_asr.py`**：调用 `qwen3.5-omni-plus` 对全集音频重新跑高准确率逐句 ASR（chunk 60s，并行 4 路），输出 `reference_asr.json`。Stage 1 的 flash ASR 主要目的是说话人轮次切分，文本本身有 5-10% 错误率，不能用作 OCR 准确率参考。
- **`ocr_accuracy.py`**：对比 `visual.json` 的 OCR 结果与 `reference_asr.json`，通过「时间重叠 ±1.5s + 字符级重叠阈值 0.5」双重匹配，计算 hit_rate / recall / 字符级 P/R/F1 / miss_candidates（OCR=无字幕 但参考有人声的「真漏」）。输出 `ocr_eval.json`。

评估模块**不进生产 pipeline**，只在 `tests/batch_test_ocr.py` 中编排。批量测试支持集间并行（默认 3 路），单集流程：Stage 1 → Stage 2 → reference ASR → evaluate。

## 约定

- **包前缀统一是 `src.`**（如 `from src.core.config import get_config`）。
- 各阶段 JSON 输出统一用 `src/core/helpers/json_utils.py::save_json` 写入，保持编码一致。
- 场景检测器（`src/scene/`）实现 `detect_scenes()` 并返回 `Scene` 数据类实例（`src/core/models/scene.py`）；TransNetV2 另外提供 `release()`，用于在关键帧抽取前释放 ONNX 内存。
- `src/scene/speaker_anchor.py` 是 Stage 2 的主路径，PySceneDetect/TransNetV2 是无 audio 时的回退路径。
- Windows 上 `cv2.imwrite` 对中文路径会失败，统一用 `src/pipeline/stage2_visual.py::_imwrite_unicode`（imencode + tofile）。
- 所有系统输出均为中文；面向用户的字符串、日志、评估报告都使用中文。

## 数据布局（已 gitignore）

```
data/
├── videos/{作品名}/{剧集}.{mp4,mkv}   # 输入视频（不入库）
├── gt/*.json                          # 评估用人工 Ground Truth
└── output/{video_dir}/                # 每个视频的流水线产物
    ├── audio.json
    ├── scenes.json, visual.json
    ├── keyframes/*.jpg                # speaker-anchor 抽帧
    ├── reference_asr.json             # (评估时) omni-plus 全集 ASR
    └── ocr_eval.json                  # (评估时) OCR 准确率报告
models/transnetv2.onnx                 # ONNX 权重，已 gitignore
experiments/                           # 实验记录（每次跑批的 README + JSON 汇总）
data/output/_batch_reports/            # 批量测试日志与跨集汇总
```
