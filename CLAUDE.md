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
--vp-threshold 0    # 关闭声纹阈值过滤（默认 0.4, 低于此值统一标为「路人」)

# 用 Ground Truth 评估 Stage 1 的声纹归属
python -m tests.eval_audio --video "052 鸟蛋之争" --save

# 批量测试 OCR（含 omni-plus 参考 ASR，无 GT 时的软评估）
python -m tests.batch_test_ocr --episodes "S1E1,S1E2" --workers 3
python -m tests.batch_test_ocr --only-eval            # 只重算评估（前面都跑过）

# Stage 3 单集完整评估（建库 P1+P2 + 建库评估 + 用库 BM25 检索）
python -m scripts.stage3_eval --video "家有儿女/第二季/第001集"
python -m scripts.stage3_eval --video "家有儿女/第二季/第001集" --redo-dryrun  # 强制重跑建库
python -m scripts.stage3_eval --video "家有儿女/第二季/第001集" --reuse-queries # 复用已有 query 集

# Stage 3 单独跑 P1+P2（建库）或 P3+P4+P5（完整 KB）
python -m scripts.stage3_p1p2 --video "家有儿女/第二季/第001集" --save
python -m scripts.stage3_p345 --video "家有儿女/第二季/第001集"
```

`tests/` 目录下是评估脚本（目前没有 pytest 单元测试）。流水线产物落到 `data/output/{video_dir}/`，评估报告落到 `tests/results/` 与 `data/output/_batch_reports/`，实验记录落到 `experiments/`。

## 架构

### 三阶段流水线（`src/pipeline/`）

`orchestrator.py` 是唯一入口 —— `cli.py` 调用 `run_pipeline()`，后者依据 `--stage` 参数条件性触发各阶段：

1. **`stage1_audio.py`**（约 700 行，单体文件）—— 通过 ffmpeg 提取 WAV，用 `qwen3.5-omni-plus` 检测片头/片尾曲，做静音裁剪与智能切块，调用 `qwen3.5-omni-flash` 完成 ASR + 说话人轮次 + 情感 + 时间戳，最后用讯飞声纹 1:N 匹配说话人。产出 `audio.json`，其中 `segments[]` 携带 `speaker_pred`、`vp_score`、`emotion`、`text`。
2. **`stage2_visual.py`** —— 以 Stage 1 的声纹 segments 为驱动生成 speaker anchors（midpoint / switch / silence 三类），每锚点抽 1 帧，调用 `qwen3-vl-plus` 做 tiered 2-phase OCR（Phase 1 中心帧 → Phase 2 仅对 miss 抽 4 个邻域帧），可选地调用 `qwen-vl-max` 生成视觉描述（caption，已默认冻结不重跑）。当无 Stage 1 数据时，回退到 `src/scene/transnet_detector` (TransNetV2) 做场景检测。产出 `scenes.json`、`keyframes/`、`visual.json`。
3. **`stage3/`**（子包）—— Stage 3 完整 P1-P6 实现，核心逻辑在 `src/pipeline/stage3/`：
   - `p1_p2_actions.py`：P1 Action 抽取 + P2 Event 聚合（`run_p1p2()`，用 `qwen3.7-plus`）
   - `p345_kb.py`：P3 PlotArc + P4 Video 摘要 + P5 Global 角色/Arc（`run_p345()`，跨集累积到 `data/output/_global/`）
   - `p6_profile.py`：P6 角色深度画像（`run_p6()`，all-in-one 单次调用，产出性格 6 维 + 行为模式带例证）
   - `llm_json.py`：LLM JSON 调用统一封装（重试 + JSON 闭合修复）
   - `__init__.py::run_stage3()`：P1→P5 完整编排（**P6 独立跑**，不在 run_stage3 内），供 `orchestrator.run_pipeline` 调用
   
   评估在 `src/eval/stage3_build_quality.py`（P1/P2 层）+ `stage3_kb_quality.py`（P3-P5 层）+ `stage3_retrieval.py`（用库 BM25）+ `stage3_profile_quality.py`（P6 画像质量），公共工具在 `_quality_utils.py`。CLI 包装：`scripts/stage3_p1p2.py` / `scripts/stage3_p345.py` / `scripts/stage3_p6.py`，编排入口 `scripts/stage3_eval.py`。**P1/P2 prompt 已打板**（字段完整率 100% / Schema 0 违反 / 用库 Recall@5=0.98 / MRR=0.95）。

### Stage 3 内部架构（P1-P6）

```
Perception (Stage 1/2)
      ↓
Action Layer (P1)        — src/pipeline/stage3/p1_p2_actions.py::call_p1
  · 输入: audio_segment + scene + ocr + caption (batch_size=5)
  · 输出: 11 类 Communicative Action + evidence (绑定 keyframe + segment_id)
  · target 字段留空 (P2 推理)
      ↓
Event Layer (P2)         — src/pipeline/stage3/p1_p2_actions.py::call_p2
  · 输入: 同集按时间序排列的若干 Action (batch_size=8)
  · 输出: Event (含 participants / motivation / motivation_confidence /
            outcome / summary / retrieval_text / keywords + actions[].target)
  · A→B→A 模式强制 target 推理 (允许空: 独白 / 群体广播 / 旁白)
      ↓
PlotArc (P3)             — src/pipeline/stage3/p345_kb.py::run_p3
Video 摘要 (P4)          — src/pipeline/stage3/p345_kb.py::run_p4
Global 角色/Arc (P5)     — src/pipeline/stage3/p345_kb.py::run_p5
  · 跨集累积到 data/output/_global/{characters,global_arcs,video_summaries}.json
      ↓ (独立跑, 不在 run_stage3; 验证质量后再决定是否进主流程)
角色深度画像 (P6)       — src/pipeline/stage3/p6_profile.py::run_p6
  · 输入: characters.json + 所有集 events + actions (all-in-one 单次调用)
  · 输出: data/output/_global/character_profiles.json (personality 6 维 + 行为模式带 event/action 例证)
  · 用 name+aliases 匹配 (不依赖 events.participants, 该字段归一不稳定)
```

**11 类 Action 枚举**（`config/prompts.yaml::stage3_p1_action_extract`）：`inform / ask / answer / command / refuse / promise / threaten / deceive / argue / invite / react`。关键硬约束：`ask > react`（疑问句即使带情绪也归 ask），react 仅限不索取信息的纯感叹。

Stage 2 强依赖 Stage 1 的 `audio.json` 做声纹锚点生成，因此阶段顺序有意义——单独运行 Stage 2 时若没有 `audio.json` 会回退到 TransNetV2 全扫描路径（成本高、且无 speaker_pred 上下文）。

### Speaker Anchor 抽帧策略（Stage 2 核心）

`src/pipeline/speaker_anchor.py` 把声纹 segments 转成抽帧锚点：
- **midpoint**：每段中点，作为该 utterance 字幕的主采样点（最小段长 0.3s）
- **switch**：说话人切换处 + 0.3s 偏移，捕获反应镜头（默认开启，但当前 OCR 默认让 switch 继承同 segment_id 的 midpoint 结果，不单独采样）
- **silence**：>3s 静默段中点，做兜底（默认直接标为「无字幕」，零成本）
- 去重最小间隔 0.5s，优先级 midpoint > switch > silence

这是替代「TransNetV2 8 帧/场景 → 过滤 → 取 1 帧」的方案，理由：每帧都绑定 speaker_pred + 台词文本，下游 VLM 描述有强上下文；总采样量从 4000+ 降到 400 左右，4× 速度提升 + 4× 成本下降。

### 配置流转

- `src/core/config.py` 把 `config/pipeline.yaml` 和 `config/prompts.yaml` 加载到一个 frozen 的 `AppConfig` 数据类。项目根目录通过向上扫描 `pyproject.toml`（要求包含 `name = "videolens"`）自动定位。
- `get_config()` 是模块级单例，禁止直接构造 `AppConfig`。
- **声纹组按 `data/videos/` 下的影视作品子目录名作为键**（如 `喜羊羊与灰太狼`、`家有儿女`）。`src/core/path_utils.py::get_show_name()` 从视频目录参数推断作品名，`load_voiceprint_config()` 再从 `pipeline.yaml` 的 `voiceprint_groups:` 中取 `group_id` 和 `name_mapping`。若没有匹配项，声纹识别会被静默跳过。

### 视频路径解析

`src/core/path_utils.py::resolve_video_path()` 支持三类视频格式（`.mp4 / .mkv / .mov / .avi`），按优先级匹配：直接子路径 → 一级子目录 → 二级子目录（用于 `家有儿女/第一季/第01集.mkv` 这类季结构）。流水线写入 `data/output/{video_dir}/`——传入的 `video_dir` 字符串需要和输入时一致。

### LLM 客户端（`src/core/llm/`）

所有通义千问 API 调用都经由 `QwenVLClient`（视觉）或 `QwenTextClient`（文本），二者都构建在 `base_client.py` 之上。它们要求 `DASHSCOPE_API_KEY` 作为**系统环境变量**（不写入 `.env`）。讯飞凭证（`XFYUN_*`）则放在 `.env` 中。

**模型分工**：
- `qwen3.5-omni-plus`：Stage 1 片头/片尾检测 + 评估参考 ASR
- `qwen3.5-omni-flash`：Stage 1 ASR + 说话人轮次 + 情感 + 时间戳
- `qwen3-vl-plus`：Stage 2 字幕 OCR（tiered 2-phase）
- `qwen-vl-max`：Stage 2 视觉描述（可选，默认冻结）
- `qwen3.7-plus`：Stage 3 P1-P5 文本生成（默认 `enable_thinking=False` 控成本；关键集可开 thinking 提字段/Schema 质量，但 token 涨 3-4 倍）

`base_client.py::chat()` 的 `enable_thinking` 参数通过 DashScope `extra_body` 控制，旧模型（omni/vl 系列）会忽略此参数，无副作用。

### 成本追踪

`src/core/cost.py` 按 `pipeline.yaml` 的 `pricing:` 块计价，累计每次调用的 token 使用量。各阶段在调用 LLM 时传入 `stage=` 标签，便于把成本归属回具体阶段。

**Omni 系列按输入模态分档计价**（文本/音频/图片各有独立费率，输出按文本输出计）。`BaseLLMClient.chat()` 和 stage1 的两处 omni stream 调用都会从 DashScope 响应里的 `usage.prompt_tokens_details.{text,audio}_tokens` 拆分出各模态 token 后上报。新增 omni/多模态调用时，不要把 `prompt_tokens` 当作单一输入价计算 —— 必须走 `CostTracker.record()` 的分模态参数，或直接用 `src.core.llm.base_client._report_usage()`。`qwen3.5-omni-{plus,flash}` 在 DEFAULT_PRICING 中已有正确的官方价（中国内地，元/百万 Token）：text_input 7/2.2，audio_input 53/18，image_input 40/13.3，text_output 213/72。`qwen3-vl-{plus,flash}` 在 DEFAULT_PRICING 中也已配置（text/image input 同价，plus 1.4 / flash 0.6）。

### 评估工具（`src/eval/`）

OCR 评估因为缺人工 GT，采用 omni-plus 全集 ASR 作为软 Ground Truth：

- **`reference_asr.py`**：调用 `qwen3.5-omni-plus` 对全集音频重新跑高准确率逐句 ASR（chunk 60s，并行 4 路），输出 `reference_asr.json`。Stage 1 的 flash ASR 主要目的是说话人轮次切分，文本本身有 5-10% 错误率，不能用作 OCR 准确率参考。
- **`ocr_accuracy.py`**：对比 `visual.json` 的 OCR 结果与 `reference_asr.json`，通过「时间重叠 ±1.5s + 字符级重叠阈值 0.5」双重匹配，计算 hit_rate / recall / 字符级 P/R/F1 / miss_candidates（OCR=无字幕 但参考有人声的「真漏」）。输出 `ocr_eval.json`。

Stage 3 评估分**建库**和**用库**两个维度：

- **`stage3_build_quality.py`**：宽松校验（Action 必填 `actor/action/utterance`；Event 必填 `title/motivation/summary/retrieval_text`）+ Schema 合规（action 枚举 / motivation_confidence 枚举 / target 类型）+ 软质量（`vp_low_confidence_ratio` / `char_unknown_event_ratio` / `empty_target_link_ratio` / `event_with_empty_actions_ratio`）+ 覆盖率。输出 `build_quality_report.json`。
- **`stage3_retrieval.py`**：LLM 自生成 query（每个 event 反向生成 1 个自然问句）+ 自实现 BM25（jieba 分词，~30 行）+ Recall@1/3/5/10 + MRR。输出 `retrieval_queries.json` + `retrieval_eval.json`。

所有评估模块**不进生产 pipeline**，只在 `tests/batch_test_ocr.py`（Stage 1/2）或 `scripts/stage3_eval.py`（Stage 3）中编排。

## 约定

- **包前缀统一是 `src.`**（如 `from src.core.config import get_config`）。
- 各阶段 JSON 输出统一用 `src/core/helpers/json_utils.py::save_json` 写入，保持编码一致。
- 场景检测器（`src/scene/`）实现 `detect_scenes()` 并返回 `Scene` 数据类实例（`src/core/models/scene.py`）；TransNetV2 另外提供 `release()`，用于在关键帧抽取前释放 ONNX 内存。
- `src/pipeline/speaker_anchor.py` 是 Stage 2 的主路径，TransNetV2 是无 audio 时的回退路径。
- Windows 上 `cv2.imwrite` 对中文路径会失败，统一用 `src/pipeline/stage2_visual.py::_imwrite_unicode`（imencode + tofile）。
- 所有系统输出均为中文；面向用户的字符串、日志、评估报告都使用中文。

## 数据布局（已 gitignore）

```
data/
├── videos/{作品名}/{剧集}.{mp4,mkv}   # 输入视频（不入库）
├── gt/*.json                          # 评估用人工 Ground Truth
└── output/{video_dir}/                # 每个视频的流水线产物
    ├── audio.json                     # Stage 1 产物 (声纹 segments)
    ├── scenes.json, visual.json       # Stage 2 产物 (scenes + ocr + captions)
    ├── keyframes/*.jpg                # speaker-anchor 抽帧
    ├── stage3_dryrun.json             # Stage 3 P1+P2 产物 (actions + events + cost)
    ├── build_quality_report.json      # (评估时) Stage 3 建库字段完整性
    ├── retrieval_queries.json         # (评估时) LLM 自生成 query 集
    ├── retrieval_eval.json            # (评估时) Stage 3 用库 Recall@K / MRR
    ├── stage3_eval_summary.json       # (评估时) 三份报告汇总
    ├── reference_asr.json             # (评估时) omni-plus 全集 ASR
    └── ocr_eval.json                  # (评估时) OCR 准确率报告
data/output/_global/                   # 跨集全局产物 (P3-P6 累积)
  ├── characters.json                 # P5 全局角色表
  ├── global_arcs.json                # P5 全局剧情弧
  ├── video_summaries.json            # P4 跨集视频摘要
  ├── character_profiles.json         # P6 角色深度画像 (性格 6 维 + 行为模式)
  └── profile_quality_report.json     # (评估时) P6 画像质量
models/transnetv2.onnx                 # ONNX 权重，已 gitignore
experiments/                           # 实验记录（_archive/ 是历史归档）
data/output/_batch_reports/            # 批量测试日志与跨集汇总
```

## 已搁置 / 已删的实验路线

- **Stage 1.5 LLM 说话人修正**：已删（`src/eval/speaker_correction.py` + `prompts.yaml::stage1_5_speaker_correct`）。两版实验均触顶（保守策略 + thinking + batch 24 后修正后 47.77% vs baseline 46.5%，仅 +1.27%），信息论天花板限制，纯文本方法不可行。回上游优化方向（声纹库扩展 / omni-plus diarization）。
- **Stage 1 配置块**：已删（`theme_window / chunk_duration / silence_db / silence_remove_min / speech_remove_max` 全是死配置，业务代码不读 config 走 cli 参数）。
- **死代码模块**：`src/core/paths.py` / `src/core/helpers/ffmpeg.py` / `src/core/helpers/scene_utils.py` / `src/voiceprint/matcher.py` / `src/core/models/transcript.py` 均已删。
- **实验脚本归档**：`experiments/llm_naming_no_cluster.py` + `speaker_clustering_test.py`（Stage 1.5 路线残留）已移到 `experiments/_archive/`。
