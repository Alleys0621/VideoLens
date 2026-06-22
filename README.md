# VideoLens

影视视频内容分析系统 — 基于多模态 AI 的视频音频处理、视觉理解与知识库生成。

## 架构概览

```
视频输入
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1: 音频处理                                        │
│  a) ffmpeg 提取 WAV                                       │
│  b) qwen3.5-omni-plus 检测片头/片尾曲                     │
│  c) qwen3.5-omni-flash ASR + 说话人轮次 + 情感 + 时间戳   │
│  d) 讯飞声纹 1:N 说话人识别                               │
│  → audio.json                                             │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2: 视觉处理                                        │
│  a) 声纹 segments → speaker anchors (midpoint/switch/     │
│     silence 三类，src/scene/speaker_anchor.py)            │
│  b) 每锚点抽 1 帧关键帧 (keyframes/)                      │
│  c) qwen3-vl-plus 分层 2-phase OCR                        │
│     · Phase 1: midpoint 中心帧 → 命中则结束               │
│     · Phase 2: 仅对 miss 抽 4 个邻域帧 [-.5,-.25,+.25,+.5]│
│  d) (可选) qwen-vl-max 视觉描述，默认冻结不重跑           │
│  → scenes.json + keyframes/ + visual.json                │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3: 结构化知识库（设计中）                          │
│  → knowledge.json                                         │
└──────────────────────────────────────────────────────────┘
```

**关键设计**：Stage 2 不再做基于视觉的场景检测（PySceneDetect/TransNetV2 仅作无 `audio.json` 时的回退路径），而是直接用 Stage 1 的声纹 segments 生成抽帧锚点，每个关键帧都绑定说话人 + 台词，下游 VLM 描述有强上下文；总采样量从 4000+ 帧降到 ~400 帧，4× 速度 + 4× 成本改善。

## 项目结构

```
VideoLens/
├── src/                           # 主代码包
│   ├── pipeline/                  # Pipeline 模块
│   │   ├── stage1_audio.py        #   Stage 1: 音频处理
│   │   ├── stage2_visual.py       #   Stage 2: 视觉处理 (speaker-anchor + tiered OCR)
│   │   ├── stage3_knowledge.py    #   Stage 3: 知识库 (占位)
│   │   └── orchestrator.py        #   Pipeline 编排器 (resolve_video_path / run_pipeline)
│   ├── core/                      # 核心基础设施
│   │   ├── config.py              #   配置管理 (frozen AppConfig 单例)
│   │   ├── cost.py                #   成本追踪 (分模态计价)
│   │   ├── logging.py             #   日志
│   │   ├── paths.py               #   路径管理
│   │   ├── llm/                   #   LLM 客户端
│   │   │   ├── base_client.py     #     基类 (_report_usage 分模态计价)
│   │   │   ├── qwen_text.py       #     文本 LLM
│   │   │   └── qwen_vl.py         #     视觉 LLM
│   │   ├── helpers/               #   工具函数
│   │   │   ├── ffmpeg.py
│   │   │   ├── json_utils.py      #     save_json (统一编码)
│   │   │   ├── text_utils.py
│   │   │   └── prompt_loader.py
│   │   └── models/                #   数据模型
│   │       ├── scene.py
│   │       └── transcript.py
│   ├── scene/                     # 场景/锚点生成
│   │   ├── detector.py            #   PySceneDetect 封装 (回退路径)
│   │   ├── transnet_detector.py   #   TransNetV2 ONNX (回退路径)
│   │   └── speaker_anchor.py      #   ★ 声纹锚点生成 (Stage 2 主路径)
│   ├── voiceprint/                # 讯飞声纹
│   │   ├── client.py
│   │   └── matcher.py
│   ├── eval/                      # 评估工具 (不进生产 pipeline)
│   │   ├── reference_asr.py       #   omni-plus 全集参考 ASR
│   │   └── ocr_accuracy.py        #   OCR 准确率评估器
│   └── app/                       # CLI 入口
│       ├── cli.py
│       └── main.py
├── config/
│   ├── pipeline.yaml              # Pipeline 参数 + 模型选择 + pricing
│   └── prompts.yaml               # Prompt 模板
├── tests/                         # 评估脚本
│   ├── eval_audio.py              #   Stage 1 声纹 GT 评估
│   └── batch_test_ocr.py          #   Stage 2 OCR 批量测试编排器
├── experiments/                   # 实验记录 (每次跑批的 README + JSON 汇总)
├── data/
│   ├── videos/{作品名}/{剧集}.{mp4,mkv}   # 输入视频
│   ├── gt/*.json                          # 人工 Ground Truth
│   └── output/{video_dir}/                # 流水线产物
├── models/transnetv2.onnx         # ONNX 权重 (已 gitignore)
├── .env                           # 讯飞凭证
└── requirements.txt
```

## 快速开始

### 环境要求

- Python 3.12.x（必须 `<3.13`）
- ffmpeg（系统已安装）
- 阿里云 DashScope API Key（系统环境变量 `DASHSCOPE_API_KEY`）
- 讯飞声纹 API 凭证（写入 `.env` 的 `XFYUN_*` 字段）

### 安装

```bash
uv venv .venv
.venv\Scripts\activate    # Windows
uv pip install -r requirements.txt
```

### 运行

```bash
# 运行完整 Pipeline
python -m src.app.main run "052 鸟蛋之争"
python -m src.app.main run "家有儿女/第一季/第01集"

# 只运行 Stage 1 (音频) 或 Stage 2 (视觉)
python -m src.app.main run "052 鸟蛋之争" --stage 1

# 跳过片头/片尾曲检测
python -m src.app.main run "052 鸟蛋之争" --skip-theme

# 自定义 chunk 时长
python -m src.app.main run "052 鸟蛋之争" --chunk 90

# 声纹置信度阈值，低于此值标为「路人」
python -m src.app.main run "052 鸟蛋之争" --vp-threshold 0.4
```

### 评估

```bash
# Stage 1 声纹评估 (需要人工 GT)
python -m tests.eval_audio --video "052 鸟蛋之争" --save

# Stage 2 OCR 批量评估 (用 omni-plus 参考 ASR 作为软 GT)
python -m tests.batch_test_ocr --episodes "S1E1,S1E2" --workers 3
python -m tests.batch_test_ocr --only-eval                # 仅重算评估
python -m tests.batch_test_ocr --skip-stage1              # 已有 audio.json 时跳过 Stage 1
```

## Stage 详解

### Stage 1: 音频处理

| 步骤 | 模型 | 功能 |
|------|------|------|
| 1a | ffmpeg | 提取 16kHz/16bit/mono WAV 音频 |
| 1b | qwen3.5-omni-plus | 片头/片尾曲检测 |
| 1c | ffmpeg | 静音检测 + 预处理 + 智能切块 |
| 1d | qwen3.5-omni-flash | ASR + 说话人轮次 + 情感 + 时间戳 |
| 1e | 讯飞声纹 1:N | 说话人识别 |

**输出:** `audio.json`

```json
{
  "theme_songs": {"opening": {...}, "ending": {...}},
  "content_range": {"start": 85.2, "end": 781.4},
  "segments": [
    {"segment_id": "...", "begin_time": 85.2, "end_time": 88.5,
     "speaker_pred": "美羊羊", "vp_score": 0.74,
     "text": "...", "emotion": "关切语调"}
  ]
}
```

### Stage 2: 视觉处理

| 步骤 | 模块 | 功能 |
|------|------|------|
| 2a | `src/scene/speaker_anchor.py` | 把声纹 segments 转 speaker anchors (midpoint/switch/silence) |
| 2b | OpenCV | 每锚点抽 1 帧，写入 `keyframes/` |
| 2c | qwen3-vl-plus | Phase 1 OCR：midpoint 中心帧 |
| 2d | qwen3-vl-plus | Phase 2 OCR：仅对 miss 抽邻域 4 帧 |
| 2e | qwen-vl-max | (可选) 视觉描述，默认 `skip_captions=True` |

**输出:** `scenes.json`、`keyframes/`、`visual.json`

`scenes.json` 中的每个 scene 携带：`anchor_type`（midpoint/switch/silence）、`anchor_timestamp`、`speaker`、`segment_id`、`anchor_text` 等。`visual.json` 的 `ocr` 是 `{index: 字幕文字或"无字幕"}` 字典。

### Stage 3: 结构化知识库

占位实现，待设计。

## AI 模型依赖

Omni 系列按输入模态分别计价（文本/音频/图片各有独立费率），输出按文本输出计。

| 模型 | 用途 | 文本输入 | 音频输入 | 图片输入 | 文本输出 |
|------|------|---------|---------|----------|---------|
| qwen3.5-omni-plus | 片头/片尾曲检测 / 评估参考 ASR | 7.0 | 53.0 | 40.0 | 213.0 |
| qwen3.5-omni-flash | Stage 1 ASR + 情感 + 时间戳 | 2.2 | 18.0 | 13.3 | 72.0 |
| qwen3-vl-plus | ★ Stage 2 字幕 OCR | 1.4 | — | 1.4 | 11.2 |
| qwen-vl-max | Stage 2 视觉描述（可选） | 3.0 | — | 3.0 | 9.0 |
| 讯飞声纹 | Stage 1 说话人识别 | — | — | — | 按次计费 |

单位均为 ¥/百万 Token，价格来源 [阿里云百炼模型价格](https://help.aliyun.com/zh/model-studio/model-pricing)。

成本追踪：`src/core/cost.py` 的 `CostTracker` 通过 DashScope 响应中的 `usage.prompt_tokens_details.{text,audio}_tokens` 自动按模态归价。

**典型单集成本**（《家有儿女》22 min）：
- Stage 1（flash + 讯飞）: ~1.5 元
- Stage 2 OCR（444 anchors × tiered 2-phase）: ~1.2 元
- Stage 2 Caption（可选）: ~2.5 元
- **生产单集合计**: ~2.8 元（不含 caption）
- 评估专用 omni-plus 参考 ASR: ~5 元（仅做评估时跑，生产不需要）

## 评估指标

OCR 评估没有人工 GT，采用 omni-plus 全集 ASR 作为软 Ground Truth。指标定义见 `src/eval/ocr_accuracy.py`：

- **hit_rate (非静默)**：非 silence 锚点中 OCR 命中（非「无字幕」）的比例。**目标 ≥95%**。
- **recall (参考段覆盖)**：reference_asr.json 的句子中有多少被某 OCR anchor 命中。结构上偏低（一个 utterance 只采 1 帧，而 omni-plus 句子级切分粒度更细）。
- **char P/R/F1**：OCR 文本与匹配参考段的字符级 precision/recall/F1。
- **miss_candidates**：OCR=无字幕 但参考 ASR 该时刻有人声的「真漏」数量。

7 集《家有儿女》批量测试结果（详见 `experiments/ocr_batch_家有儿女7集/README.md`）：

| 集数 | hit_rate (非静默) | recall | char F1 | 真漏 |
|------|:-----------------:|:------:|:-------:|:----:|
| S1E1 | 95.08% | 49.2% | 81.3% | 14 |
| S1E2 | 95.02% | 46.5% | 81.7% | 12 |
| S1E3 | 95.29% | 36.4% | 77.7% | 11 |
| S2E2 | 95.87% | 50.6% | 75.3% | 16 |
| S2E3 | 98.37% | 56.4% | 71.6% | 6 |
| S2E4 | 94.34% | 44.4% | 78.4% | 11 |
| S2E5 | 97.53% | 54.5% | 77.9% | 9 |
| **均值** | **96.02%** | **48.3%** | **77.7%** | **11.3** |

## License

MIT
