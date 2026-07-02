# VideoLens

影视视频内容分析系统 — 基于多模态 AI 的视频音频处理、视觉理解与结构化知识库生成。

## 架构概览

```
视频输入
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1: 音频处理 (src/pipeline/stage1_audio.py)         │
│  a) ffmpeg 提取 WAV                                       │
│  b) qwen3.5-omni-plus 检测片头/片尾曲                     │
│  c) qwen3.5-omni-flash ASR + 说话人轮次 + 情感 + 时间戳   │
│  d) 讯飞声纹 1:N 说话人识别                               │
│  → audio.json                                             │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2: 视觉处理 (src/pipeline/stage2_visual.py)        │
│  a) 声纹 segments → speaker anchors (midpoint/switch/     │
│     silence 三类，src/pipeline/speaker_anchor.py)         │
│  b) 每锚点抽 1 帧关键帧 (keyframes/)                      │
│  c) qwen3-vl-plus 分层 2-phase OCR                        │
│     · Phase 1: midpoint 中心帧 → 命中则结束               │
│     · Phase 2: 仅对 miss 抽 4 个邻域帧 [-.5,-.25,+.25,+.5]│
│  d) (可选) qwen-vl-max 视觉描述，默认冻结不重跑           │
│  → scenes.json + keyframes/ + visual.json                 │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3: 结构化知识库 (src/pipeline/stage3/)             │
│  P1: qwen3.7-plus 批量抽 Action                           │
│      · actor / action_type / utterance / target / emotion │
│      · 11 类 Communicative Action 枚举                    │
│      · evidence 绑定 segment_id + keyframe + caption      │
│  P2: qwen3.7-plus 聚合 Event                              │
│      · title / participants / motivation                  │
│      · motivation_confidence (explicit|inferred)          │
│      · outcome / summary / retrieval_text / keywords      │
│      · A→B→A 模式强制 target 推理                         │
│  → stage3_dryrun.json (actions + events)                  │
│  P3-P5 (PlotArc / Video / Global): 已实现                 │
└──────────────────────────────────────────────────────────┘
```

**关键设计**：
- Stage 2 不做基于视觉的场景检测（TransNetV2 仅作无 `audio.json` 时的回退路径），而是直接用 Stage 1 的声纹 segments 生成抽帧锚点，每个关键帧都绑定说话人 + 台词，下游 VLM 描述有强上下文；总采样量从 4000+ 帧降到 ~400 帧。
- Stage 3 采用 4 层架构：**Perception → Action(P1) → Event(P2) → Plot/Video/Global(P3-P5)**，P1-P5 全部已实现。

## 项目结构

```
VideoLens/
├── src/                           # 主代码包
│   ├── pipeline/                  # Pipeline 模块
│   │   ├── stage1_audio.py        #   Stage 1: 音频处理
│   │   ├── stage2_visual.py       #   Stage 2: 视觉处理 (speaker-anchor + tiered OCR)
│   │   ├── stage3/                #   Stage 3: P1-P5 子包 (p1_p2_actions / p345_kb / llm_json)
│   │   └── orchestrator.py        #   Pipeline 编排器 (resolve_video_path / run_pipeline)
│   ├── core/                      # 核心基础设施
│   │   ├── config.py              #   配置管理 (frozen AppConfig 单例)
│   │   ├── cost.py                #   成本追踪 (分模态计价)
│   │   ├── logging.py             #   日志
│   │   ├── llm/                   #   LLM 客户端
│   │   │   ├── base_client.py     #     基类 (_report_usage 分模态计价 + enable_thinking)
│   │   │   ├── qwen_text.py       #     文本 LLM (qwen3.7-plus)
│   │   │   └── qwen_vl.py         #     视觉 LLM
│   │   ├── helpers/               #   工具函数
│   │   │   ├── json_utils.py      #     save_json (统一编码)
│   │   │   ├── text_utils.py      #     JSON 提取
│   │   │   └── prompt_loader.py
│   │   └── models/                #   数据模型
│   │       └── scene.py           #     Scene 数据类
│   ├── scene/                     # 场景/锚点生成
│   │   ├── transnet_detector.py   #   TransNetV2 ONNX (回退路径)
│   │   └── speaker_anchor.py      #   ★ 声纹锚点生成 (Stage 2 主路径)
│   ├── voiceprint/                # 讯飞声纹
│   │   └── client.py
│   ├── eval/                      # 评估工具 (不进生产 pipeline)
│   │   ├── reference_asr.py       #   omni-plus 全集参考 ASR
│   │   ├── ocr_accuracy.py        #   OCR 准确率评估器
│   │   ├── stage3_build_quality.py #  Stage 3 建库字段完整性 + schema 合规
│   │   └── stage3_retrieval.py    #   Stage 3 用库 BM25 检索 + Recall@K
│   └── app/                       # CLI 入口
│       ├── cli.py
│       └── main.py
├── scripts/                       # 编排脚本
│   ├── stage3_p1p2.py             #   Stage 3 P1+P2 单集建库 (薄 CLI)
│   ├── stage3_p345.py             #   Stage 3 P3+P4+P5 完整 KB (薄 CLI)
│   ├── stage3_eval.py             #   Stage 3 单集评估编排 (建库 + 建库评估 + 用库)
│   ├── frontend_app.py            #   陪看智能体「小影」Streamlit 应用
│   └── export_for_gt.py           #   导出 Ground Truth
├── config/
│   ├── pipeline.yaml              # Pipeline 参数 + 模型选择 + pricing + voiceprint_groups
│   └── prompts.yaml               # Prompt 模板 (Stage 1a/1b/2/3-P1/3-P2)
├── tests/                         # 评估脚本
│   ├── eval_audio.py              #   Stage 1 声纹 GT 评估
│   ├── batch_test_ocr.py          #   Stage 2 OCR 批量测试编排器
│   └── results/                   #   评估结果
├── experiments/                   # 实验记录 (含 _archive 历史归档)
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

### 运行 Stage 1 + Stage 2（生产 pipeline）

```bash
# 运行完整 Pipeline (Stage 1 + 2)
python -m src.app.main run "家有儿女/第一季/第01集"

# 只运行 Stage 1 (音频) 或 Stage 2 (视觉)
python -m src.app.main run "家有儿女/第一季/第01集" --stage 1

# 常用参数
--skip-theme        # 跳过片头/片尾曲检测
--chunk 90          # Omni 切块时长 (秒)
--vp-threshold 0    # 关闭声纹阈值过滤 (默认 0.4, 低于此值统一标为「路人」)
```

### 运行 Stage 3（评估 pipeline，未进 orchestrator）

```bash
# 单集 Stage 3 完整评估: dry-run → 建库评估 → 用库评估
python -m scripts.stage3_eval --video "家有儿女/第二季/第001集"

# 强制重跑建库 (默认复用已有 stage3_dryrun.json)
python -m scripts.stage3_eval --video "家有儿女/第二季/第001集" --redo-dryrun

# 只重跑用库评估 (复用已有 query 集)
python -m scripts.stage3_eval --video "家有儿女/第二季/第001集" --reuse-queries

# 只跑建库 (P1+P2) 不评估
python -m scripts.stage3_p1p2 --video "家有儿女/第二季/第001集" --save
```

### Stage 1/2 评估

```bash
# Stage 1 声纹评估 (需要人工 GT)
python -m tests.eval_audio --video "052 鸟蛋之争" --save

# Stage 2 OCR 批量评估 (用 omni-plus 参考 ASR 作为软 GT)
python -m tests.batch_test_ocr --episodes "S1E1,S1E2" --workers 3
python -m tests.batch_test_ocr --only-eval                # 仅重算评估
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

**输出:** `audio.json`，`segments[]` 携带 `speaker_pred` / `vp_score` / `emotion` / `text`。

### Stage 2: 视觉处理

| 步骤 | 模块 | 功能 |
|------|------|------|
| 2a | `src/pipeline/speaker_anchor.py` | 把声纹 segments 转 speaker anchors (midpoint/switch/silence) |
| 2b | OpenCV | 每锚点抽 1 帧，写入 `keyframes/` |
| 2c | qwen3-vl-plus | Phase 1 OCR：midpoint 中心帧 |
| 2d | qwen3-vl-plus | Phase 2 OCR：仅对 miss 抽邻域 4 帧 |
| 2e | qwen-vl-max | (可选) 视觉描述，默认 `skip_captions=True` |

**输出:** `scenes.json`、`keyframes/`、`visual.json`。

### Stage 3: 结构化知识库

**4 层架构**（来自 `意见.txt`）：Perception → Action(P1) → Event(P2) → Plot(P3) → Video(P4) → Global(P5)。P1-P5 全部已实现。

| 步骤 | 模块 | 功能 |
|------|------|------|
| 预处理 | `src/pipeline/stage3/p1_p2_actions.py::should_skip` | 规则过滤 silence/empty/路人噪音（6-10% 削减） |
| 预处理 | `deduplicate_anchors` | 同 segment_id 多锚点去重（midpoint 优先） |
| **P1** | `call_p1` + qwen3.7-plus | 批量抽 Action (batch_size=5)：actor + 11 类 action_type + utterance + evidence |
| **P2** | `call_p2` + qwen3.7-plus | 聚合 Event (batch_size=8)：title + participants + motivation + outcome + summary + retrieval_text + keywords + actions.target |
| 3.5 | event_id 跨 batch 重编号 | 统一 event_id 格式为 `{video_id}_e{seq:03d}` |

**P1 Action 11 类枚举**: `inform / ask / answer / command / refuse / promise / threaten / deceive / argue / invite / react`（其中 `ask > react` 优先级硬约束，避免疑问句被误判为纯感叹）。

**P2 Target 推理规则**：
- 优先级 1：utterance 内容直接指明
- 优先级 2：**A→B→A 默认推断**（强制）—— 2+ actor 交替发言时，每个 actor.target = 时序下一个发言者
- 优先级 3：caption_struct target 参考
- **允许 target 为空的 3 种情况**：(a) 独白；(b) 群体广播无接收方；(c) 旁白

**输出:** `stage3_dryrun.json`，顶层含 `video_id / stats / cost / actions / events / characters_to_resolve`。

## AI 模型依赖

Omni 系列按输入模态分别计价（文本/音频/图片各有独立费率），输出按文本输出计。

| 模型 | 用途 | 文本输入 | 音频输入 | 图片输入 | 文本输出 |
|------|------|---------|---------|----------|---------|
| qwen3.5-omni-plus | 片头/片尾曲检测 / 评估参考 ASR | 7.0 | 53.0 | 40.0 | 213.0 |
| qwen3.5-omni-flash | Stage 1 ASR + 情感 + 时间戳 | 2.2 | 18.0 | 13.3 | 72.0 |
| qwen3-vl-plus | ★ Stage 2 字幕 OCR | 1.4 | — | 1.4 | 11.2 |
| qwen-vl-max | Stage 2 视觉描述（可选） | 3.0 | — | 3.0 | 9.0 |
| qwen3.7-plus | ★ Stage 3 P1/P2 文本生成 | 1.0 | — | — | 4.0 |
| 讯飞声纹 | Stage 1 说话人识别 | — | — | — | 按次计费 |

单位均为 ¥/百万 Token，价格来源 [阿里云百炼模型价格](https://help.aliyun.com/zh/model-studio/model-pricing)。

成本追踪：`src/core/cost.py` 的 `CostTracker` 通过 DashScope 响应中的 `usage.prompt_tokens_details.{text,audio}_tokens` 自动按模态归价。

**典型单集成本**（《家有儿女》22 min）：
- Stage 1（flash + 讯飞）: ~1.5 元
- Stage 2 OCR（~440 anchors × tiered 2-phase）: ~1.2 元
- Stage 2 Caption（可选）: ~2.5 元
- **Stage 1+2 合计**: ~2.7 元（不含 caption）
- Stage 3 P1+P2（关 thinking）: ~1 元
- Stage 3 P1+P2（开 thinking）: ~3 元（输出 token 暴涨 3-4 倍，但 JSON 稳定性显著提升）
- 评估专用 omni-plus 参考 ASR: ~5 元（仅评估时跑）

`base_client.py::chat()` 默认 `enable_thinking=False`；Stage 3 实验显示开 thinking 让字段完整率/Schema 合规/用库 Recall 全面提升，但成本/耗时也涨 3-4 倍。生产扩量建议关 thinking，单集关键集可开。

## 评估指标

### Stage 2 OCR 评估（无人工 GT，用 omni-plus 参考 ASR 作软 GT）

模块 `src/eval/ocr_accuracy.py`：
- **hit_rate (非静默)**：非 silence 锚点中 OCR 命中比例。目标 ≥95%。
- **recall (参考段覆盖)**：reference_asr.json 的句子中被 OCR anchor 命中的比例。
- **char P/R/F1**：OCR 文本与匹配参考段的字符级 P/R/F1。
- **miss_candidates**：OCR=无字幕 但参考有人声的「真漏」数量。

7 集《家有儿女》批量测试结果（详见 `experiments/_archive/ocr_batch_家有儿女7集/`）：

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

### Stage 3 评估（建库 + 用库两维度）

**建库评估** (`src/eval/stage3_build_quality.py`)：
- **字段完整率（宽松）**：Action 必填 `actor/action/utterance`；Event 必填 `title/motivation/summary/retrieval_text`
- **Schema 违反**：action 枚举 / motivation_confidence 枚举 / target 类型校验
- **软质量指标**：vp_low_confidence_ratio / char_unknown_event_ratio / empty_target_link_ratio / event_with_empty_actions_ratio
- **覆盖率**：actions_per_candidate / events_per_action / avg_actions_per_event

**用库评估** (`src/eval/stage3_retrieval.py`)：
- LLM 自生成 query 集（每个 event 反向生成 1 个自然问句）
- 自实现 BM25（jieba 分词，~30 行）+ Recall@1/3/5/10 + MRR

《家有儿女》第001集基线（qwen3.7-plus + thinking + 强制 A→B→A 推理）：

| 指标 | 数值 |
|---|---|
| 建库规模 | 331 actions / 63 events |
| 字段完整率 | 100% (Action + Event) |
| Schema 违反 | 0 |
| empty_target_link_ratio | 0.11 (A→B→A 强制推理生效) |
| char_unknown_event_ratio | 0.00 |
| Recall@1 / @5 / @10 | 0.92 / 0.98 / 1.00 |
| MRR | 0.95 |
| 单集 cost | 3.05 元 (开 thinking) / ~1 元 (关 thinking) |

## License

MIT
