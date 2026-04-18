# VideoLens

影视视频内容分析与检索系统 — 基于多模态 AI 的长视频场景分割、内容理解与语义检索平台。

## 架构概览

```
视频输入
  │
  ▼
┌─────────────────────────────────────────────────┐
│  Stage 1: 场景分割 (vl.scene)                   │
│  PySceneDetect → 关键帧提取 (OpenCV)              │
└─────────────────────────────────────────────────┘
  │ scenes.json + keyframes/
  ▼
┌─────────────────────────────────────────────────┐
│  Stage 2: 多模态场景理解                          │
│  Qwen3.5-Omni-Plus (音频+画面同时理解)            │
│  + CLIP 视觉编码 + 角色信息提取                    │
└─────────────────────────────────────────────────┘
  │ transcripts/ + embeddings/ + characters/
  ▼
┌─────────────────────────────────────────────────┐
│  Stage 3: 视觉语义理解 + 索引                      │
│  VLM 结构化场景描述 (Qwen-VL-Max, 按需)           │
│  + FAISS 向量索引 + enriched 元数据               │
└─────────────────────────────────────────────────┘
  │ captions/ + index/ + scenes/metadata.json
  ▼
┌─────────────────────────────────────────────────┐
│  Stage 4: 多模态时序对齐                          │
│  ASR↔场景对齐 / 角色追踪 / 叙事事件检测            │
└─────────────────────────────────────────────────┘
  │ alignment/aligned_timeline.json
  ▼
┌─────────────────────────────────────────────────┐
│  Stage 5: 结构化知识库生成                        │
│  三层 JSON: 视频层 → 大事件层 → 小事件层            │
└─────────────────────────────────────────────────┘
  │ knowledge/knowledge_base.json
  ▼
结构化视频知识库（支持检索 / QA / 分析）
```

## 项目结构

```
VideoLens/
├── vl/                         # 主代码包
│   ├── core/                   # 核心基础设施
│   │   ├── config.py           # 配置管理 (.env + pipeline.yaml + prompts.yaml)
│   │   ├── logging.py          # 日志配置
│   │   ├── paths.py            # 路径管理器 (统一管理输入输出路径)
│   │   ├── models/             # 数据模型
│   │   │   ├── scene.py        #   Scene (场景: 时间范围/关键帧/CLIP向量/VLM描述)
│   │   │   ├── transcript.py   #   TranscriptSegment (转录片段: 台词/说话人/时间戳)
│   │   │   ├── character.py    #   Character (角色: omni 提取的角色信息)
│   │   │   └── video_meta.py   #   VideoMeta (视频元数据)
│   │   ├── llm/                # LLM 客户端
│   │   │   ├── base_client.py  #   BaseLLMClient (JSON 提取等共用逻辑)
│   │   │   ├── qwen_text.py    #   QwenTextClient (文本生成/JSON提取)
│   │   │   └── qwen_vl.py      #   QwenVLClient (视觉语言模型)
│   │   └── helpers/            # 工具函数
│   │       ├── ffmpeg.py       #   音频提取 (ffmpeg)
│   │       └── json_utils.py   #   JSON 读写
│   ├── scene/                  # 场景检测 & 关键帧提取
│   │   ├── detector.py         #   SceneDetector (PySceneDetect 封装)
│   │   └── frame_sampler.py    #   FrameSampler (均匀/智能关键帧采样)
│   ├── asr/                    # 语音转写
│   │   ├── qwen_omni.py        #   QwenOmni (Qwen3.5-Omni-Plus 全模态理解, 推荐)
│   │   ├── qwen_asr.py         #   QwenASR (qwen3-asr-flash 纯音频 API)
│   │   └── transcriber.py      #   Transcriber (faster-whisper 本地转录)
│   ├── vision/                 # 视觉处理
│   │   └── clip_encoder.py     #   CLIPEncoder (图像/文本向量编码)
│   ├── store/                  # 存储层
│   │   ├── vector_store.py     #   VectorStore (FAISS 向量索引)
│   │   └── metadata_store.py   #   MetadataStore (JSON 元数据)
│   ├── pipeline/               # 五阶段流水线
│   │   ├── orchestrator.py     #   PipelineOrchestrator (编排 + 断点续跑)
│   │   ├── stage1_ingestion.py #   Stage 1: 场景分割
│   │   ├── stage2_analysis.py  #   Stage 2: 多模态场景理解
│   │   ├── stage3_understanding.py # Stage 3: 视觉语义理解 + 索引
│   │   ├── stage4_alignment.py #   Stage 4: 多模态时序对齐
│   │   └── stage5_knowledge.py #   Stage 5: 结构化知识库生成
│   └── app/                    # CLI 入口
│       ├── main.py             #   入口点 (加载 .env + HF 镜像配置)
│       └── cli.py              #   Typer CLI 命令定义
├── config/
│   ├── pipeline.yaml           # 模型 & 流水线参数配置
│   └── prompts.yaml            # LLM Prompt 模板
├── data/
│   ├── videos/                 # 输入视频目录
│   └── output/                 # 处理输出 (按 video_id 组织)
├── .env                        # 环境变量 (API Keys 等)
├── pyproject.toml              # 项目配置 & poe 任务
└── requirements.txt            # Python 依赖
```

## 快速开始

### 环境要求

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) (推荐) 或 pip
- ffmpeg (系统已安装，用于音频提取)

### 安装

```bash
# 1. 创建虚拟环境
uv venv .venv

# 2. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. 安装依赖
uv pip install -r requirements.txt
```

### 配置

```bash
# 复制环境变量模板并填入 API Key
cp .env.example .env
```

| 变量 | 用途 | 获取方式 |
|------|------|---------|
| `DASHSCOPE_API_KEY` | Qwen VL/Text/Omni API | [阿里云 DashScope](https://dashscope.console.aliyun.com/) |

### 建库（索引视频）

```bash
# 索引视频（首次运行会自动下载 CLIP 等本地模型）
python -m vl.app.main index data/videos/your_video.mp4

# 指定视频类型
python -m vl.app.main index data/videos/your_video.mp4 --genre anime

# 禁用断点续跑
python -m vl.app.main index data/videos/your_video.mp4 --no-resume

# poe 快捷命令
poe index data/videos/your_video.mp4
```

索引完成后，输出目录结构：

```
data/output/
├── scenes/{video_id}/          # 场景关键帧 + scenes.json + metadata.json
├── transcripts/{video_id}/     # 转录结果 (带说话人/情感/时间戳)
├── embeddings/{video_id}/      # CLIP 向量 (clip_vectors.npy)
├── characters/{video_id}/      # 角色聚类结果
├── captions/{video_id}/        # VLM 结构化场景描述
├── index/{video_id}/           # FAISS 索引 + doc_store.json
├── alignment/{video_id}/       # 时序对齐数据
├── knowledge/{video_id}/       # 结构化知识库 JSON
├── preprocessing/              # 提取的音频文件 (.wav)
└── checkpoints/                # 断点续跑文件
```

### 用库（检索 / 问答 / 分析）

```bash
# 语义检索
python -m vl.app.main search "两个人在草地上交谈"
python -m vl.app.main search "孙悟空" --video 052 --top-k 5

# 视频问答 (QA)
python -m vl.app.main qa "主角为什么这么做" --video 052
python -m vl.app.main qa "视频里发生了什么冲突" --top-k 8

# 视频分析
python -m vl.app.main analyze 052 --type summary
python -m vl.app.main analyze 052 --type characters
python -m vl.app.main analyze 052 --type timeline

# poe 快捷命令
poe search "打斗场景"
poe qa "主角是谁" -- --video 052
poe analyze 052 -- --type summary
```

## 五阶段流水线详解

### Stage 1: 场景分割

**功能**: 使用 PySceneDetect 检测视频中的场景切换点，并为每个场景提取关键帧。

**模块**: `vl/scene/detector.py` (SceneDetector), `vl/scene/frame_sampler.py` (FrameSampler)

**入口函数**:
```python
# vl/pipeline/stage1_ingestion.py
def run_stage1(video_path: str, output_dir: str, config: AppConfig) -> list[Scene]
```

**输入**: 视频文件 (`data/videos/{video_id}.mp4`)

**输出**:
| 文件 | 内容 |
|------|------|
| `scenes/{video_id}/scenes.json` | 场景列表 (scene_id, start_time, end_time, index) |
| `scenes/{video_id}/keyframes/scene_NNNN.jpg` | 每个场景的关键帧图片 |

**配置项** (`config/pipeline.yaml`):
```yaml
scene_detection:
  backend: pyscenedetect
  content_threshold: 27.0    # 场景切换阈值 (越小越灵敏)
  min_scene_len: 1.0         # 最短场景时长 (秒)
```

**测试**: `python -m vl.app.main test-stage 052 --stage 1`

---

### Stage 2: 多模态场景理解

**功能**: 使用 Qwen3.5-Omni-Plus 同时理解音频和画面，像人一样看视频。同时进行 CLIP 视觉编码和角色信息提取。

**模块**:
- `vl/asr/qwen_omni.py` — QwenOmni: 全模态理解 (音频+画面+结构化 prompt)
- `vl/vision/clip_encoder.py` — CLIPEncoder: 图像/文本向量编码

**入口函数**:
```python
# vl/pipeline/stage2_analysis.py
def run_stage2(video_path, video_id, scenes, audio_path, output_dir, config) -> dict[str, list[str]]
```

**输入**:
| 数据 | 来源 |
|------|------|
| 场景列表 | `scenes/{video_id}/scenes.json` (Stage 1) |
| 音频文件 | `preprocessing/{video_id}.wav` (自动从视频提取) |

**输出**:
| 文件 | 内容 |
|------|------|
| `transcripts/{video_id}/transcript.json` | 转录片段 (text, speaker, emotion, start/end_time) |
| `embeddings/{video_id}/clip_vectors.npy` | CLIP 视觉向量 (numpy array) |
| `characters/{video_id}/characters.json` | 角色信息 (从 omni 输出提取) |

**Qwen3.5-Omni-Plus 工作流程**:
1. 将场景按时间分组 (~2分钟/组)
2. 每组: 音频切片 + 2~5张关键帧 → API 调用
3. 结构化 prompt 引导输出 JSON: `{speakers, transcript, visual}`
4. 解析: 说话人/台词/情感 → TranscriptSegment, 视觉描述 → scene.structured_caption

**配置项**:
```yaml
asr:
  backend: qwen-omni         # "qwen-omni" (推荐) / "qwen" / "whisper"
  language: zh
  chunk_duration: 120        # 每个片段时长 (秒)
  max_keyframes_per_chunk: 5 # 每个片段最多关键帧数
```

**测试**: `python -m vl.app.main test-stage 052 --stage 2`

---

### Stage 3: 视觉语义理解 + 索引构建

**功能**: 如果 Stage 2 (Qwen-Omni) 已生成视觉描述，则跳过 VLM 调用，仅构建 FAISS 索引和 enriched 元数据。如果视觉描述不完整，使用 Qwen-VL-Max 补充。

**模块**: `vl/pipeline/stage3_understanding.py`, `vl/store/vector_store.py`

**入口函数**:
```python
# vl/pipeline/stage3_understanding.py
def run_stage3(video_id, video_title, genre, scenes, scene_transcripts,
               characters_info, output_dir, config)
```

**输入**:
| 数据 | 来源 |
|------|------|
| 场景列表 (含 structured_caption) | Stage 2 输出 |
| 场景台词映射 | Stage 2 输出 |
| 角色信息 | Stage 2 输出 |
| CLIP 向量 | `embeddings/{video_id}/clip_vectors.npy` (Stage 2) |

**输出**:
| 文件 | 内容 |
|------|------|
| `captions/{video_id}/captions.json` | 结构化场景描述 |
| `index/{video_id}/index.faiss` | FAISS 向量索引 |
| `index/{video_id}/doc_store.json` | 文档存储 (场景描述+台词+时间) |
| `scenes/{video_id}/metadata.json` | enriched 场景元数据 |

**配置项**:
```yaml
models:
  vlm: qwen-vl-max           # 视觉语言模型 (按需调用)
clip:
  embedding_dim: 512
  batch_size: 32
```

**测试**: `python -m vl.app.main test-stage 052 --stage 3`

---

### Stage 4: 多模态时序对齐

**功能**: 将视觉、音频、文本、角色数据按时序对齐，构建统一时间线，检测叙事事件。

**模块**: `vl/pipeline/stage4_alignment.py`

**入口函数**:
```python
# vl/pipeline/stage4_alignment.py
def run_stage4(video_id, video_title, scenes, scene_transcripts,
               output_dir, config) -> dict
```

**输入**:
| 数据 | 来源 |
|------|------|
| enriched 场景列表 | `scenes/{video_id}/metadata.json` (Stage 3) |
| 转录片段 | `transcripts/{video_id}/transcript.json` (Stage 2) |
| 角色数据 | `characters/{video_id}/characters.json` (Stage 2) |

**输出**:
| 文件 | 内容 |
|------|------|
| `alignment/{video_id}/aligned_timeline.json` | 统一时间线 (叙事事件 + 角色弧线 + 对齐场景) |

**对齐策略**:
1. ASR → 场景: 基于时间戳精确匹配
2. 角色 → 场景: omni 视觉描述 + ASR 说话人交叉验证
3. 叙事事件: 室内↔室外 / 白天↔晚上 变化触发新事件

**测试**: `python -m vl.app.main test-stage 052 --stage 4`

---

### Stage 5: 结构化知识库生成

**功能**: 基于时序对齐数据，增量生成三层结构化知识库。

**模块**: `vl/pipeline/stage5_knowledge.py`

**入口函数**:
```python
# vl/pipeline/stage5_knowledge.py
def run_stage5(video_id, video_title, genre, aligned_timeline,
               output_dir, config) -> dict
```

**输入**:
| 数据 | 来源 |
|------|------|
| 对齐时间线 | `alignment/{video_id}/aligned_timeline.json` (Stage 4) |

**输出**:
| 文件 | 内容 |
|------|------|
| `knowledge/{video_id}/knowledge_base.json` | 三层结构化知识库 |

**三层结构**:
```json
{
  "视频名": {
    "第一幕：白天室外": {
      "Character_A与Character_B在草地上": "两人并肩站立，面向同一方向",
      "Character_A跑向实验室": "独自奔跑，表情紧张"
    },
    "第二幕：白天室内": {
      "Character_A与Character_B对峙": "两人彼此注视，气氛紧张"
    }
  }
}
```

**测试**: `python -m vl.app.main test-stage 052 --stage 5`

---

## 模块测试指南

使用 `test-stage` 命令可以单独运行流水线中的某个阶段，方便边开发边测试。

### 基本用法

```bash
python -m vl.app.main test-stage <video_id> --stage <1-5>

# poe 快捷命令
poe test-stage <video_id> -- --stage <1-5>
```

### 各阶段前置条件

| Stage | 前置数据 | 自动检查 |
|-------|---------|---------|
| 1 | `data/videos/{video_id}.mp4` | 视频文件存在性 |
| 2 | Stage 1 输出 (`scenes.json`) + 视频文件 | 场景数据 + 自动提取音频 |
| 3 | Stage 2 输出 (transcripts + embeddings + characters) | 全部前置文件 |
| 4 | Stage 2/3 输出 | 场景 + 转录 + 角色数据 |
| 5 | Stage 4 输出 (`aligned_timeline.json`) | 对齐数据文件 |

### 测试示例

```bash
# 以 052.mp4 为例 (video_id = 052)

# Step 1: 场景分割 (不需要 API key)
python -m vl.app.main test-stage 052 --stage 1
# 输出: 检测到 N 个场景

# Step 2: 多模态理解 (需要 DASHSCOPE_API_KEY)
python -m vl.app.main test-stage 052 --stage 2
# 输出: N 个场景有台词, M 个转录片段

# Step 3: 索引构建 (不需要 API key 如果 Stage 2 已有描述)
python -m vl.app.main test-stage 052 --stage 3
# 输出: N 个场景, M 个有结构化描述

# Step 4: 时序对齐 (不需要 API key)
python -m vl.app.main test-stage 052 --stage 4
# 输出: N 个叙事事件, M 个角色弧线

# Step 5: 知识库生成 (需要 DASHSCOPE_API_KEY)
python -m vl.app.main test-stage 052 --stage 5
# 输出: N 个大事件, M 个小事件
```

### 注意事项

- `test-stage` 不修改 checkpoint 系统，不会影响正式流水线状态
- Stage 2 和 Stage 5 需要 `DASHSCOPE_API_KEY`
- 如果前置数据不存在，会提示缺失的文件路径和应先运行的阶段
- 每次测试会覆盖对应阶段的输出文件

## CLI 命令参考

| 命令 | 说明 | 示例 |
|------|------|------|
| `index` | 索引视频 (5 阶段流水线) | `python -m vl.app.main index data/videos/052.mp4` |
| `search` | 语义检索场景 | `python -m vl.app.main search "对话场景" --video 052` |
| `qa` | 视频问答 | `python -m vl.app.main qa "主角做了什么" --video 052` |
| `analyze` | 视频分析 (摘要/角色/时间线) | `python -m vl.app.main analyze 052 --type summary` |
| `test-stage` | 单独测试某个阶段 | `python -m vl.app.main test-stage 052 --stage 3` |

## 依赖的 AI 模型

| 模型 | 用途 | 类型 |
|------|------|------|
| Qwen3.5-Omni-Plus | 全模态理解 (音频+画面) | API 调用 |
| Qwen-VL-Max | 结构化场景描述 (按需补充) | API 调用 |
| Qwen-Plus | 知识库生成 / 查询扩展 / 重排 / QA | API 调用 |
| CLIP ViT-B-32 | 图像/文本向量编码 | 本地模型 (首次自动下载) |
| faster-whisper | 纯音频转录 (whisper 后端) | 本地模型 (首次自动下载) |

## License

MIT
