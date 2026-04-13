# VideoLens

影视视频内容分析与检索系统 — 基于多模态 AI 的长视频场景分割、内容理解与语义检索平台。

## 架构概览

```
视频输入
  │
  ▼
┌─────────────────────────────────────────────┐
│  Stage 1: 场景分割 (vl.scene)               │
│  PySceneDetect → 关键帧提取 (OpenCV)          │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  Stage 2: 多维分析                           │
│  ├─ ASR 转写 (faster-whisper medium)         │
│  ├─ CLIP 视觉编码 (clip-ViT-B-32)           │
│  └─ VLM 场景描述 (Qwen-VL-Max)              │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  Stage 3: 索引构建                           │
│  FAISS 向量索引 + 元数据存储                  │
└─────────────────────────────────────────────┘
  │
  ▼
语义检索：文本查询 → CLIP 编码 → FAISS 检索 → 结果排序
```

## 项目结构

```
VideoLens/
├── vl/                         # 主代码包
│   ├── __init__.py
│   ├── core/                   # 配置、数据模型、LLM 客户端、工具函数
│   │   ├── config.py           # 从 .env / pipeline.yaml 加载配置
│   │   ├── logging.py          # 日志配置
│   │   ├── paths.py            # 路径管理
│   │   ├── models/             # 数据模型 (scene, transcript, character, video_meta)
│   │   ├── llm/                # LLM 客户端 (qwen_vl, qwen_text)
│   │   └── helpers/            # 工具函数 (ffmpeg, json_utils)
│   ├── scene/                  # 场景检测 & 关键帧提取
│   │   ├── detector.py
│   │   └── frame_sampler.py
│   ├── asr/                    # 语音转写 (Phase 2: 说话人识别)
│   │   ├── transcriber.py
│   │   ├── aligner.py
│   │   └── diarizer.py
│   ├── vision/                 # CLIP 编码 (Phase 2: 人脸检测/聚类)
│   │   ├── clip_encoder.py
│   │   ├── face_detector.py
│   │   └── face_cluster.py
│   ├── store/                  # FAISS 向量存储 + JSON 元数据存储
│   │   ├── vector_store.py
│   │   └── metadata_store.py
│   ├── pipeline/               # 三阶段流水线编排 (含断点续跑)
│   │   ├── orchestrator.py
│   │   ├── stage1_ingestion.py
│   │   ├── stage2_analysis.py
│   │   └── stage3_indexing.py
│   └── app/                    # CLI 入口 (index / search)
│       ├── main.py
│       └── cli.py
├── config/
│   ├── pipeline.yaml           # 模型 & 流水线参数
│   └── prompts.yaml            # LLM Prompt 模板
├── data/
│   ├── videos/                 # 输入视频
│   └── output/                 # 处理输出 (场景/转写/索引等)
├── .env.example                # 环境变量模板
├── pyproject.toml              # 项目元数据 & PDM 构建配置
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

需要配置：

| 变量 | 用途 | 获取方式 |
|------|------|---------|
| `DASHSCOPE_API_KEY` | 通义千问 VL/Text API | [阿里云 DashScope](https://dashscope.console.aliyun.com/) |
| `HF_TOKEN` | pyannote 说话人识别 (Phase 2) | [HuggingFace](https://huggingface.co/settings/tokens) |
| `HF_ENDPOINT` | HuggingFace 镜像 (国内网络必须) | 固定值: `https://hf-mirror.com` |

> **国内网络注意**: `HF_ENDPOINT=https://hf-mirror.com` 需要在 `.env` 中配置，用于 HuggingFace 模型下载加速。
> ```bash
> # .env 中添加:
> HF_ENDPOINT=https://hf-mirror.com
> ```

### 建库（索引视频）

将视频放入 `data/videos/` 目录后，运行以下命令进行场景分割、多维分析和索引构建：

```bash
# 索引视频（首次运行会自动下载 faster-whisper、CLIP 等模型）
python -m vl.app.main index data/videos/your_video.mp4

# 指定视频类型 (movie|tv|anime)
python -m vl.app.main index data/videos/your_video.mp4 --genre tv

# 禁用断点续跑（从头开始）
python -m vl.app.main index data/videos/your_video.mp4 --no-resume
```

也可通过 poethepoet 快捷运行：

```bash
poe index data/videos/your_video.mp4
poe index data/videos/your_video.mp4 -- --genre anime
```

索引完成后，输出结构如下：

```
data/output/
├── video_id/
│   ├── scenes/          # 场景关键帧图片
│   ├── transcript.json  # ASR 转写结果
│   └── index/
│       ├── index.faiss  # FAISS 向量索引
│       └── doc_store.json  # 场景元数据
```

### 用库（语义检索）

```bash
# 搜索场景（默认搜索所有已索引视频）
python -m vl.app.main search "两个人在雨中对话"

# 在指定视频中搜索
python -m vl.app.main search "孙悟空" --video video_id

# 指定返回结果数量
python -m vl.app.main search "打斗场景" --top-k 5
```

也可通过 poethepoet 快捷运行：

```bash
poe search "两个人在雨中对话"
poe search "孙悟空" -- --video video_id
```

## 依赖的 AI 模型

| 模型 | 用途 | 自动下载 |
|------|------|---------|
| faster-whisper medium | 中文语音转写 | 首次运行时下载 |
| clip-ViT-B-32 | 图像/文本向量编码 | 首次运行时下载 |
| qwen-vl-max | 场景内容描述 | API 调用 |
| qwen-plus | 查询扩展/重排 (Phase 2) | API 调用 |

> faster-whisper 和 CLIP 模型在首次运行时会自动从 HuggingFace 下载。大陆网络环境需在 `.env` 中配置 `HF_ENDPOINT=https://hf-mirror.com`。

## Phase 1 vs Phase 2

| 功能 | Phase 1 | Phase 2 |
|------|:-------:|:-------:|
| 场景检测 (PySceneDetect) | done | |
| 关键帧提取 | done | |
| 语音转写 (faster-whisper) | done | |
| CLIP 视觉编码 | done | |
| VLM 场景描述 | done | |
| FAISS 语义检索 | done | |
| 断点续跑 | done | |
| 说话人识别 (pyannote) | | planned |
| 人脸检测/角色聚类 (insightface) | | planned |
| 动漫场景检测 (TransNetV2) | | planned |
| LLM 查询扩展 & 重排 | | planned |

## License

MIT
