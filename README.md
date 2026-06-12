# VideoLens

影视视频内容分析系统 — 基于多模态 AI 的视频音频处理、视觉理解与知识库生成。

## 架构概览

```
视频输入
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1: 音频处理                                        │
│  a) qwen3.5-omni-plus 检测片头/片尾曲                     │
│  b) qwen3.5-omni-flash ASR + 说话人轮次 + 情感 + 时间戳   │
│  c) 讯飞声纹 1:N 说话人识别                                │
│  → audio.json                                             │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2: 视觉处理                                        │
│  PySceneDetect 场景检测 + 关键帧提取                       │
│  + Qwen-VL-Max OCR + Caption                             │
│  → scenes.json + keyframes/ + visual.json                │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3: 结构化知识库                                     │
│  基于 Stage1+2 输出生成结构化知识库                         │
│  → knowledge.json                                         │
└──────────────────────────────────────────────────────────┘
```

## 项目结构

```
VideoLens/
├── src/                           # 主代码包
│   ├── pipeline/                  # Pipeline 模块
│   │   ├── stage1_audio.py        #   Stage 1: 音频处理
│   │   ├── stage2_visual.py       #   Stage 2: 视觉处理
│   │   ├── stage3_knowledge.py    #   Stage 3: 知识库
│   │   └── orchestrator.py        #   Pipeline 编排器
│   ├── core/                      # 核心基础设施
│   │   ├── config.py              #   配置管理
│   │   ├── cost.py                #   成本追踪
│   │   ├── logging.py             #   日志
│   │   ├── paths.py               #   路径管理
│   │   ├── llm/                   #   LLM 客户端
│   │   │   ├── base_client.py     #     基类
│   │   │   ├── qwen_text.py       #     文本 LLM
│   │   │   └── qwen_vl.py         #     视觉 LLM
│   │   ├── helpers/               #   工具函数
│   │   │   ├── ffmpeg.py          #     音频提取
│   │   │   ├── json_utils.py      #     JSON 读写
│   │   │   ├── text_utils.py      #     JSON 提取
│   │   │   └── prompt_loader.py   #     Prompt 加载
│   │   └── models/                #   数据模型
│   │       ├── scene.py           #     Scene
│   │       └── transcript.py      #     TranscriptSegment
│   ├── scene/                     # 场景检测
│   │   └── detector.py            #   PySceneDetect 封装
│   ├── voiceprint/                # 讯飞声纹
│   │   ├── client.py              #   API 客户端
│   │   └── matcher.py             #   声纹匹配器
│   └── app/                       # CLI 入口
│       ├── cli.py                 #   命令定义
│       └── main.py                #   入口点
├── config/
│   ├── pipeline.yaml              # Pipeline 参数配置
│   └── prompts.yaml               # Prompt 模板
├── experiments/                   # 实验结果与评估脚本
│   ├── eval_car.py                #   声纹识别准确率评估
│   └── ...
├── data/
│   ├── videos/                    # 输入视频
│   ├── gt/                        # Ground Truth
│   └── output/                    # 处理输出
├── .env                           # 环境变量
└── requirements.txt               # Python 依赖
```

## 快速开始

### 环境要求

- Python >= 3.12
- ffmpeg (系统已安装)

### 安装

```bash
uv venv .venv
.venv\Scripts\activate    # Windows
uv pip install -r requirements.txt
```

### 配置

1. 设置 `DASHSCOPE_API_KEY` 系统环境变量 (阿里云 DashScope)
2. 编辑 `.env` 填入讯飞声纹 API 凭证

### 运行

```bash
# 运行完整 Pipeline
python -m src.app.main run "052 鸟蛋之争"

# 只运行 Stage 1 (音频)
python -m src.app.main run "052 鸟蛋之争" --stage 1

# 跳过片头/片尾曲检测
python -m src.app.main run "052 鸟蛋之争" --skip-theme

# 自定义 chunk 时长
python -m src.app.main run "052 鸟蛋之争" --chunk 90
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
     "text": "...", "emotion": "关切语调",
     "speaker_gt": "", "emotion_gt": ""}
  ]
}
```

### Stage 2: 视觉处理

| 步骤 | 模型 | 功能 |
|------|------|------|
| 2a | PySceneDetect | 场景边界检测 |
| 2b | OpenCV | 关键帧提取 |
| 2c | — | 字幕帧过滤 |
| 2d | qwen-vl-max | OCR 字幕识别 |
| 2e | qwen-vl-max | 视觉描述生成 |

**输出:** `scenes.json`, `keyframes/`, `visual.json`

### Stage 3: 结构化知识库

(新设计，待实现)

## AI 模型依赖

| 模型 | 用途 | 类型 | 定价 (input/output, ¥/M tokens) |
|------|------|------|------|
| qwen3.5-omni-plus | 片头/片尾曲检测 | API | 53.0 / 40.0 |
| qwen3.5-omni-flash | ASR + 情感 + 时间戳 | API | 18.0 / 13.3 |
| qwen-vl-max | OCR + 视觉描述 | API | 3.0 / 9.0 |
| 讯飞声纹 | 说话人识别 | API | 按次计费 |

## License

MIT
