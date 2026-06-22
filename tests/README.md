# tests/ — 评估与批量测试脚本

本目录下的脚本均为**评估/批量测试工具**，不参与生产 pipeline。生产入口在 `python -m src.app.main run`。

## 脚本一览

| 文件 | 用途 |
|------|------|
| `eval_audio.py` | Stage 1 声纹识别 GT 评估（需要人工标注的 `data/gt/*.json`） |
| `batch_test_ocr.py` | Stage 2 OCR 字幕识别批量测试（用 omni-plus 参考 ASR 作为软 GT） |
| `results/` | 各次评估的产物（已 gitignore 大体量报告） |

---

## 一、`eval_audio.py` — Stage 1 声纹识别评估

以人工标注的 Ground Truth (GT) 为基准，通过时间重叠匹配计算多项指标。

**核心原则：仅统计主角团角色。** CAR/SAR 的分母只包含主角团 GT 段，非主角团（如歌手、路人）不参与计算。

### 评估指标

#### CAR (Character Attribution Rate) — 角色归属正确率

```
CAR = 主角团正确识别的 GT 段数 / 主角团 GT 段数
```

**算法：**

1. 筛选 GT 中属于主角团的台词段
2. 对每个主角团 GT 段，遍历所有预测段，计算时间重叠
3. 对每个有重叠的预测段，计算「文本贡献量」= `重叠比例 × 文本长度`
4. 按角色累加贡献量，取最大者作为该 GT 段的预测角色
5. 与 GT 角色比较，一致则记为正确

**适用场景：** 评估 pipeline 对「谁在说话」的整体判断能力，容许段落切分不完全对齐。

#### SAR (Speaker Attribution Rate) — 逐段匹配准确率

```
SAR = 主角团逐段正确匹配数 / 主角团 GT 段数
```

**算法：**

1. 筛选 GT 中属于主角团的台词段
2. 对每个主角团 GT 段，找到时间重叠最大的预测段
3. 直接比较 `speaker_pred` 与 `speaker_gt`

**适用场景：** 评估逐段对齐时的识别精度，对段落切分更敏感。

#### 情感准确率

对有 `emotion_gt` 的全部 GT 段（含非主角团），找到时间重叠最大的预测段，比较 `emotion` 字段。

### 声纹置信度阈值 (`--vp-threshold`)

声纹识别返回的 `vp_score` 表示匹配置信度。当低于设定阈值时，预测角色标记为「路人」：

- **0 (默认)：** 不过滤，使用 pipeline 原始输出
- **0.3~0.5：** 根据实验确定，过滤低置信度的错误归属

### 数据依赖

```
data/
├── output/{video}/audio.json        # Pipeline 输出 (必需)
└── gt/
    └── {系列名}{集数}_gt.json        # 统一 GT (声纹+情感)
```

### 用法

```bash
# 基本评估
python -m tests.eval_audio --video "052 鸟蛋之争"

# 指定声纹置信度阈值
python -m tests.eval_audio --video "052 鸟蛋之争" --vp-threshold 0.4

# 输出详细报告 (每段 GT 的匹配详情)
python -m tests.eval_audio --video "052 鸟蛋之争" --detail

# 保存报告到 tests/results/
python -m tests.eval_audio --video "052 鸟蛋之争" --save
```

报告保存到 `tests/results/eval_{video}.json`。

---

## 二、`batch_test_ocr.py` — Stage 2 OCR 批量测试

针对没有人工 GT 的批量测试场景：调用 `qwen3.5-omni-plus` 对全集音频跑高准确率 ASR 作为**软 Ground Truth**，再对比 Stage 2 的 OCR 结果计算准确率。

### 流程（每集）

```
Stage 1 → Stage 2 (speaker-anchor + qwen3-vl-plus OCR)
   → omni-plus 参考 ASR (chunk 60s, 并行 4 路)
   → ocr_accuracy.evaluate_episode()
   → ocr_eval.json
```

集间可并行（默认 3 路），单集约 20 min、~8 元（含参考 ASR）。

### 用法

```bash
# 跑全集 7 集
python -m tests.batch_test_ocr --workers 3

# 只跑指定剧集 (ep_id 在脚本内 EPISODES 列表)
python -m tests.batch_test_ocr --episodes "S1E1,S1E2,S2E2"

# 已有 audio.json 时跳过 Stage 1 (省时省钱)
python -m tests.batch_test_ocr --episodes "S1E1" --skip-stage1

# 跳过 omni-plus 参考 ASR (无法计算 recall/precision，但能算 hit_rate)
python -m tests.batch_test_ocr --episodes "S1E1" --skip-reference

# 前面都跑过，仅重算评估
python -m tests.batch_test_ocr --only-eval
```

### 输出

- `data/output/{video}/ocr_eval.json` — 单集指标 + miss_candidates 列表
- `data/output/_batch_reports/batch_ocr_{ts}.json` — 批次元信息 + 逐集结果
- `data/output/_batch_reports/batch_ocr_{ts}_aggregate.json` — 跨集汇总

### 评估指标（见 `src/eval/ocr_accuracy.py`）

| 指标 | 含义 | 目标 |
|------|------|------|
| `hit_rate_nonsilence` | 非静默锚点中 OCR 命中比例 | **≥95%** |
| `recall` | reference ASR 句子被 OCR 命中比例（结构上偏低，每个 utterance 只采 1 帧） | 趋势参考 |
| `char_precision` / `recall` / `f1` | OCR 文本与匹配参考段的字符级 P/R/F1 | — |
| `n_miss_candidates` | OCR=无字幕但参考有人声的「真漏」 | 越低越好 |

历史批量结果见 `experiments/ocr_batch_家有儿女7集/`。
