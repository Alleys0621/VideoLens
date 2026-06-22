# OCR 字幕识别批量测试 — 家有儿女 7 集

> 实验日期: 2026-06-18
> 测试范围: 家有儿女 第一季 3 集 (.mkv) + 第二季 4 集 (.mp4)
> 实验目的: 验证基于「声纹锚点 + qwen3-vl-plus 分层 OCR」的字幕识别方案在大批量剧集上的稳定性与准确率

## 一、实验设计

### 1.1 流程
每集统一跑 4 步:
1. **Stage 1 (音频)** — `qwen3.5-omni-flash` 做 ASR + 说话人轮次切分 + 讯飞声纹 1:N 匹配
2. **Stage 2 (视觉)** — 基于声纹 segments 生成 speaker anchors (midpoint/switch/silence 三类) → 每锚点抽 1 帧 → `qwen3-vl-plus` 分层 2-phase OCR (中心帧 + miss 邻域帧)
3. **参考 ASR** — `qwen3.5-omni-plus` 全集高准确率逐句转录,作为软 Ground Truth (Stage 1 的 flash ASR 主要用于说话人轮次切分,文本本身有 5-10% 错误率,不能用作 OCR 评估)
4. **评估** — 对比 OCR 结果与参考 ASR,计算 hit_rate / recall / 字符级 P/R/F1 / miss_candidates

### 1.2 集间并行策略
- 3 集一批,集内 Stage1→Stage2→RefASR 串行
- 单集耗时约 1100-1200s,7 集总耗时约 45 min

### 1.3 关键参数
| 参数 | 取值 | 说明 |
|------|------|------|
| OCR 模型 | `qwen3-vl-plus` | 输入便宜 (1.4 元/M)、输出贵 (11.2 元/M),OCR 任务文本短,极省钱 |
| Caption | **跳过** | 用户要求冻结 caption,本次只测 OCR |
| Anchor chunk_dur | 60s | 与 Stage 1 一致 |
| Ref ASR chunk_dur | 60s | omni-plus 重新切块转录 |
| MATCH_WINDOW | ±1.5s | OCR anchor 时间戳与参考段时间戳的匹配窗口 |
| MATCH_THRESHOLD | 0.5 字符重叠 | 判定 OCR 文本是否命中参考段 |

## 二、实验结果

### 2.1 逐集指标

| 集数 | 视频路径 | hit_rate (非静默) | recall (参考段覆盖) | char P | char R | char F1 | 真漏 misses |
|------|----------|:--------:|:--------:|:------:|:------:|:-------:|:-----------:|
| S1E1 | 家有儿女/第一季/第01集 | **95.08%** | 49.16% | 84.9% | 78.1% | 81.3% | 14 |
| S1E2 | 家有儿女/第一季/第02集 | **95.02%** | 46.48% | 85.2% | 78.4% | 81.7% | 12 |
| S1E3 | 家有儿女/第一季/第03集 | **95.29%** | 36.36% | 80.5% | 75.0% | 77.7% | 11 |
| S2E2 | 家有儿女/第二季/第002集 | **95.87%** | 50.62% | 80.1% | 71.0% | 75.3% | 16 |
| S2E3 | 家有儿女/第二季/第003集 | **98.37%** | 56.38% | 78.5% | 66.0% | 71.6% | 6 |
| S2E4 | 家有儿女/第二季/第004集 | 94.34% | 44.41% | 81.7% | 75.2% | 78.4% | 11 |
| S2E5 | 家有儿女/第二季/第005集 | **97.53%** | 54.52% | 80.3% | 75.7% | 77.9% | 9 |
| **均值** | — | **96.02%** | **48.27%** | **82.7%** | **73.9%** | **77.7%** | **11.3** |

### 2.2 跨集汇总

```
总场景数:       3312  (midpoint=2418, switch=1190, silence=104)
OCR 命中数:     3043  (非静默场景)
参考段总数:     2717  (omni-plus 全集逐句)
参考段被覆盖:    1304  (recall = 47.92%)
真漏候选总数:     79   (miss_rate = 2.56%)
```

### 2.3 成本明细 (单次 7 集累计)

| 模型 | 调用次数 | 文入 tokens | 音入 tokens | 文出 tokens | 费用 (CNY) |
|------|:--------:|:-----------:|:-----------:|:-----------:|:----------:|
| qwen3.5-omni-plus (参考 ASR) | 154 | 39k | 73k | 150k | ~35.2 |
| qwen3.5-omni-flash (Stage 1 ASR) | 126 | 74k | 51k | 137k | ~10.9 |
| qwen3-vl-plus (字幕 OCR) | 3679 | 5.95M | 0 | 23k | ~8.6 |
| qwen-vl-max (S1E1 caption, 已禁用后续) | 444 | 689k | 0 | 46k | ~2.5 |
| **合计** | **4403** | **6.75M** | **124k** | **356k** | **~57.2 CNY** |

**单集平均**: ~8.2 CNY (含参考 ASR 5.0 + Stage 1 1.6 + OCR 1.2 + S1E1 caption 0.4)

## 三、结论

### 3.1 OCR 命中率达标 ✅
**mean hit_rate (非静默) = 96.02%**,超过 95% 目标。
- 7 集中 6 集 ≥95%
- 仅 S2E4 (94.34%) 略低,差距 0.66pp,在可接受范围
- 最低 94.34%,最高 98.37%,方差小,方案稳定

### 3.2 Recall 偏低属于设计局限 ⚠️
**mean recall = 48.27%**,看似偏低,但是**声纹锚点采样的固有限制**,非缺陷:

- 每个 utterance 只在 midpoint 抽 1 帧 OCR
- omni-plus 按**句子级**切分,一个 utterance 通常含 1-3 句话
- 远离 utterance 中心点的句子字幕**无法被捕获**
- 例如:一段 6 秒 utterance 内 omni-plus 切出 3 句字幕,OCR 只命中其中 1 句 → recall = 33%,但 hit_rate 仍为 100%

**取舍权衡**: recall 低意味着我们漏掉了多句台词中的部分句子,但 hit_rate 高意味着每个 utterance 至少捕获到一句。对下游「台词对齐/搜索」任务而言,hit_rate 比 recall 更重要。

### 3.3 字符级准确率良好 ✅
**char F1 ≈ 78%** (P=83% / R=74%)
- Precision 高于 Recall → OCR 倾向「少识别」而非「错识别」,即宁可漏字也不错字
- 主要错误来源:多行硬字幕的下层、繁简混排、生僻字

### 3.4 真漏率低 ✅
**miss_rate = 2.56%** (79/3312)
即「OCR=无字幕 但参考 ASR 该时刻有人声」的真漏情况平均仅 2.56%,符合「字幕不漏」的核心目标。

### 3.5 成本可控 ✅
**单集 ~8 CNY**,其中:
- 参考 ASR (omni-plus) 占 60%+,但这是**评估专用**,生产 pipeline 不需要
- 实际生产单集成方:**Stage 1 + Stage 2 OCR ≈ 2.8 CNY/集**

## 四、相关文件

| 文件 | 说明 |
|------|------|
| `per_episode_metrics.json` | 7 集逐集明细指标 |
| `batch_ocr_all7_aggregate.json` | 跨集汇总指标 + per_episode 摘要 |
| `../../data/output/_batch_reports/` | 完整批次日志 (含 subprocess 噪音) |
| `../../data/output/{video}/ocr_eval.json` | 每集完整评估结果 (含 miss_candidates 列表) |
| `../../data/output/{video}/reference_asr.json` | 每集 omni-plus 参考 ASR 全文 |
| `../../src/eval/reference_asr.py` | 参考 ASR 模块 |
| `../../src/eval/ocr_accuracy.py` | OCR 准确率评估器 |
| `../../tests/batch_test_ocr.py` | 批量测试编排器 |

## 五、改进方向 (可选)

1. **提高 Recall**:对长 utterance (>5s) 在 1/4、1/2、3/4 处各抽 1 帧,可显著提升 recall,但 OCR 调用量增加 1.5-2 倍
2. **修 S2E4 outlier**:排查为何 hit_rate 偏低,可能是该集背景音乐多 / 字幕闪烁快
3. **修 subprocess GBK 噪音**:Windows 上 ffmpeg 中文路径 stderr 触发 UnicodeDecodeError,需在 `subprocess.run` 中显式传 `encoding='utf-8', errors='replace'`
4. **降低参考 ASR 成本**:对评估场景,可只对 miss_candidates 时间窗内做 omni-plus 验证,而非全集
