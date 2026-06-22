"""评估工具包 — 用于 OCR 字幕识别的批量测试.

模块:
  - reference_asr: 用 omni-plus 跑全集高准确率 ASR, 作为软 GT
  - ocr_accuracy:  对比 OCR 结果与 reference, 计算 precision/recall/hit_rate
"""
