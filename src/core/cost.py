"""LLM 调用代价追踪器

单例模式，全局追踪所有 LLM API 调用的 token 用量、费用和延迟。
支持按模型汇总，按阶段分组，输出结构化报告。

定价设计要点：
  Qwen-Omni 系列（qwen3.5-omni-plus / flash）按 **输入模态** 分别计价：
  文本输入、音频输入、图片/视频输入各有独立费率；输出按模态（文本/音频）分别计价。
  官方价格见 https://help.aliyun.com/zh/model-studio/model-pricing

  纯文本/视觉模型（qwen-plus / qwen-vl-max 等）可只用 text_input + text_output 字段，
  其余模态字段为 0。ModelPricing.calculate() 对 0 单价的模态自动跳过。
"""

from dataclasses import dataclass, field


@dataclass
class CallRecord:
    """单次 API 调用记录"""
    model: str
    # 分模态输入 token
    text_tokens_in: int = 0
    audio_tokens_in: int = 0
    image_tokens_in: int = 0
    # 输出 token (目前只用到文本输出)
    text_tokens_out: int = 0
    audio_tokens_out: int = 0
    cost: float = 0.0          # CNY
    latency: float = 0.0       # 秒
    stage: str = ""            # 所属阶段

    @property
    def input_tokens(self) -> int:
        return self.text_tokens_in + self.audio_tokens_in + self.image_tokens_in

    @property
    def output_tokens(self) -> int:
        return self.text_tokens_out + self.audio_tokens_out


@dataclass
class ModelPricing:
    """模型分模态定价 (CNY / 百万 tokens)

    未使用的模态单价保持 0，calculate() 会自动跳过。
    """
    text_input_per_m: float = 0.0
    audio_input_per_m: float = 0.0
    image_input_per_m: float = 0.0
    text_output_per_m: float = 0.0
    audio_output_per_m: float = 0.0

    def calculate(
        self,
        text_tokens_in: int = 0,
        audio_tokens_in: int = 0,
        image_tokens_in: int = 0,
        text_tokens_out: int = 0,
        audio_tokens_out: int = 0,
    ) -> float:
        """按各模态单价分别计价后求和，单价为 0 的模态自动跳过。"""
        cost = 0.0
        if text_tokens_in and self.text_input_per_m:
            cost += text_tokens_in * self.text_input_per_m / 1_000_000
        if audio_tokens_in and self.audio_input_per_m:
            cost += audio_tokens_in * self.audio_input_per_m / 1_000_000
        if image_tokens_in and self.image_input_per_m:
            cost += image_tokens_in * self.image_input_per_m / 1_000_000
        if text_tokens_out and self.text_output_per_m:
            cost += text_tokens_out * self.text_output_per_m / 1_000_000
        if audio_tokens_out and self.audio_output_per_m:
            cost += audio_tokens_out * self.audio_output_per_m / 1_000_000
        return cost


# ──────────────────────────────────────────────────────────
# DashScope Qwen 系列默认定价 (CNY / 百万 tokens, 中国内地)
# 来源: https://help.aliyun.com/zh/model-studio/model-pricing
# ──────────────────────────────────────────────────────────

# 每个条目字段:
#   text_input / audio_input / image_input / text_output / audio_output
DEFAULT_PRICING: dict[str, dict] = {
    # --- 纯文本模型 ---
    "qwen-plus":          {"text_input": 0.8, "text_output": 2.0},
    "qwen-turbo":         {"text_input": 0.3, "text_output": 0.6},
    "qwen-max":           {"text_input": 2.0, "text_output": 6.0},

    # --- 视觉模型 (文本/图片同档输入价) ---
    "qwen-vl-max":        {"text_input": 3.0, "image_input": 3.0, "text_output": 9.0},
    "qwen-vl-plus":       {"text_input": 1.5, "image_input": 1.5, "text_output": 4.5},

    # --- Qwen3-VL (输入便宜, 输出贵, 适合 OCR 这类输入大输出小的任务) ---
    "qwen3-vl-plus":      {"text_input": 1.4, "image_input": 1.4, "text_output": 11.2},
    "qwen3-vl-flash":     {"text_input": 0.6, "image_input": 0.6, "text_output": 4.8},

    # --- Omni 模型 (按输入模态分档) ---
    "qwen3.5-omni-plus":  {"text_input": 7.0,  "audio_input": 53.0,  "image_input": 40.0,  "text_output": 213.0},
    "qwen3.5-omni-flash": {"text_input": 2.2,  "audio_input": 18.0,  "image_input": 13.3,  "text_output": 72.0},
    "qwen3-omni-flash":   {"text_input": 1.8,  "audio_input": 15.8,  "image_input": 3.3,   "text_output": 12.7},
    "qwen-omni-turbo":    {"text_input": 0.4,  "audio_input": 25.0,  "image_input": 1.5,   "text_output": 4.5},

    # --- ASR ---
    "qwen3-asr-flash":    {"audio_input": 0.5, "text_output": 1.0},
}


class CostTracker:
    """全局 LLM 调用代价追踪器"""

    def __init__(self):
        self._records: list[CallRecord] = []
        self._pricing: dict[str, ModelPricing] = {}
        self._load_default_pricing()

    # ── 配置 ──────────────────────────────────────────────

    def _load_default_pricing(self):
        for model, prices in DEFAULT_PRICING.items():
            self._pricing[model] = ModelPricing(
                text_input_per_m=prices.get("text_input", 0.0),
                audio_input_per_m=prices.get("audio_input", 0.0),
                image_input_per_m=prices.get("image_input", 0.0),
                text_output_per_m=prices.get("text_output", 0.0),
                audio_output_per_m=prices.get("audio_output", 0.0),
            )

    def configure_pricing(self, pricing_config: dict):
        """从 pipeline.yaml 加载自定义定价 (覆盖默认值)

        配置格式 (单档模型可省略未用模态):
          pricing:
            qwen-plus:
              text_input: 0.8
              text_output: 2.0
            qwen3.5-omni-flash:
              text_input: 2.2
              audio_input: 18.0
              image_input: 13.3
              text_output: 72.0
        """
        for model, prices in pricing_config.items():
            self._pricing[model] = ModelPricing(
                text_input_per_m=prices.get("text_input", prices.get("input", 0.0)),
                audio_input_per_m=prices.get("audio_input", 0.0),
                image_input_per_m=prices.get("image_input", 0.0),
                text_output_per_m=prices.get("text_output", prices.get("output", 0.0)),
                audio_output_per_m=prices.get("audio_output", 0.0),
            )

    def get_pricing(self, model: str) -> ModelPricing:
        """获取模型定价 (支持模糊匹配模型名前缀)"""
        # 精确匹配
        if model in self._pricing:
            return self._pricing[model]
        # 前缀匹配 (如 qwen3.5-omni-flash-2026-03-15 → qwen3.5-omni-flash)
        for key in self._pricing:
            if model.startswith(key):
                return self._pricing[key]
        # 无匹配，免费
        return ModelPricing()

    # ── 记录 ──────────────────────────────────────────────

    def record(
        self,
        model: str,
        text_tokens_in: int = 0,
        audio_tokens_in: int = 0,
        image_tokens_in: int = 0,
        text_tokens_out: int = 0,
        audio_tokens_out: int = 0,
        latency: float = 0.0,
        stage: str = "",
    ):
        """记录一次 API 调用

        参数均为分模态 token 数；模型未启用的模态传 0 即可。
        """
        pricing = self.get_pricing(model)
        cost = pricing.calculate(
            text_tokens_in=text_tokens_in,
            audio_tokens_in=audio_tokens_in,
            image_tokens_in=image_tokens_in,
            text_tokens_out=text_tokens_out,
            audio_tokens_out=audio_tokens_out,
        )

        self._records.append(CallRecord(
            model=model,
            text_tokens_in=text_tokens_in,
            audio_tokens_in=audio_tokens_in,
            image_tokens_in=image_tokens_in,
            text_tokens_out=text_tokens_out,
            audio_tokens_out=audio_tokens_out,
            cost=cost,
            latency=latency,
            stage=stage,
        ))

    # ── 汇总 ──────────────────────────────────────────────

    @property
    def total_calls(self) -> int:
        return len(self._records)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self._records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self._records)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self._records)

    @property
    def total_latency(self) -> float:
        return sum(r.latency for r in self._records)

    def summary_by_model(self) -> dict[str, dict]:
        """按模型汇总"""
        models: dict[str, dict] = {}
        for r in self._records:
            if r.model not in models:
                models[r.model] = {
                    "calls": 0,
                    "text_in": 0, "audio_in": 0, "image_in": 0,
                    "text_out": 0, "audio_out": 0,
                    "input_tokens": 0, "output_tokens": 0,
                    "cost": 0.0, "latency": 0.0,
                }
            m = models[r.model]
            m["calls"] += 1
            m["text_in"] += r.text_tokens_in
            m["audio_in"] += r.audio_tokens_in
            m["image_in"] += r.image_tokens_in
            m["text_out"] += r.text_tokens_out
            m["audio_out"] += r.audio_tokens_out
            m["input_tokens"] += r.input_tokens
            m["output_tokens"] += r.output_tokens
            m["cost"] += r.cost
            m["latency"] += r.latency
        return models

    def summary_by_stage(self) -> dict[str, dict]:
        """按阶段汇总"""
        stages: dict[str, dict] = {}
        for r in self._records:
            stage = r.stage or "unknown"
            if stage not in stages:
                stages[stage] = {
                    "calls": 0,
                    "text_in": 0, "audio_in": 0, "image_in": 0,
                    "text_out": 0, "audio_out": 0,
                    "input_tokens": 0, "output_tokens": 0,
                    "cost": 0.0, "latency": 0.0,
                }
            s = stages[stage]
            s["calls"] += 1
            s["text_in"] += r.text_tokens_in
            s["audio_in"] += r.audio_tokens_in
            s["image_in"] += r.image_tokens_in
            s["text_out"] += r.text_tokens_out
            s["audio_out"] += r.audio_tokens_out
            s["input_tokens"] += r.input_tokens
            s["output_tokens"] += r.output_tokens
            s["cost"] += r.cost
            s["latency"] += r.latency
        return stages

    def reset(self):
        """重置所有记录"""
        self._records.clear()

    # ── 报告 ──────────────────────────────────────────────

    def report(self) -> str:
        """生成可读的代价报告"""
        if not self._records:
            return "(无 LLM 调用记录)"

        lines = []
        lines.append("=" * 95)
        lines.append("  LLM 调用代价报告")
        lines.append("=" * 95)

        # 按模型汇总
        by_model = self.summary_by_model()
        lines.append("")
        header = (
            f"  {'模型':<22s} {'调用':>4s} "
            f"{'文入':>9s} {'音入':>9s} {'图入':>9s} "
            f"{'文出':>9s} {'费用(CNY)':>11s} {'延迟(s)':>8s}"
        )
        lines.append(header)
        lines.append("  " + "-" * 92)

        for model, m in sorted(by_model.items(), key=lambda x: -x[1]["cost"]):
            lines.append(
                f"  {model:<22s} {m['calls']:>4d} "
                f"{m['text_in']:>9,d} {m['audio_in']:>9,d} {m['image_in']:>9,d} "
                f"{m['text_out']:>9,d} {m['cost']:>11.4f} {m['latency']:>8.1f}"
            )

        lines.append("  " + "-" * 92)
        lines.append(
            f"  {'合计':<22s} {self.total_calls:>4d} "
            f"{sum(m['text_in'] for m in by_model.values()):>9,d} "
            f"{sum(m['audio_in'] for m in by_model.values()):>9,d} "
            f"{sum(m['image_in'] for m in by_model.values()):>9,d} "
            f"{self.total_output_tokens:>9,d} "
            f"{self.total_cost:>11.4f} {self.total_latency:>8.1f}"
        )

        # 按阶段汇总 (如果有)
        by_stage = self.summary_by_stage()
        if by_stage and len(by_stage) > 1:
            lines.append("")
            lines.append(
                f"  {'阶段':<18s} {'调用':>4s} "
                f"{'文入':>9s} {'音入':>9s} {'图入':>9s} "
                f"{'文出':>9s} {'费用(CNY)':>11s}"
            )
            lines.append("  " + "-" * 75)
            for stage, s in sorted(by_stage.items(), key=lambda x: -x[1]["cost"]):
                lines.append(
                    f"  {stage:<18s} {s['calls']:>4d} "
                    f"{s['text_in']:>9,d} {s['audio_in']:>9,d} {s['image_in']:>9,d} "
                    f"{s['text_out']:>9,d} {s['cost']:>11.4f}"
                )

        lines.append("")
        lines.append(
            f"  总费用: {self.total_cost:.4f} CNY | "
            f"总 tokens: {self.total_tokens:,d} | 总调用: {self.total_calls} 次"
        )
        lines.append("=" * 95)

        return "\n".join(lines)

    def report_dict(self) -> dict:
        """生成结构化报告 dict"""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_cny": round(self.total_cost, 6),
            "total_latency_s": round(self.total_latency, 2),
            "by_model": self.summary_by_model(),
            "by_stage": self.summary_by_stage(),
        }


# ──────────────────────────────────────────────────────────
# 全局单例
# ──────────────────────────────────────────────────────────

_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """获取全局代价追踪器"""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker


def reset_cost_tracker():
    """重置全局追踪器 (用于新的一次完整运行)"""
    global _tracker
    if _tracker is not None:
        _tracker.reset()
