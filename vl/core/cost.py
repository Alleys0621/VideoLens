"""LLM 调用代价追踪器

单例模式，全局追踪所有 LLM API 调用的 token 用量、费用和延迟。
支持按模型汇总，按阶段分组，输出结构化报告。
"""

import time
from dataclasses import dataclass, field


@dataclass
class CallRecord:
    """单次 API 调用记录"""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0          # CNY
    latency: float = 0.0       # 秒
    stage: str = ""            # 所属阶段


@dataclass
class ModelPricing:
    """模型定价 (CNY / 百万 tokens)"""
    input_per_m: float = 0.0
    output_per_m: float = 0.0

    def calculate(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_m + output_tokens * self.output_per_m) / 1_000_000


# ──────────────────────────────────────────────────────────
# DashScope Qwen 系列默认定价 (CNY / 百万 tokens)
# 来源: https://help.aliyun.com/zh/model-studio/getting-started/models
# ──────────────────────────────────────────────────────────

DEFAULT_PRICING: dict[str, dict] = {
    "qwen-plus":              {"input": 0.8,  "output": 2.0},
    "qwen-turbo":             {"input": 0.3,  "output": 0.6},
    "qwen-max":               {"input": 2.0,  "output": 6.0},
    "qwen-vl-max":            {"input": 3.0,  "output": 9.0},
    "qwen-vl-plus":           {"input": 1.5,  "output": 4.5},
    "qwen3.5-omni-plus":      {"input": 2.0,  "output": 6.0},
    "qwen3.5-omni-flash":     {"input": 0.5,  "output": 1.5},
    "qwen3-asr-flash":        {"input": 0.5,  "output": 1.0},
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
                input_per_m=prices["input"],
                output_per_m=prices["output"],
            )

    def configure_pricing(self, pricing_config: dict):
        """从 pipeline.yaml 加载自定义定价 (覆盖默认值)

        配置格式:
          pricing:
            qwen-plus:
              input_per_m: 0.8
              output_per_m: 2.0
        """
        for model, prices in pricing_config.items():
            self._pricing[model] = ModelPricing(
                input_per_m=prices.get("input_per_m", prices.get("input", 0)),
                output_per_m=prices.get("output_per_m", prices.get("output", 0)),
            )

    def get_pricing(self, model: str) -> ModelPricing:
        """获取模型定价 (支持模糊匹配模型名前缀)"""
        # 精确匹配
        if model in self._pricing:
            return self._pricing[model]
        # 前缀匹配 (如 qwen-plus-latest → qwen-plus)
        for key in self._pricing:
            if model.startswith(key):
                return self._pricing[key]
        # 无匹配，免费
        return ModelPricing()

    # ── 记录 ──────────────────────────────────────────────

    def record(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency: float = 0.0,
        stage: str = "",
    ):
        """记录一次 API 调用"""
        pricing = self.get_pricing(model)
        cost = pricing.calculate(input_tokens, output_tokens)

        self._records.append(CallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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
                    "calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "cost": 0.0, "latency": 0.0,
                }
            m = models[r.model]
            m["calls"] += 1
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
                    "calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "cost": 0.0, "latency": 0.0,
                }
            s = stages[stage]
            s["calls"] += 1
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
        lines.append("=" * 65)
        lines.append("  LLM 调用代价报告")
        lines.append("=" * 65)

        # 按模型汇总
        by_model = self.summary_by_model()
        lines.append("")
        lines.append(f"  {'模型':<25s} {'调用':>5s} {'输入':>10s} {'输出':>10s} {'费用(CNY)':>10s} {'延迟(s)':>8s}")
        lines.append("  " + "-" * 70)

        for model, m in sorted(by_model.items(), key=lambda x: -x[1]["cost"]):
            lines.append(
                f"  {model:<25s} {m['calls']:>5d} "
                f"{m['input_tokens']:>10,d} {m['output_tokens']:>10,d} "
                f"{m['cost']:>10.4f} {m['latency']:>8.1f}"
            )

        lines.append("  " + "-" * 70)
        lines.append(
            f"  {'合计':<25s} {self.total_calls:>5d} "
            f"{self.total_input_tokens:>10,d} {self.total_output_tokens:>10,d} "
            f"{self.total_cost:>10.4f} {self.total_latency:>8.1f}"
        )

        # 按阶段汇总 (如果有)
        by_stage = self.summary_by_stage()
        if by_stage and len(by_stage) > 1:
            lines.append("")
            lines.append(f"  {'阶段':<15s} {'调用':>5s} {'输入':>10s} {'输出':>10s} {'费用(CNY)':>10s}")
            lines.append("  " + "-" * 55)
            for stage, s in sorted(by_stage.items(), key=lambda x: -x[1]["cost"]):
                lines.append(
                    f"  {stage:<15s} {s['calls']:>5d} "
                    f"{s['input_tokens']:>10,d} {s['output_tokens']:>10,d} "
                    f"{s['cost']:>10.4f}"
                )

        lines.append("")
        lines.append(f"  总费用: ¥{self.total_cost:.4f} | 总 tokens: {self.total_tokens:,d} | 总调用: {self.total_calls} 次")
        lines.append("=" * 65)

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
