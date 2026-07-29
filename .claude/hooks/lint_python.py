"""PostToolUse lint: 检查 Claude 新增的 Python 代码是否违反 CLAUDE.md 代码规范.

检查 5 项 (CLAUDE.md "代码规范（强制）" 段):
  1. os.getenv / os.environ 直读 (仅 src/core/config.py 允许)
  2. 业务代码 import dashscope (仅 retriever.py / asr_server.py 允许)
  3. print() 调用 (仅 src/eval/* 允许, 离线脚本)
  4. 裸 except: / except Exception: 不跟 as
  5. Optional[X] (应该用 X | None)

输入 (stdin JSON, Claude Code hook 协议):
  Write: { tool_name, tool_input: { file_path, content } }
  Edit:  { tool_name, tool_input: { file_path, old_string, new_string } }

输出 (stdout JSON):
  无违规: {} (空对象)
  有违规: { "hookSpecificOutput": { "hookEventName": "PostToolUse",
                                    "additionalContext": "<反馈文本>" } }

退出码: 始终 0 (不阻塞工具调用, 通过 additionalContext 反馈让 Claude 自动修)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


# ---------- 白名单 ----------
# 规则按 (regex, whitelist_paths) 组织. whitelist 是 POSIX 风格路径片段匹配.
RULES = [
    {
        "name": "os.getenv/os.environ 直读",
        "fix": "改用 `from src.core.config import get_config; cfg = get_config()` 读 AppConfig 字段",
        "pattern": re.compile(r"\bos\.(getenv|environ)\b"),
        "whitelist": {"src/core/config.py"},  # 唯一允许 env 读取的入口
    },
    {
        "name": "业务代码 import dashscope",
        "fix": "改用 BaseLLMClient / QwenTextClient / QwenVLClient (src/core/llm/)",
        "pattern": re.compile(r"^\s*import\s+dashscope\b|^\s*from\s+dashscope\b", re.MULTILINE),
        "whitelist": {"src/agent/retriever.py", "src/agent/asr_server.py"},
    },
    {
        "name": "print() 调用",
        "fix": "改用 `from src.core.logging import get_logger; logger = get_logger()` + logger.info/warning",
        "pattern": re.compile(r"^\s*print\s*\(", re.MULTILINE),
        "whitelist_dir": "src/eval/",  # 离线研究脚本允许 print
    },
    {
        "name": "裸 except: 或 except Exception: 缺 'as e'",
        "fix": "改成 `except Exception as e:` + logger.warning/exec",
        "pattern": re.compile(r"\bexcept\s+Exception\s*:|\bexcept\s*:"),
        "whitelist": set(),  # 无例外
    },
    {
        "name": "Optional[X] (旧式类型注解)",
        "fix": "改成新式 `X | None`",
        "pattern": re.compile(r"\bOptional\["),
        "whitelist": set(),
    },
]


def _normalize(path: str) -> str:
    """转 POSIX 风格相对路径 (相对项目根)."""
    p = Path(path).resolve()
    cwd = Path.cwd().resolve()
    try:
        rel = p.relative_to(cwd)
    except ValueError:
        return path.replace("\\", "/")
    return str(rel).replace("\\", "/")


def _in_whitelist(rel_path: str, rule: dict) -> bool:
    if rel_path in rule.get("whitelist", set()):
        return True
    wl_dir = rule.get("whitelist_dir")
    if wl_dir and rel_path.startswith(wl_dir):
        return True
    return False


def _scan(text: str, rel_path: str) -> list[tuple[str, str, int]]:
    """返回 [(rule_name, fix, line_number), ...]"""
    findings: list[tuple[str, str, int]] = []
    for rule in RULES:
        if _in_whitelist(rel_path, rule):
            continue
        for m in rule["pattern"].finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            findings.append((rule["name"], rule["fix"], line_no))
    return findings


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)  # 输入异常不阻塞

    tool_input = payload.get("tool_input", {})
    file_path = tool_input.get("file_path", "")
    if not file_path or not file_path.endswith(".py"):
        sys.exit(0)

    # Write 扫 content; Edit 只扫 new_string (只查新增违规, 不报历史遗留)
    tool_name = payload.get("tool_name", "")
    if tool_name == "Write":
        text = tool_input.get("content", "")
    elif tool_name == "Edit":
        text = tool_input.get("new_string", "")
    else:
        sys.exit(0)

    if not text:
        sys.exit(0)

    rel_path = _normalize(file_path)
    findings = _scan(text, rel_path)

    if not findings:
        sys.exit(0)

    # 按 rule_name 聚合
    by_rule: dict[str, list[tuple[str, int]]] = {}
    for name, fix, line_no in findings:
        by_rule.setdefault(name, (fix, []))[1].append(line_no)  # type: ignore[assignment]
    # 上面 setdefault 用 tuple 不便, 改清晰结构:
    by_rule = {}
    for name, fix, line_no in findings:
        if name not in by_rule:
            by_rule[name] = (fix, [])
        by_rule[name][1].append(line_no)

    lines = [f"检测到 {len(findings)} 处可能违反 CLAUDE.md 代码规范 ({rel_path}):"]
    for name, (fix, line_nos) in by_rule.items():
        nos = ",".join(str(n) for n in line_nos[:5])
        if len(line_nos) > 5:
            nos += f",... (+{len(line_nos) - 5})"
        lines.append(f"  - 行 {nos}: {name}")
        lines.append(f"    修复: {fix}")
    lines.append("请修复以上违规. 若属于合理的白名单情况 (如新增 eval/* 脚本), 可忽略.")

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": "\n".join(lines),
        }
    }
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
