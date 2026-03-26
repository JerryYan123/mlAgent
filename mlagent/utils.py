"""Logging, prompt traces, experiment log, text helpers."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("mlagent")


def setup_logging(level: int = logging.INFO) -> None:
    if not logging.root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    try:
        import litellm
        litellm.suppress_debug_info = True
    except Exception:
        pass
    os.environ.setdefault("LITELLM_LOG", "ERROR")


def process_backspace_chars(text: str) -> str:
    """Strip ANSI and approximate tqdm carriage-return lines (from comind-new)."""
    if not text:
        return text
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", text)
    lines = text.split("\n")
    out_lines = []
    for line in lines:
        if "\r" in line:
            line = line.split("\r")[-1]
        if "\b" in line:
            buf: list[str] = []
            for ch in line:
                if ch == "\b":
                    if buf:
                        buf.pop()
                else:
                    buf.append(ch)
            line = "".join(buf)
        out_lines.append(line)
    return "\n".join(out_lines)


def auto_select_gpu() -> str:
    """Pick GPU with most free memory (nvidia-smi)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return "0"
        best_idx, best_free = "0", -1.0
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            idx, used, total = parts[0], float(parts[1]), float(parts[2])
            free = total - used
            if free > best_free:
                best_free = free
                best_idx = idx
        return best_idx
    except Exception:
        return "0"


class PromptTracer:
    def __init__(self, out_dir: Path | None, enabled: bool = True) -> None:
        self.out_dir = out_dir
        self.enabled = enabled and out_dir is not None
        self._counter = 0
        if self.enabled:
            out_dir.mkdir(parents=True, exist_ok=True)

    def write(self, phase: str, prompt: str, response: str) -> None:
        if not self.enabled or self.out_dir is None:
            return
        self._counter += 1
        path = self.out_dir / f"{self._counter:04d}_{phase}.txt"
        path.write_text(
            f"phase={phase}\n"
            f"timestamp={time.time()}\n\n"
            f"--- PROMPT ---\n{prompt}\n\n--- RESPONSE ---\n{response}\n",
            encoding="utf-8",
        )


class ExperimentLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rounds: list[dict[str, Any]] = []
        if path.exists():
            try:
                self.rounds = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.rounds = []

    def log_round(
        self,
        round_num: int,
        plan: str,
        coding_summary: str,
        grade: Any,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        row: dict[str, Any] = {
            "round": round_num,
            "plan": plan,
            "coding_summary": coding_summary,
        }
        if grade is not None:
            if is_dataclass(grade):
                row["grade"] = asdict(grade)
            elif hasattr(grade, "__dict__"):
                row["grade"] = {
                    "score": getattr(grade, "score", None),
                    "valid_submission": getattr(grade, "valid_submission", None),
                    "error": getattr(grade, "error", None),
                    "medal": getattr(grade, "medal", None),
                }
            else:
                row["grade"] = str(grade)
        if extra:
            row.update(extra)
        self.rounds.append(row)
        self.path.write_text(json.dumps(self.rounds, indent=2, ensure_ascii=False), encoding="utf-8")


def format_summary(
    coding_summary: str,
    grade: Any,
    best_score: float | None,
    round_num: int,
    is_lower_better: bool = False,
) -> str:
    parts = [f"=== After round {round_num} ===", f"Coding agent summary:\n{coding_summary}"]
    if grade is not None:
        sc = getattr(grade, "score", None)
        valid = getattr(grade, "valid_submission", True)
        err = getattr(grade, "error", None)
        parts.append(f"Grading: score={sc}, valid={valid}, error={err}")
    if best_score is not None:
        direction = "lower is better" if is_lower_better else "higher is better"
        parts.append(f"Best score so far: {best_score} ({direction})")
    return "\n\n".join(parts)


def format_parallel_summary(
    agent_summaries: list[str],
    grades: list[Any],
    best_score: float | None,
    round_num: int,
    is_lower_better: bool = False,
) -> str:
    """Merge results from N parallel coding agents into one summary for the planner."""
    parts = [f"=== After round {round_num} ({len(agent_summaries)} agent(s)) ==="]
    for i, (summary, grade) in enumerate(zip(agent_summaries, grades)):
        sc = getattr(grade, "score", None)
        valid = getattr(grade, "valid_submission", False)
        err = getattr(grade, "error", None)
        parts.append(
            f"--- Agent {i} ---\n{summary}\n"
            f"Grade: score={sc}, valid={valid}, error={err}"
        )
    if best_score is not None:
        direction = "lower is better" if is_lower_better else "higher is better"
        parts.append(f"Global best score so far: {best_score} ({direction})")
    return "\n\n".join(parts)


def extract_xml_tag(text: str, tag: str) -> str | None:
    """Extract first <tag>...</tag> content."""
    pat = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


class TokenTracker:
    """Accumulates token usage across LLM calls. Thread-safe for parallel agents."""

    _COST_PER_1M: dict[str, tuple[float, float]] = {
        # Flagship models (standard tier) — prices as of 2026-03
        "gpt-5.4-pro": (30.00, 180.00),
        "gpt-5.4-nano": (0.20, 1.25),
        "gpt-5.4-mini": (0.75, 4.50),
        "gpt-5.4": (2.50, 15.00),
        # Specialized models
        "gpt-5.3-codex": (1.75, 14.00),
        "gpt-5.3-chat": (1.75, 14.00),
        # Legacy models
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "o3-mini": (1.10, 4.40),
    }

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self.input = 0
        self.output = 0

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.input += prompt_tokens
            self.output += completion_tokens

    def snapshot(self) -> tuple[int, int]:
        """Return (input, output) totals so far — use for computing deltas."""
        with self._lock:
            return self.input, self.output

    @classmethod
    def estimate_cost(cls, input_tokens: int, output_tokens: int, model: str) -> float | None:
        key = model.lower()
        for name, (inp_cost, out_cost) in cls._COST_PER_1M.items():
            if name in key:
                return (input_tokens * inp_cost + output_tokens * out_cost) / 1_000_000
        return None
