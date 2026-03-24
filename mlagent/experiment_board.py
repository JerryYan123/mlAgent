"""ExperimentBoard, ExperimentMap, and TodoList — shared state structures."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# ExperimentMap — tree-structured experiment record for board_replan v2
# ---------------------------------------------------------------------------

@dataclass
class MapEntry:
    branch: str
    content: str
    score: float | None = None
    entry_type: str = "experiment"  # "experiment" or "note"
    round_num: int | None = None
    timestamp: float = field(default_factory=time.time)


class ExperimentMap:
    """Tree-structured experiment record grouped by approach branch.

    - Coding agent appends experiment results via log_to_board tool.
    - Planner appends strategic notes via [MAP:branch] tags.
    - Append-only: nothing is ever deleted.
    """

    def __init__(self, is_lower_better: bool = True) -> None:
        self.entries: list[MapEntry] = []
        self.is_lower_better = is_lower_better

    def add_result(
        self,
        branch: str,
        experiment: str,
        result: str,
        score: float | None = None,
        round_num: int | None = None,
    ) -> None:
        content = f"{experiment}: {result}" if result else experiment
        self.entries.append(
            MapEntry(branch=branch, content=content, score=score,
                     entry_type="experiment", round_num=round_num)
        )

    def add_note(
        self, branch: str, note: str, round_num: int | None = None
    ) -> None:
        self.entries.append(
            MapEntry(branch=branch, content=note, score=None,
                     entry_type="note", round_num=round_num)
        )

    def _best_score(self, entries: list[MapEntry]) -> MapEntry | None:
        scored = [e for e in entries if e.score is not None]
        if not scored:
            return None
        if self.is_lower_better:
            return min(scored, key=lambda e: e.score)  # type: ignore[arg-type]
        return max(scored, key=lambda e: e.score)  # type: ignore[arg-type]

    def to_string(self) -> str:
        if not self.entries:
            return "(empty — no experiments recorded yet)"
        branches: dict[str, list[MapEntry]] = {}
        for e in self.entries:
            branches.setdefault(e.branch, []).append(e)

        lines: list[str] = []
        for branch, entries in branches.items():
            best = self._best_score(entries)
            best_str = f" (best: {best.score})" if best else ""
            lines.append(f"### {branch}{best_str}")
            for e in entries:
                prefix = "- [note] " if e.entry_type == "note" else "- "
                score_str = f" → {e.score}" if e.score is not None else ""
                lines.append(f"{prefix}{e.content}{score_str}")
        return "\n".join(lines)

    def branch_string(self, branch: str) -> str:
        entries = [e for e in self.entries if e.branch == branch]
        if not entries:
            return f"(no entries in '{branch}')"
        lines: list[str] = []
        for e in entries:
            prefix = "- [note] " if e.entry_type == "note" else "- "
            score_str = f" → {e.score}" if e.score is not None else ""
            lines.append(f"{prefix}{e.content}{score_str}")
        return "\n".join(lines)

    def has_experiments_from_round(self, round_num: int) -> bool:
        return any(
            e.round_num == round_num and e.entry_type == "experiment"
            for e in self.entries
        )

    def save(self, path: Path) -> None:
        data = [
            {
                "branch": e.branch,
                "content": e.content,
                "score": e.score,
                "type": e.entry_type,
                "round": e.round_num,
                "ts": e.timestamp,
            }
            for e in self.entries
        ]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def parse_map_tags(text: str) -> list[tuple[str, str]]:
        """Extract [MAP:branch] note lines from planner output."""
        pattern = r"\[MAP:([^\]]+)\]\s*(.+)"
        return re.findall(pattern, text)

    def update_from_planner(self, text: str, round_num: int | None = None) -> None:
        for branch, note in self.parse_map_tags(text):
            self.add_note(branch.strip(), note.strip(), round_num)


# ---------------------------------------------------------------------------
# Legacy ExperimentBoard (kept for backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class BoardEntry:
    entry_type: str
    content: str
    round_num: int | None = None
    step: int | None = None
    timestamp: float = field(default_factory=time.time)


class ExperimentBoard:
    """Legacy flat board. Kept for old runs; new code uses ExperimentMap."""

    def __init__(self) -> None:
        self.entries: list[BoardEntry] = []

    def add(self, entry_type: str, content: str,
            round_num: int | None = None, step: int | None = None) -> None:
        self.entries.append(BoardEntry(entry_type, content, round_num, step))

    def to_string(self) -> str:
        if not self.entries:
            return "(empty board)"
        lines: list[str] = []
        for e in self.entries:
            tag = e.entry_type.upper()
            loc = ""
            if e.round_num is not None:
                loc = f" R{e.round_num}"
                if e.step is not None:
                    loc += f"/S{e.step}"
            lines.append(f"[{tag}{loc}] {e.content}")
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        data = [
            {"type": e.entry_type, "content": e.content,
             "round": e.round_num, "step": e.step, "ts": e.timestamp}
            for e in self.entries
        ]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# TodoList for codex_todo strategy
# ---------------------------------------------------------------------------

@dataclass
class TodoItem:
    index: int
    task: str
    hypothesis: str
    status: str = "pending"  # pending, in_progress, done, skipped
    result: str = ""


class TodoList:
    """Structured task list managed by planner, executed by coder."""

    def __init__(self) -> None:
        self.items: list[TodoItem] = []

    def add(self, task: str, hypothesis: str = "") -> int:
        idx = len(self.items)
        self.items.append(TodoItem(index=idx, task=task, hypothesis=hypothesis))
        return idx

    def update(self, index: int, status: str, result: str = "") -> bool:
        if 0 <= index < len(self.items):
            self.items[index].status = status
            if result:
                self.items[index].result = result
            return True
        return False

    def pending_items(self) -> list[TodoItem]:
        return [it for it in self.items if it.status == "pending"]

    def to_string(self) -> str:
        if not self.items:
            return "(empty todo list)"
        lines: list[str] = []
        status_icon = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]", "skipped": "[-]"}
        for it in self.items:
            icon = status_icon.get(it.status, "[ ]")
            line = f"{icon} {it.index}. {it.task}"
            if it.hypothesis:
                line += f"  (why: {it.hypothesis})"
            if it.result:
                line += f"  → {it.result}"
            lines.append(line)
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        data = [
            {
                "index": it.index,
                "task": it.task,
                "hypothesis": it.hypothesis,
                "status": it.status,
                "result": it.result,
            }
            for it in self.items
        ]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def parse_todo_list(text: str) -> list[tuple[str, str]]:
        """Extract todo items from LLM output.

        Expects lines like:
          - [ ] Task description (why: hypothesis)
        or numbered:
          1. Task description (why: hypothesis)
        """
        items: list[tuple[str, str]] = []
        for line in text.split("\n"):
            line = line.strip()
            cleaned = re.sub(r"^[-*]\s*\[.\]\s*\d*\.?\s*", "", line)
            cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
            if not cleaned or cleaned == line:
                continue
            hyp_match = re.search(r"\(why:\s*(.+?)\)\s*$", cleaned)
            hypothesis = hyp_match.group(1) if hyp_match else ""
            task = re.sub(r"\s*\(why:.*?\)\s*$", "", cleaned).strip()
            if task:
                items.append((task, hypothesis))
        return items

    def replace_from_llm(self, text: str) -> None:
        parsed = self.parse_todo_list(text)
        if not parsed:
            return
        old_results = {it.task: (it.status, it.result) for it in self.items}
        self.items = []
        for task, hypothesis in parsed:
            idx = self.add(task, hypothesis)
            if task in old_results:
                status, result = old_results[task]
                self.items[idx].status = status
                self.items[idx].result = result
