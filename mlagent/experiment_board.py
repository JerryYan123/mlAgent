"""ExperimentBoard and TodoList — shared state structures for advanced planning strategies."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class BoardEntry:
    entry_type: str  # context, insight, hypothesis, experiment, decision, finding
    content: str
    round_num: int | None = None
    step: int | None = None
    timestamp: float = field(default_factory=time.time)


class ExperimentBoard:
    """Accumulates structured knowledge across rounds.

    Entry types:
      context    — static competition/data info (populated once)
      insight    — data observations from exploration
      experiment — model run with score
      hypothesis — idea to test
      decision   — strategic choice by planner
      finding    — conclusion from results
    """

    def __init__(self) -> None:
        self.entries: list[BoardEntry] = []

    def add(
        self,
        entry_type: str,
        content: str,
        round_num: int | None = None,
        step: int | None = None,
    ) -> None:
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

    def experiments_summary(self) -> str:
        exps = [e for e in self.entries if e.entry_type == "experiment"]
        if not exps:
            return "No experiments recorded yet."
        return "\n".join(f"- {e.content}" for e in exps)

    def save(self, path: Path) -> None:
        data = [
            {
                "type": e.entry_type,
                "content": e.content,
                "round": e.round_num,
                "step": e.step,
                "ts": e.timestamp,
            }
            for e in self.entries
        ]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def parse_board_tags(text: str) -> list[tuple[str, str]]:
        """Extract [BOARD:type] entries from LLM output."""
        pattern = r"\[BOARD:(\w+)\]\s*(.+)"
        return re.findall(pattern, text)

    def update_from_llm(
        self, text: str, round_num: int | None = None, step: int | None = None
    ) -> None:
        for entry_type, content in self.parse_board_tags(text):
            self.add(entry_type.lower(), content.strip(), round_num, step)


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
