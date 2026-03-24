"""Coding agent: tool-calling loop with JupyterExecutor."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from mlagent.config import LLMConfig
from mlagent.jupyter_executor import ExecutionResult, JupyterExecutor
from mlagent.llm import ToolCallingLLM
from mlagent.utils import PromptTracer

logger = logging.getLogger(__name__)


def build_tools(submission_name: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_cell",
                "description": "Run Python in the persistent Jupyter kernel. State persists across calls.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "goal": {"type": "string", "description": "What this cell should accomplish"},
                    },
                    "required": ["code", "goal"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_cell",
                "description": (
                    "Edit a previously executed cell by search-and-replace, "
                    "then re-execute the modified code. Use for bug fixes or small parameter changes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_index": {
                            "type": "integer",
                            "description": "0-based index of the cell to edit (from execution history)",
                        },
                        "search": {"type": "string", "description": "Exact text to find in the cell"},
                        "replace": {"type": "string", "description": "Replacement text"},
                        "goal": {"type": "string", "description": "What this edit should accomplish"},
                    },
                    "required": ["cell_index", "search", "replace", "goal"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_submission",
                "description": f"Check if {submission_name} exists in the workspace; returns head/shape info.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file under the working directory (relative path).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_lines": {"type": "integer", "default": 120},
                    },
                    "required": ["path"],
                },
            },
        },
    ]


def _coding_system(plan: str, competition: Any, work_dir: Path, submission_name: str) -> str:
    desc = getattr(competition, "description", "")[:12000]
    return f"""You are the coding agent. Work in the Jupyter kernel (folder: {work_dir}).
Data is under ./input/ (symlink to competition data). Write submission to ./{submission_name}.

Competition description (excerpt):
{desc}

Plan from planning agent:
{plan}

You MUST use tools. Available tools: execute_cell, edit_cell, check_submission, read_file.

## Workflow
- Start by loading and briefly exploring the data.
- Build a working baseline that produces a valid {submission_name} as early as possible.
- Once you have a valid submission, iterate to improve: tune hyperparameters, try different
  features or models, and check the effect on your validation metric.
- Use check_submission to verify your output file before finishing.

## Bug Handling
- If a cell errors, read the traceback carefully and fix the specific issue.
- Use edit_cell to make small fixes to a previous cell instead of rewriting from scratch.
- If the same error keeps recurring, simplify your approach.

## Iteration Strategy
- After getting a working model, try variations: regularization strength, learning rate,
  feature count, number of folds, different algorithms.
- Keep changes small and measurable — change one thing at a time when tuning.
- Always print your validation metric so you can track improvements.

## Rules
- Use relative paths; data is under input/
- Each execute_cell should be focused on one task
- The kernel state persists: variables from earlier cells are available in later ones
- Prioritize having a valid {submission_name} over complex approaches"""


class CodingAgent:
    def __init__(
        self,
        coding_cfg: LLMConfig,
        tracer: Optional[PromptTracer] = None,
        agent_idx: int = 0,
    ) -> None:
        self.coding_cfg = coding_cfg
        self.tracer = tracer
        self.agent_idx = agent_idx
        self._tag = f"[Agent {agent_idx}]"

    def run_round(
        self,
        plan: str,
        competition: Any,
        jupyter: JupyterExecutor,
        work_dir: Path,
        submission_name: str,
        max_steps: int,
    ) -> str:
        llm = ToolCallingLLM(self.coding_cfg)
        llm.set_system(_coding_system(plan, competition, work_dir, submission_name))
        llm.append_user(
            "Start the round. Use tools to implement the plan. "
            f"You have {max_steps} steps available."
        )
        tools = build_tools(submission_name)

        for step in range(max_steps):
            if step > 0:
                remaining = max_steps - step
                status = f"[Status] Step {step + 1}/{max_steps} ({remaining} remaining)."
                if remaining <= 2:
                    status += (
                        f" You are running low on steps. "
                        f"Prioritize producing a valid {submission_name} now."
                    )
                llm.append_user(status)

            step_result = llm.complete_with_tools(tools)
            tool_calls = step_result.tool_calls
            if not tool_calls:
                logger.info("%s Step %d/%d — no tool call, prompting retry", self._tag, step + 1, max_steps)
                llm.append_assistant(step_result.content, None)
                llm.append_user(
                    "You must call one of the tools "
                    "(execute_cell, edit_cell, check_submission, read_file)."
                )
                continue

            assistant_msg = {
                "role": "assistant",
                "content": step_result.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in tool_calls
                ],
            }
            llm.messages.append(assistant_msg)

            for tc in tool_calls:
                goal = tc.arguments.get("goal", "")[:80] if isinstance(tc.arguments, dict) else ""
                logger.info("%s Step %d/%d — %s: %s", self._tag, step + 1, max_steps, tc.name, goal)
                out = self._dispatch_tool(tc.name, tc.arguments, jupyter, work_dir, submission_name)
                result_data = {}
                try:
                    result_data = json.loads(out) if out.startswith("{") else {}
                except Exception:
                    pass
                ok = result_data.get("success", "?")
                et = result_data.get("execution_time")
                et_str = f" ({et:.1f}s)" if isinstance(et, (int, float)) else ""
                logger.info("%s         → success=%s%s", self._tag, ok, et_str)
                if self.tracer:
                    self.tracer.write(
                        f"coding_tool_{tc.name}_s{step}",
                        json.dumps({"args": tc.arguments}, indent=2),
                        out[:8000],
                    )
                llm.append_tool_result(tc.id, out)

        logger.info("%s All %d steps done, generating summary...", self._tag, max_steps)
        llm.append_user(
            "The coding round is over. Provide a concise summary: "
            "what was tried, what worked, what errors occurred, current metrics, "
            "and suggestions for the next round."
        )
        summary = llm.chat_no_tools()
        if self.tracer:
            self.tracer.write("coding_summary", "", summary)
        return summary or "Round ended; no summary generated."

    def _dispatch_tool(
        self,
        name: str,
        args: dict[str, Any],
        jupyter: JupyterExecutor,
        work_dir: Path,
        submission_name: str,
    ) -> str:
        if name == "execute_cell":
            code = args.get("code", "")
            goal = args.get("goal", "")
            r: ExecutionResult = jupyter.execute_cell(code, goal)
            return json.dumps(
                {
                    "success": r.success,
                    "timeout": r.timeout,
                    "output": r.output[:20000],
                    "error": (r.error or "")[:8000],
                    "execution_time": r.execution_time,
                },
                ensure_ascii=False,
            )
        if name == "edit_cell":
            cell_index = int(args.get("cell_index", -1))
            search = args.get("search", "")
            replace = args.get("replace", "")
            goal = args.get("goal", "")
            if cell_index < 0 or cell_index >= len(jupyter.nb.cells):
                return json.dumps(
                    {"success": False, "error": f"Invalid cell_index {cell_index}. "
                     f"Valid range: 0-{len(jupyter.nb.cells) - 1}."}
                )
            old_source = jupyter.nb.cells[cell_index].source
            if search not in old_source:
                return json.dumps(
                    {"success": False, "error": "Search string not found in the specified cell."}
                )
            new_source = old_source.replace(search, replace, 1)
            r = jupyter.execute_cell(new_source, goal)
            return json.dumps(
                {
                    "success": r.success,
                    "timeout": r.timeout,
                    "output": r.output[:20000],
                    "error": (r.error or "")[:8000],
                    "execution_time": r.execution_time,
                    "edited_cell": cell_index,
                },
                ensure_ascii=False,
            )
        if name == "check_submission":
            p = work_dir / submission_name
            if not p.exists():
                return json.dumps({"exists": False})
            try:
                import pandas as pd

                df = pd.read_csv(p, nrows=5)
                return json.dumps(
                    {"exists": True, "shape_head": f"columns={list(df.columns)}, head=\n{df.to_string()}"},
                    ensure_ascii=False,
                )
            except Exception as e:
                return json.dumps({"exists": True, "error": str(e)})
        if name == "read_file":
            rel = args.get("path", "")
            max_lines = int(args.get("max_lines", 120))
            path = (work_dir / rel).resolve()
            if not str(path).startswith(str(work_dir.resolve())):
                return "Error: path escapes work_dir"
            if not path.is_file():
                return f"Error: not a file: {rel}"
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            text = "\n".join(lines[:max_lines])
            if len(lines) > max_lines:
                text += f"\n... ({len(lines) - max_lines} more lines)"
            return text
        return f"unknown tool {name}"
