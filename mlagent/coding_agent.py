"""Coding agent: tool-calling loop with JupyterExecutor."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from mlagent.config import AgentConfig, LLMConfig
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
        {
            "type": "function",
            "function": {
                "name": "finish_round",
                "description": "End this coding round and return a short summary for the planning agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "What was tried, metrics printed, issues, next hints",
                        },
                    },
                    "required": ["summary"],
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

You MUST use tools. Typical flow: execute_cell for experiments, check_submission when appropriate, finish_round when done.
Rules:
- Use relative paths; data under input/
- One execute_cell should be focused; you can call it many times
- Call finish_round with a concise summary when the round objective is met or you cannot proceed"""


class CodingAgent:
    def __init__(self, coding_cfg: LLMConfig, tracer: Optional[PromptTracer] = None) -> None:
        self.coding_cfg = coding_cfg
        self.tracer = tracer

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
            "Start the round: call execute_cell or other tools. End with finish_round."
        )
        tools = build_tools(submission_name)
        summary = ""
        for step in range(max_steps):
            step_result = llm.complete_with_tools(tools)
            raw = step_result.raw_message
            tool_calls = step_result.tool_calls
            if not tool_calls:
                # nudge model to use tools
                llm.append_assistant(step_result.content, None)
                llm.append_user("You must call one of the tools (execute_cell, check_submission, read_file, finish_round).")
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
                out = self._dispatch_tool(tc.name, tc.arguments, jupyter, work_dir, submission_name)
                if self.tracer:
                    self.tracer.write(
                        f"coding_tool_{tc.name}_s{step}",
                        json.dumps({"args": tc.arguments}, indent=2),
                        out[:8000],
                    )
                llm.append_tool_result(tc.id, out)
                if tc.name == "finish_round":
                    summary = tc.arguments.get("summary", "") or out
                    return summary

        return summary or "Round ended without finish_round; step limit reached."

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
                    "llm_terminated": r.llm_terminated,
                    "timeout": r.timeout,
                    "output": r.output[:20000],
                    "error": (r.error or "")[:8000],
                    "execution_time": r.execution_time,
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
        if name == "finish_round":
            return json.dumps({"ok": True, "summary": args.get("summary", "")}, ensure_ascii=False)
        return f"unknown tool {name}"
