"""Codex-style todo list planner: generates and dynamically revises a task checklist."""

from __future__ import annotations

import logging
from typing import Any, Optional

from mlagent.config import AgentConfig
from mlagent.experiment_board import TodoList
from mlagent.llm import PlanningLLM
from mlagent.utils import PromptTracer, TokenTracker

logger = logging.getLogger(__name__)


def _todo_system(competition_description: str, data_preview: str, metric_hint: str) -> str:
    return f"""You are an ML competition strategist that works by managing a Todo List.

The coding agent works autonomously with many steps per round. It can build multiple
models, do stacking, and calibrate — all within one round. Your todo list tells it
WHAT to accomplish. The agent handles debugging on its own.

Your todo list format (one item per line):
  - [ ] Task description (why: hypothesis for why this helps)

## Suggested Task Types
- Build a baseline model + valid submission (save OOF predictions to ./artifacts/).
- Build a different model family (save OOF to ./artifacts/ for future stacking).
- Stack existing OOF artifacts into a meta-learner + calibrate probabilities.
- Multiple model + stacking tasks can appear in the same round.
- Diverse weak models stacked > one strong model tuned heavily.

## Guidelines
- Start with 5-8 focused tasks per round.
- Order by priority: baseline + valid submission FIRST, then improvements.
- After seeing results, revise: mark done, add new, remove irrelevant, reorder.
- Always ensure at least one task produces a valid submission file.
- When budget is low, trim to essentials.

Task description:
{competition_description}

Data preview:
{data_preview}

Metric: {metric_hint}"""


class TodoPlanningAgent:
    def __init__(
        self,
        cfg: AgentConfig,
        competition: Any,
        todo: TodoList,
        tracer: Optional[PromptTracer] = None,
        tracker: Optional[TokenTracker] = None,
    ) -> None:
        self.cfg = cfg
        self.competition = competition
        self.todo = todo
        self.tracer = tracer
        self.max_rounds = cfg.max_rounds
        self.max_steps = cfg.max_steps_per_round
        self.replan_interval = cfg.replan_interval

        llm_cfg = cfg.planning_llm
        self.llm = PlanningLLM(llm_cfg, tracker=tracker)

        desc = getattr(competition, "description", "") or ""
        preview = competition.get_data_preview() if hasattr(competition, "get_data_preview") else ""
        lower = getattr(competition, "is_lower_better", False)
        metric_hint = f"lower is better: {lower}" if lower else "higher is better"

        sys_prompt = _todo_system(desc, preview, metric_hint)
        self.llm.messages = [{"role": "system", "content": sys_prompt}]

    def plan(self, round_num: int, last_summary: Optional[str]) -> str:
        """Generate or refresh the todo list for a round."""
        budget = (
            f"[Budget] Round {round_num}/{self.max_rounds} "
            f"({self.max_rounds - round_num} remaining), "
            f"{self.max_steps} coding steps this round."
        )
        todo_str = self.todo.to_string()

        if round_num == 1:
            prompt = (
                f"{budget}\n\n"
                f"Round 1: no prior results. Create the initial todo list.\n\n"
                f"Output ONLY the todo list in the format:\n"
                f"- [ ] Task description (why: hypothesis)\n\n"
                f"Start with data exploration and a fast baseline, then improvements."
            )
        else:
            prompt = (
                f"{budget}\n\n"
                f"## Current Todo List\n{todo_str}\n\n"
                f"Results from round {round_num - 1}:\n{last_summary or '(none)'}\n\n"
                f"Revise the todo list based on results. Keep completed items as-is. "
                f"Add new tasks, remove irrelevant ones, reorder by priority.\n"
                f"Output the full revised todo list."
            )

        self.llm.append_user(prompt)
        reply = self.llm.chat()

        self.todo.replace_from_llm(reply)

        if self.tracer:
            self.tracer.write(f"todo_plan_r{round_num}", prompt, reply)

        return self.todo.to_string()

    def revise(self, round_num: int, step: int, mid_summary: str) -> str:
        """Mid-round revision: update todos based on coding progress."""
        todo_str = self.todo.to_string()
        remaining = self.max_steps - step
        prompt = (
            f"[Mid-round checkpoint] Round {round_num}, step {step}/{self.max_steps} "
            f"({remaining} steps remaining).\n\n"
            f"## Current Todo List\n{todo_str}\n\n"
            f"## Coding progress\n{mid_summary}\n\n"
            f"Revise the todo list: mark completed items, add new ones based on "
            f"findings, remove or deprioritize if needed. "
            f"Output the full revised todo list."
        )

        self.llm.append_user(prompt)
        reply = self.llm.chat()

        self.todo.replace_from_llm(reply)

        if self.tracer:
            self.tracer.write(f"todo_revise_r{round_num}_s{step}", prompt, reply)

        return self.todo.to_string()
