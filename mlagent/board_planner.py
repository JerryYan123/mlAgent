"""Board-aware planning agent with inner-loop replanning."""

from __future__ import annotations

import logging
from typing import Any, Optional

from mlagent.config import AgentConfig
from mlagent.experiment_board import ExperimentBoard
from mlagent.llm import PlanningLLM
from mlagent.utils import PromptTracer

logger = logging.getLogger(__name__)


def _board_system(competition_description: str, data_preview: str, metric_hint: str) -> str:
    return f"""You are an expert ML competition strategist with access to an Experiment Board.

The board tracks everything discovered so far: data insights, experiments run, scores,
hypotheses, and strategic decisions. You MUST read the board before every plan and use
it to avoid repeating work and to build on what works.

You can add entries to the board by including tagged lines in your response:
  [BOARD:insight] observation about data
  [BOARD:hypothesis] idea to test and why
  [BOARD:decision] strategic choice you're making
  [BOARD:finding] conclusion drawn from results

Guidelines:
- Round 1: populate the board with initial hypotheses based on the competition/data.
- Always check what experiments have already been run before suggesting new ones.
- Exploit what works: if a model scored well, suggest tuning it rather than starting over.
- Explore strategically: if scores are low, try a fundamentally different approach.
- Each plan MUST lead to a valid submission file.
- When budget is low, focus on the best known approach.
- You will be called multiple times within a round (inner-loop replanning).
  When you see mid-round results, give focused adjustments, not full rewrites.

Task description:
{competition_description}

Data preview:
{data_preview}

Metric: {metric_hint}"""


class BoardPlanningAgent:
    def __init__(
        self,
        cfg: AgentConfig,
        competition: Any,
        board: ExperimentBoard,
        tracer: Optional[PromptTracer] = None,
    ) -> None:
        self.cfg = cfg
        self.competition = competition
        self.board = board
        self.tracer = tracer
        self.max_rounds = cfg.max_rounds
        self.max_steps = cfg.max_steps_per_round
        self.replan_interval = cfg.replan_interval

        llm_cfg = cfg.planning_llm
        self.llm = PlanningLLM(llm_cfg)

        desc = getattr(competition, "description", "") or ""
        preview = competition.get_data_preview() if hasattr(competition, "get_data_preview") else ""
        lower = getattr(competition, "is_lower_better", False)
        metric_hint = f"lower is better: {lower}" if lower else "higher is better"

        sys_prompt = _board_system(desc, preview, metric_hint)
        self.llm.messages = [{"role": "system", "content": sys_prompt}]

    def plan(self, round_num: int, last_summary: Optional[str]) -> str:
        """Generate a plan for the start of a round."""
        budget = (
            f"[Budget] Round {round_num}/{self.max_rounds} "
            f"({self.max_rounds - round_num} remaining), "
            f"{self.max_steps} coding steps, replanning every {self.replan_interval} steps."
        )
        board_str = self.board.to_string()

        if round_num == 1:
            prompt = (
                f"{budget}\n\n"
                f"## Current Board\n{board_str}\n\n"
                f"Round 1: no prior results.\n\n"
                f"1. Populate the board with initial insights and hypotheses about this "
                f"competition (use [BOARD:insight] and [BOARD:hypothesis] tags).\n"
                f"2. Provide a concrete plan for the coding agent.\n\n"
                f"The coder will pause every {self.replan_interval} steps for your feedback."
            )
        else:
            prompt = (
                f"{budget}\n\n"
                f"## Current Board\n{board_str}\n\n"
                f"Results from round {round_num - 1}:\n{last_summary or '(none)'}\n\n"
                f"Review the board, add any new insights or decisions, "
                f"then provide the plan for round {round_num}."
            )

        self.llm.append_user(prompt)
        reply = self.llm.chat()

        self.board.update_from_llm(reply, round_num)

        if self.tracer:
            self.tracer.write(f"board_plan_r{round_num}", prompt, reply)
        return reply

    def replan(self, round_num: int, step: int, mid_summary: str) -> str:
        """Mid-round replanning: coder paused, planner adjusts."""
        board_str = self.board.to_string()
        remaining = self.max_steps - step
        prompt = (
            f"[Mid-round update] Round {round_num}, step {step}/{self.max_steps} "
            f"({remaining} steps remaining).\n\n"
            f"## Current Board\n{board_str}\n\n"
            f"## Coding progress so far\n{mid_summary}\n\n"
            f"Update the board with any new findings (use [BOARD:...] tags). "
            f"Then give focused next steps for the remaining {remaining} steps. "
            f"Keep it concise — this is a mid-round adjustment, not a full replan."
        )

        self.llm.append_user(prompt)
        reply = self.llm.chat()

        self.board.update_from_llm(reply, round_num, step)

        if self.tracer:
            self.tracer.write(f"board_replan_r{round_num}_s{step}", prompt, reply)
        return reply
