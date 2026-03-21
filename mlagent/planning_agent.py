"""Planning agent: long-term memory across rounds."""

from __future__ import annotations

import logging
from typing import Any, Optional

from mlagent.config import AgentConfig
from mlagent.llm import PlanningLLM
from mlagent.utils import PromptTracer

logger = logging.getLogger(__name__)


def _planning_system(competition_description: str, data_preview: str, metric_hint: str) -> str:
    return f"""You are the planning agent for an MLE-bench competition.

Task description:
{competition_description}

Data preview:
{data_preview}

Metric: {metric_hint}

Each turn you receive results from a coding agent (Jupyter notebook experiments). Propose a clear, actionable plan for the NEXT coding round: what to implement, try, or debug. Be specific (models, validation, features). Keep the plan under ~800 words."""


class PlanningAgent:
    def __init__(
        self,
        cfg: AgentConfig,
        competition: Any,
        tracer: Optional[PromptTracer] = None,
    ) -> None:
        self.cfg = cfg
        self.competition = competition
        self.tracer = tracer
        llm_cfg = cfg.planning_llm
        if not llm_cfg.keep_history:
            logger.warning("planning_llm.keep_history should be true for long-term memory")
        self.llm = PlanningLLM(llm_cfg)

        desc = getattr(competition, "description", "") or ""
        preview = competition.get_data_preview() if hasattr(competition, "get_data_preview") else ""
        lower = getattr(competition, "is_lower_better", False)
        metric_hint = f"lower is better: {lower}" if lower else "higher is better"

        sys_prompt = _planning_system(desc, preview, metric_hint)
        self.llm.messages = [{"role": "system", "content": sys_prompt}]

    def plan(self, round_num: int, last_summary: Optional[str]) -> str:
        if round_num == 1:
            self.llm.append_user(
                "Round 1: no prior coding results. Propose the first plan for the coding agent."
            )
        else:
            self.llm.append_user(
                f"Results from previous round(s):\n{last_summary or '(none)'}\n\n"
                f"Propose the plan for round {round_num}."
            )
        reply = self.llm.chat()
        if self.tracer:
            self.tracer.write(f"planning_r{round_num}", str(self.llm.messages[-2:]), reply)
        return reply
