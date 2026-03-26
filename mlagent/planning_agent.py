"""Planning agent: long-term memory across rounds."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from mlagent.config import AgentConfig
from mlagent.llm import PlanningLLM
from mlagent.utils import PromptTracer, TokenTracker

logger = logging.getLogger(__name__)


def _planning_system(competition_description: str, data_preview: str, metric_hint: str) -> str:
    return f"""You are an expert ML competition strategist.

The coding agent works autonomously each round: it has many steps and can build multiple
models, do stacking, and calibration all within a single round. Your job is to tell it
WHAT to achieve, and HOW. The agent handles debugging and error recovery
on its own.

## How to Plan Each Round
- Each round, give the coding agent a clear goal with 2-4 concrete tasks.
- The agent can accomplish multiple things per round (e.g. build 2 models + stack them).
- Focus on what models/features to try, not on debugging or error handling.
- If last round failed, suggest a simpler approach — but trust the agent to handle bugs.

## Recommended Progression Across Rounds
- Early rounds: build diverse base models. For each model, tell the agent to save
  K-fold OOF (out-of-fold) train predictions and test predictions to ./artifacts/ as .npy.
  A single round can include multiple diverse models if there are enough steps.
- Later rounds: once 3+ diverse OOF sets exist in ./artifacts/, tell the agent to build
  a stacking meta-learner (train on stacked OOF features) and calibrate probabilities.
  Stacking + calibration can be done in the same round as building new base models.
- Final rounds: refine the best approach. Small tuning, not major new experiments.

## Model Combination (Stacking)
- Diverse weak models stacked together outperform a single strong model tuned heavily.
- OOF protocol: K-fold cross-validation → save oof_train.npy and test_preds.npy per model.
- Stacking: load OOF artifacts, stack as columns, train a meta-learner.
  Prefer a non-linear meta-learner (XGBoost, LightGBM) when you have 4+ diverse OOF
  sources — it captures interactions between base models that LR misses.
  LR is acceptable for 2-3 OOF sources or as a quick sanity check.
- Calibration: after stacking, apply CalibratedClassifierCV (isotonic) to the
  meta-model ONLY when the metric is probability-based (log-loss, Brier). For ranking
  metrics (AUC, accuracy, F1), calibration does NOT change rankings and won't help.
  Always tell the agent to verify calibration improves CV score before submitting.
- Select diverse OOF sources (e.g. linear, NB, tree, neural) — avoid near-duplicates.

## Key Principles
- Each round MUST produce a valid submission file.
- When budget is low, refine best known approach rather than exploring.
- If a round timed out, suggest simpler models/fewer features next round.
- Be specific about models and features, but let the agent decide execution order.

Task description:
{competition_description}

Data preview:
{data_preview}

Metric: {metric_hint}

Each turn you receive results from coding agents (Jupyter notebook experiments).
When multiple coding agents are available, produce diverse plans for each — e.g., one
tries a baseline while another tries a different model family. Diversity across agents
maximises the chance of finding a winning approach in each round.
Propose a clear, actionable plan for each coding agent. Keep each plan under ~800 words."""


class PlanningAgent:
    def __init__(
        self,
        cfg: AgentConfig,
        competition: Any,
        tracer: Optional[PromptTracer] = None,
        tracker: Optional[TokenTracker] = None,
    ) -> None:
        self.cfg = cfg
        self.competition = competition
        self.tracer = tracer
        self.max_rounds = cfg.max_rounds
        self.max_steps_per_round = cfg.max_steps_per_round

        llm_cfg = cfg.planning_llm
        if not llm_cfg.keep_history:
            logger.warning("planning_llm.keep_history should be true for long-term memory")
        self.llm = PlanningLLM(llm_cfg, tracker=tracker)

        desc = getattr(competition, "description", "") or ""
        preview = competition.get_data_preview() if hasattr(competition, "get_data_preview") else ""
        lower = getattr(competition, "is_lower_better", False)
        metric_hint = f"lower is better: {lower}" if lower else "higher is better"

        sys_prompt = _planning_system(desc, preview, metric_hint)
        self.llm.messages = [{"role": "system", "content": sys_prompt}]

    def plan(self, round_num: int, last_summary: Optional[str]) -> str:
        budget = (
            f"[Budget] Round {round_num}/{self.max_rounds} "
            f"({self.max_rounds - round_num} remaining), "
            f"{self.max_steps_per_round} coding steps per round."
        )

        if round_num == 1:
            self.llm.append_user(
                f"{budget}\n\n"
                f"Round 1: no prior coding results.\n\n"
                f"First, create an overall strategy for all {self.max_rounds} rounds — "
                f"a roadmap of what to try and in what order (baseline, improvements, advanced).\n\n"
                f"Then, provide the detailed plan for Round 1.\n\n"
                f"Format:\n"
                f"## Overall Strategy\n(roadmap for {self.max_rounds} rounds)\n\n"
                f"## Round 1 Plan\n(specific actions for the coding agent this round)"
            )
        else:
            self.llm.append_user(
                f"{budget}\n\n"
                f"Results from round {round_num - 1}:\n{last_summary or '(none)'}\n\n"
                f"Review your overall strategy and adjust if needed based on these results. "
                f"Then provide the plan for round {round_num}."
            )
        reply = self.llm.chat()
        if self.tracer:
            self.tracer.write(f"planning_r{round_num}", str(self.llm.messages[-2:]), reply)
        return reply

    def plan_parallel(
        self,
        round_num: int,
        last_summary: Optional[str],
        num_agents: int,
    ) -> list[str]:
        """Generate plans for N coding agents.

        When num_agents == 1 this delegates to plan() directly.
        When num_agents > 1 it asks the planner for N diverse sub-plans
        using ``## Agent 0 Plan`` / ``## Agent 1 Plan`` headers.
        """
        if num_agents <= 1:
            return [self.plan(round_num, last_summary)]

        budget = (
            f"[Budget] Round {round_num}/{self.max_rounds} "
            f"({self.max_rounds - round_num} remaining), "
            f"{self.max_steps_per_round} coding steps per round, "
            f"{num_agents} parallel coding agents."
        )

        if round_num == 1:
            self.llm.append_user(
                f"{budget}\n\n"
                f"Round 1: no prior coding results.\n\n"
                f"You have {num_agents} coding agents running in parallel this round. "
                f"Produce {num_agents} **diverse** plans — each agent should try a clearly "
                f"different approach so we explore the solution space efficiently.\n\n"
                f"First, create an overall strategy for all {self.max_rounds} rounds — "
                f"a roadmap of what to try and in what order.\n\n"
                f"Then provide plans using these exact headers:\n"
                + "\n".join(
                    f"## Agent {i} Plan\n(specific actions for agent {i})"
                    for i in range(num_agents)
                )
            )
        else:
            self.llm.append_user(
                f"{budget}\n\n"
                f"Results from round {round_num - 1}:\n{last_summary or '(none)'}\n\n"
                f"You have {num_agents} coding agents running in parallel this round. "
                f"Review your overall strategy and adjust if needed based on these results. "
                f"Then provide {num_agents} diverse plans using these exact headers:\n"
                + "\n".join(
                    f"## Agent {i} Plan\n(specific actions for agent {i})"
                    for i in range(num_agents)
                )
            )

        reply = self.llm.chat()
        if self.tracer:
            self.tracer.write(f"planning_r{round_num}", str(self.llm.messages[-2:]), reply)

        plans = self._parse_agent_plans(reply, num_agents)
        return plans

    @staticmethod
    def _parse_agent_plans(text: str, num_agents: int) -> list[str]:
        """Split LLM reply by ``## Agent N Plan`` headers."""
        pattern = r"##\s+Agent\s+(\d+)\s+Plan"
        splits = list(re.finditer(pattern, text, re.IGNORECASE))
        if len(splits) < num_agents:
            logger.warning(
                "Could not parse %d agent plans (found %d headers), "
                "replicating full plan for all agents",
                num_agents,
                len(splits),
            )
            return [text] * num_agents

        plans: list[str] = []
        for idx, match in enumerate(splits):
            start = match.end()
            end = splits[idx + 1].start() if idx + 1 < len(splits) else len(text)
            plans.append(text[start:end].strip())
        return plans[:num_agents]
