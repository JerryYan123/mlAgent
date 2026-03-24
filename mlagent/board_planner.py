"""Board-aware planning agent using ExperimentMap."""

from __future__ import annotations

import logging
from typing import Any, Optional

from mlagent.config import AgentConfig
from mlagent.experiment_board import ExperimentMap
from mlagent.llm import PlanningLLM
from mlagent.utils import PromptTracer, TokenTracker

logger = logging.getLogger(__name__)


def _board_system(competition_description: str, data_preview: str, metric_hint: str) -> str:
    return f"""You are an expert ML competition strategist. You maintain an Experiment Map —
an append-only, branch-grouped record of every experiment and strategic note accumulated
across all rounds. Read it carefully before every plan.

You can annotate the map with tagged lines in your response:
  [MAP:branch_name] your concise note
Use these to sketch planned directions, mark promising/dead-end branches, or leave
strategic guidance for future rounds. Keep annotations brief (1 line each).

The coding agent works autonomously each round with many steps. It can build multiple
models, do stacking, and calibrate — all within one round. Your job is to set the
round-level goal and overall strategy, not micro-manage individual steps.

## Round 1 — Initial Strategy Map
Round 1 is special: the map is empty and you must build the initial blueprint.
1. Analyze the competition task, data characteristics, and evaluation metric.
2. Sketch 3-5 approach branches you plan to explore across the entire experiment,
   using [MAP:branch] tags. For example:
     [MAP:TF-IDF + Linear] baseline with char/word n-grams, fast to iterate
     [MAP:Tree Ensembles] LightGBM/XGBoost on count features
     [MAP:Stacking] meta-learner after 3+ diverse OOF sets
   These branches form the initial "map" — later rounds will expand and refine it.
3. Then give the coding agent 2-4 concrete goals for round 1 (typically: explore data,
   build 1-2 diverse baselines, save OOF predictions).

## Later Rounds — Replan Based on Results
- Review the Experiment Map: what worked, what didn't, what's untried.
- If a direction failed or stalled, annotate it and pivot:
    [MAP:Tree Ensembles] diminishing returns, deprioritize
- If a direction is promising, deepen it or fork a new sub-branch:
    [MAP:TF-IDF + Linear] best so far, try sublinear TF + bigrams
- Add new branches if new ideas emerge from results.
- Give the coding agent 2-4 concrete goals for the current round.

## Strategy Progression
- Early rounds: diverse base models, each saving OOF predictions to ./artifacts/.
- Mid rounds: tune top performers, try different feature engineering.
- Late rounds: stacking (meta-learner on OOF features) + probability calibration.
- Each round MUST produce a valid submission.
- When budget is low, refine best known approach rather than exploring.

## Guidelines
- Don't repeat experiments already recorded in the map.
- Focus on WHAT to do, not HOW (the coding agent handles implementation).
- If scores are stuck for 2+ rounds, pivot to a fundamentally different approach.

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
        exp_map: ExperimentMap,
        tracer: Optional[PromptTracer] = None,
        tracker: Optional[TokenTracker] = None,
    ) -> None:
        self.cfg = cfg
        self.competition = competition
        self.exp_map = exp_map
        self.tracer = tracer
        self.max_rounds = cfg.max_rounds
        self.max_steps = cfg.max_steps_per_round

        llm_cfg = cfg.planning_llm
        self.llm = PlanningLLM(llm_cfg, tracker=tracker)

        desc = getattr(competition, "description", "") or ""
        preview = competition.get_data_preview() if hasattr(competition, "get_data_preview") else ""
        lower = getattr(competition, "is_lower_better", False)
        metric_hint = f"lower is better: {lower}" if lower else "higher is better"
        self.exp_map.is_lower_better = bool(lower)

        sys_prompt = _board_system(desc, preview, metric_hint)
        self.llm.messages = [{"role": "system", "content": sys_prompt}]

    def plan(self, round_num: int, last_summary: Optional[str]) -> str:
        budget = (
            f"[Budget] Round {round_num}/{self.max_rounds} "
            f"({self.max_rounds - round_num} remaining), "
            f"{self.max_steps} coding steps."
        )
        map_str = self.exp_map.to_string()

        if round_num == 1:
            prompt = (
                f"{budget}\n\n"
                f"## Experiment Map\n{map_str}\n\n"
                f"This is round 1 — the map is empty.\n\n"
                f"First, build the initial strategy map: analyze the task and sketch "
                f"3-5 approach branches using [MAP:branch] tags. These branches form "
                f"the blueprint for the entire experiment.\n\n"
                f"Then, provide a concrete plan for the coding agent for this round "
                f"(typically: explore data, build 1-2 diverse baselines, save OOF)."
            )
        else:
            prompt = (
                f"{budget}\n\n"
                f"## Experiment Map\n{map_str}\n\n"
                f"Summary from round {round_num - 1}:\n{last_summary or '(none)'}\n\n"
                f"Review the map. Update branch annotations if needed — mark dead ends, "
                f"highlight promising directions, add new branches if new ideas emerge.\n\n"
                f"Then provide the plan for round {round_num}."
            )

        self.llm.append_user(prompt)
        reply = self.llm.chat()

        self.exp_map.update_from_planner(reply, round_num)

        if self.tracer:
            self.tracer.write(f"board_plan_r{round_num}", prompt, reply)
        return reply
