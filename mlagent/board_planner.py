"""Board-aware planning agent using ExperimentBoard."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from mlagent.config import AgentConfig
from mlagent.experiment_board import ExperimentBoard
from mlagent.llm import ToolCallingLLM
from mlagent.planning_agent import PLANNING_TOOLS, _dispatch_planning_tool
from mlagent.utils import PromptTracer, TokenTracker

logger = logging.getLogger(__name__)


def _board_system(competition_description: str, data_preview: str, metric_hint: str) -> str:
    return f"""You are an expert ML competition strategist. You maintain an Experiment Board —
an append-only, branch-grouped record of every experiment and strategic note accumulated
across all rounds. Read it carefully before every plan.

You can annotate the board with tagged lines in your response:
  [BOARD:branch_name] your concise note
Use these to sketch planned directions, mark promising/dead-end branches, or leave
strategic guidance for future rounds. Keep annotations brief (1 line each).

The coding agent works autonomously each round with many steps. It can build multiple
models, do stacking, and calibrate — all within one round. Your job is to set the
round-level goal and overall strategy, not micro-manage individual steps.

## Round 1 — Initial Strategy Board
Round 1 is special: the board is empty and you must build the initial blueprint.
1. Analyze the competition task, data characteristics, and evaluation metric.
2. Sketch 3-5 approach branches you plan to explore across the entire experiment,
   using [BOARD:branch] tags. For example:
     [BOARD:TF-IDF + Linear] baseline with char/word n-grams, fast to iterate
     [BOARD:Tree Ensembles] LightGBM/XGBoost on count features
     [BOARD:Stacking] meta-learner after 3+ diverse OOF sets
   These branches form the initial "board" — later rounds will expand and refine it.
3. Then give the coding agent 2-4 concrete goals for round 1 (typically: explore data,
   build 1-2 diverse baselines, save OOF predictions).

## Later Rounds — Replan Based on Results
- Review the Experiment Board: what worked, what didn't, what's untried.
- If a direction failed or stalled, annotate it and pivot:
    [BOARD:Tree Ensembles] diminishing returns, deprioritize
- If a direction is promising, deepen it or fork a new sub-branch:
    [BOARD:TF-IDF + Linear] best so far, try sublinear TF + bigrams
- Add new branches if new ideas emerge from results.
- Give the coding agent 2-4 concrete goals for the current round.

## Strategy Progression
- Every round should aim for diversity: include BOTH traditional ML AND deep learning
  tasks in the same plan when possible. Don't wait for "later rounds" to try DL.
- From round 1: if a GPU is available and the task involves text/images, include a
  pretrained model fine-tune alongside traditional baselines.
- Each model should save OOF predictions to ./artifacts/. Once 3+ diverse OOF sets
  exist, include stacking in the plan alongside new models.
- Apply calibration only if the metric is probability-based (log-loss, Brier);
  skip it for ranking metrics (AUC).
- Each round MUST produce a valid submission.
- When budget is low, refine best known approach rather than exploring.

## Anti-Stagnation Rules (CRITICAL)
- If the graded score has not improved for 3+ rounds, you MUST NOT plan another
  round of weight-tuning, micro-blending, or calibration nudges. These are a trap.
  Instead, you MUST plan a genuinely different model family that has NOT been tried:
  e.g. fine-tune a transformer (DistilBERT/BERT), add a tree-based model (LightGBM),
  try a different feature representation (SVD, embeddings), etc.
- When the summary shows a STAGNATION or DEGRADATION warning, treat it as a hard
  constraint: the plan MUST include at least one new model or feature family.
  Do NOT plan "continue refining the current blend" or "micro-search around anchor".
- If a branch was previously marked "dead" or "blocked" due to missing packages,
  re-evaluate — the environment may have changed. Revisit blocked branches before
  giving up and micro-tuning.
- Blending/stacking is only worth planning AFTER a new diverse base model has been
  added. Never plan a round that ONLY does blending.

## Guidelines
- Don't repeat experiments already recorded on the board.
- Focus on WHAT to do, not HOW (the coding agent handles implementation).

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
        tracker: Optional[TokenTracker] = None,
    ) -> None:
        self.cfg = cfg
        self.competition = competition
        self.board = board
        self.tracer = tracer
        self.max_rounds = cfg.max_rounds
        self.max_steps = cfg.max_steps_per_round

        llm_cfg = cfg.planning_llm
        self.llm = ToolCallingLLM(llm_cfg, tracker=tracker)

        desc = getattr(competition, "description", "") or ""
        preview = competition.get_data_preview() if hasattr(competition, "get_data_preview") else ""
        lower = getattr(competition, "is_lower_better", False)
        metric_hint = f"lower is better: {lower}" if lower else "higher is better"
        self.board.is_lower_better = bool(lower)

        from mlagent.coding_agent import _data_profile_for_dir
        data_dir = getattr(competition, "data_dir", None)
        if data_dir:
            data_profile = _data_profile_for_dir(Path(data_dir))
            if data_profile:
                preview = preview + "\n\n" + data_profile

        sys_prompt = _board_system(desc, preview, metric_hint)
        self.llm.messages = [{"role": "system", "content": sys_prompt}]

        self._notebook_cells: list = []
        self._work_dir: Path = Path(".")

    def set_notebook_context(self, notebook_cells: list, work_dir: Path) -> None:
        self._notebook_cells = notebook_cells
        self._work_dir = work_dir

    def _run_tool_loop(self, max_tool_calls: int = 5) -> str:
        for _ in range(max_tool_calls):
            result = self.llm.complete_with_tools(PLANNING_TOOLS, tool_choice="auto")
            if not result.tool_calls:
                if result.content:
                    self.llm.messages.append({"role": "assistant", "content": result.content})
                return result.content or ""

            assistant_msg = {
                "role": "assistant",
                "content": result.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in result.tool_calls
                ],
            }
            self.llm.messages.append(assistant_msg)
            for tc in result.tool_calls:
                logger.info("  Board planner tool: %s(%s)", tc.name, str(tc.arguments)[:80])
                out = _dispatch_planning_tool(
                    tc.name, tc.arguments, self._notebook_cells, self._work_dir,
                )
                self.llm.append_tool_result(tc.id, out[:5000])

        self.llm.append_user("Please provide your response now.")
        return self.llm.chat_no_tools()

    def plan(self, round_num: int, last_summary: Optional[str]) -> str:
        budget = (
            f"[Budget] Round {round_num}/{self.max_rounds} "
            f"({self.max_rounds - round_num} remaining), "
            f"{self.max_steps} coding steps."
        )
        board_str = self.board.to_string()

        if round_num == 1:
            prompt = (
                f"{budget}\n\n"
                f"## Experiment Board\n{board_str}\n\n"
                f"This is round 1 — the board is empty.\n\n"
                f"Before planning, use your tools to inspect the data:\n"
                f"- Use read_file to check what data files exist and their structure.\n"
                f"- If there are multiple versions of the same data (e.g. CSV and JSON), "
                f"compare them and recommend which one the coding agent should use.\n"
                f"- Note any train/test column mismatches.\n\n"
                f"Then build the initial strategy board: analyze the task and sketch "
                f"3-5 approach branches using [BOARD:branch] tags.\n\n"
                f"Include a ## Data Notes section specifying which files to use."
            )
        else:
            prompt = (
                f"{budget}\n\n"
                f"## Experiment Board\n{board_str}\n\n"
                f"Summary from round {round_num - 1}:\n{last_summary or '(none)'}\n\n"
                f"Use your tools (read_notebook_cell, list_artifacts, read_file) to "
                f"inspect actual errors and scores if needed.\n\n"
                f"Review the board. Update branch annotations if needed — mark dead ends, "
                f"highlight promising directions, add new branches if new ideas emerge.\n\n"
                f"Then provide the plan for round {round_num}."
            )

        self.llm.append_user(prompt)
        reply = self._run_tool_loop(max_tool_calls=5)

        self.board.update_from_planner(reply, round_num)

        if self.tracer:
            self.tracer.write(f"board_plan_r{round_num}", prompt, reply)
        return reply
