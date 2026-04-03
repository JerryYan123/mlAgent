"""Planning agent: long-term memory across rounds, with diagnostic tools."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from mlagent.config import AgentConfig
from mlagent.llm import ToolCallingLLM
from mlagent.utils import PromptTracer, TokenTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Planning tools — read-only, let planner explore notebook & artifacts
# ---------------------------------------------------------------------------

PLANNING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_notebook_cell",
            "description": (
                "Read a specific cell from the coding agent's Jupyter notebook. "
                "Returns the cell's source code and its output (stdout, errors). "
                "Use this to inspect what code ran and what errors occurred."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_index": {
                        "type": "integer",
                        "description": "0-based index. Use negative indices to read from the end (e.g. -1 = last cell).",
                    },
                },
                "required": ["cell_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_artifacts",
            "description": (
                "List all saved OOF/test prediction artifacts (.npy files) in the workspace. "
                "Returns file names and sizes."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the workspace (e.g. a log or CSV header).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path under the agent workspace."},
                    "max_lines": {"type": "integer", "default": 50},
                },
                "required": ["path"],
            },
        },
    },
]


def _dispatch_planning_tool(
    name: str,
    args: dict[str, Any],
    notebook_cells: list,
    work_dir: Path,
) -> str:
    """Execute a planning tool and return the result as a string."""
    if name == "read_notebook_cell":
        idx = int(args.get("cell_index", -1))
        if not notebook_cells:
            return "No notebook cells available."
        try:
            cell = notebook_cells[idx]
        except IndexError:
            return f"Invalid cell_index {idx}. Valid range: 0 to {len(notebook_cells)-1} (or negative)."

        source = getattr(cell, "source", None) or cell.get("source", "")
        outputs = getattr(cell, "outputs", None) or cell.get("outputs", [])

        # Extract output text
        out_parts = []
        for output in outputs:
            out_type = getattr(output, "output_type", None) or output.get("output_type", "")
            if out_type == "error":
                tb = getattr(output, "traceback", None) or output.get("traceback", [])
                text = "\n".join(tb) if isinstance(tb, list) else str(tb)
                # Strip ANSI codes for readability
                text = re.sub(r"\x1b\[[0-9;]*m", "", text)
                out_parts.append(f"[ERROR]\n{text}")
            elif out_type in ("stream", "execute_result"):
                raw = getattr(output, "text", None) or output.get("text", "")
                text = "".join(raw) if isinstance(raw, list) else str(raw)
                out_parts.append(text)

        output_text = "\n".join(out_parts)
        # Truncate to avoid blowing context
        if len(source) > 2000:
            source = source[:2000] + "\n... (truncated)"
        if len(output_text) > 3000:
            output_text = output_text[:3000] + "\n... (truncated)"

        return f"Cell [{idx}] ({len(notebook_cells)} total cells):\n--- CODE ---\n{source}\n--- OUTPUT ---\n{output_text}"

    if name == "list_artifacts":
        art_dir = work_dir / "artifacts"
        if not art_dir.exists():
            return "No artifacts directory found."
        files = sorted(art_dir.glob("*.npy"))
        if not files:
            return "Artifacts directory exists but is empty."
        lines = []
        for f in files:
            size_kb = f.stat().st_size / 1024
            lines.append(f"  {f.name} ({size_kb:.0f} KB)")
        return f"{len(files)} artifacts:\n" + "\n".join(lines)

    if name == "read_file":
        rel = args.get("path", "")
        max_lines = int(args.get("max_lines", 50))
        path = (work_dir / rel).resolve()
        if not str(path).startswith(str(work_dir.resolve())):
            return "Error: path escapes workspace."
        if not path.is_file():
            return f"Error: not a file: {rel}"
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        result = "\n".join(lines[:max_lines])
        if len(lines) > max_lines:
            result += f"\n... ({len(lines) - max_lines} more lines)"
        return result

    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _planning_system(competition_description: str, data_preview: str, metric_hint: str) -> str:
    return f"""You are an expert ML competition strategist.

The coding agent works autonomously each round: it has many steps and can build multiple
models, do stacking, and calibration all within a single round. Your job is to tell it
WHAT to achieve, and HOW. The agent handles debugging and error recovery
on its own.

## Your Tools
You have diagnostic tools to inspect the coding agent's work:
- read_notebook_cell(cell_index): read code and output of any cell in the notebook.
  Use negative indices (e.g. -1, -2) to read recent cells.
- list_artifacts(): see all saved .npy prediction files on disk.
- read_file(path): read any file in the workspace.

Use these tools between rounds to understand what actually happened — don't rely solely
on the coding agent's summary, which may be incomplete or misleading. Look at actual
errors, actual scores, and actual artifacts before planning the next round.

## How to Plan Each Round
- Structure your plan as separate tasks using ### Task N headers.
  Each task will be executed independently by a separate coding agent.
  The coding agent for each task ONLY sees that task's instructions.
  Example:
    ### Task 1: TF-IDF + Logistic Regression baseline
    Build a TF-IDF model with word and char n-grams, combine with numeric features...
    ### Task 2: Fine-tune DistilBERT
    Fine-tune distilbert-base-uncased on the text with 3-fold CV...
    ### Task 3: Stack and submit
    Load OOF artifacts from Task 1 and 2, build a meta-learner...
- Each task should be self-contained: the coding agent can access saved artifacts
  from previous tasks (via ./artifacts/*.npy) and the shared Jupyter kernel state.
- After each task, the holdout score is automatically measured.
- Aim for 2-4 tasks per round, each exploring a different approach.
- From round 1: if a GPU is available and the task involves text or images, include
  a pretrained model fine-tune (e.g. distilbert, roberta-base for NLP; resnet for
  vision) alongside traditional baselines. Don't wait for "later rounds".
- For each model, tell the agent to save K-fold OOF train predictions and test
  predictions to ./artifacts/ as .npy.
- Once 3+ diverse OOF sets exist, include stacking in the plan alongside new models.
- Final rounds: refine the best approach. Small tuning, not major new experiments.

## Model Combination (Stacking)
- Stacking diverse models often outperforms any single model, but not always.
  If the coding agent reports that a single model's OOF beats the stacked OOF,
  trust that signal — plan to improve that model rather than force more blending.
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

## Score Monitoring & Rollback
- Pay close attention to the actual graded score each round, not just the agent's
  reported CV metric. The graded score is ground truth.
- If the graded score DROPS compared to a previous round's best, the new approach
  is worse. Do NOT continue down that path — revert to the approach and model
  that achieved the best graded score.
- If the graded score has NOT improved for 3+ rounds, the current strategy is
  saturated. Do NOT keep micro-tuning weights or blending. Instead, try a
  fundamentally different model family or feature set.
- If the agent reports a CV metric that is much higher than the graded score
  (e.g. CV=0.90 but graded=0.65), suspect overfitting or evaluation bugs.
  Tell the agent to use proper K-fold out-of-fold evaluation and to distrust
  any suspiciously large CV jumps.

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


# ---------------------------------------------------------------------------
# Planning Agent
# ---------------------------------------------------------------------------

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
        self.llm = ToolCallingLLM(llm_cfg, tracker=tracker)

        desc = getattr(competition, "description", "") or ""
        preview = competition.get_data_preview() if hasattr(competition, "get_data_preview") else ""
        lower = getattr(competition, "is_lower_better", False)
        metric_hint = f"lower is better: {lower}" if lower else "higher is better"

        # Add data profile with column analysis and train/test mismatch warnings
        from mlagent.coding_agent import _data_profile_for_dir
        data_dir = getattr(competition, "data_dir", None)
        if data_dir:
            data_profile = _data_profile_for_dir(Path(data_dir))
            if data_profile:
                preview = preview + "\n\n" + data_profile

        sys_prompt = _planning_system(desc, preview, metric_hint)
        self.llm.messages = [{"role": "system", "content": sys_prompt}]

        # These get set by the orchestrator each round
        self._notebook_cells: list = []
        self._work_dir: Path = Path(".")

    def set_notebook_context(self, notebook_cells: list, work_dir: Path) -> None:
        """Called by orchestrator to give planner access to the notebook."""
        self._notebook_cells = notebook_cells
        self._work_dir = work_dir

    def _run_tool_loop(self, max_tool_calls: int = 5) -> str:
        """Let planner use tools to investigate, then return its final text response."""
        for _ in range(max_tool_calls):
            result = self.llm.complete_with_tools(PLANNING_TOOLS, tool_choice="auto")

            if not result.tool_calls:
                # No more tools — this is the final text response
                if result.content:
                    self.llm.messages.append({"role": "assistant", "content": result.content})
                return result.content or ""

            # Process tool calls
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
                logger.info("  Planner tool: %s(%s)", tc.name, str(tc.arguments)[:80])
                out = _dispatch_planning_tool(
                    tc.name, tc.arguments, self._notebook_cells, self._work_dir,
                )
                self.llm.append_tool_result(tc.id, out[:5000])

        # Exhausted tool calls, ask for final answer
        self.llm.append_user("Please provide your response now.")
        return self.llm.chat_no_tools()

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
                f"Before planning, use your tools to inspect the data:\n"
                f"- Use read_file to check what data files exist and their structure.\n"
                f"- If there are multiple versions of the same data (e.g. CSV and JSON), "
                f"compare them and recommend which one the coding agent should use.\n"
                f"- Note any train/test column mismatches.\n\n"
                f"Then create an overall strategy and a detailed plan for Round 1.\n\n"
                f"Format:\n"
                f"## Data Notes\n(which files to use, any column issues)\n\n"
                f"## Overall Strategy\n(roadmap for {self.max_rounds} rounds)\n\n"
                f"## Round 1 Plan\n"
                f"Use ### Task N headers for each task. Example:\n"
                f"### Task 1: Build TF-IDF + numeric baseline\n"
                f"### Task 2: Fine-tune transformer\n"
                f"### Task 3: Stack and compare"
            )
            reply = self._run_tool_loop(max_tool_calls=5)
        else:
            # Step 1: Diagnose with tools — planner can inspect notebook cells
            self.llm.append_user(
                f"{budget}\n\n"
                f"Results from round {round_num - 1}:\n{last_summary or '(none)'}\n\n"
                f"Before planning, diagnose what happened. Use your tools "
                f"(read_notebook_cell, list_artifacts, read_file) to inspect actual "
                f"errors, outputs, and scores if the summary is unclear.\n\n"
                f"Then provide the plan for round {round_num} using ### Task N headers."
            )
            reply = self._run_tool_loop(max_tool_calls=5)

        if self.tracer:
            self.tracer.write(f"planning_r{round_num}", str(self.llm.messages[-2:]), reply)
        return reply

    def plan_parallel(
        self,
        round_num: int,
        last_summary: Optional[str],
        num_agents: int,
    ) -> list[str]:
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
                f"Use your tools to inspect the notebook if needed, then "
                f"review your overall strategy and provide {num_agents} diverse plans "
                f"using these exact headers:\n"
                + "\n".join(
                    f"## Agent {i} Plan\n(specific actions for agent {i})"
                    for i in range(num_agents)
                )
            )

        reply = self._run_tool_loop(max_tool_calls=5)
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
