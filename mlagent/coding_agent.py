"""Coding agent: tool-calling loop with JupyterExecutor."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from mlagent.config import LLMConfig
from mlagent.jupyter_executor import ExecutionResult, JupyterExecutor
from mlagent.llm import ToolCallingLLM
from mlagent.utils import PromptTracer, TokenTracker

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


def _discover_artifacts(work_dir: Path) -> str:
    """Scan ./artifacts/ for saved OOF predictions from prior rounds."""
    art_dir = work_dir / "artifacts"
    if not art_dir.exists():
        return ""
    files = sorted(art_dir.glob("*.npy"))
    if not files:
        return ""
    lines = ["## Available Artifacts (from previous rounds)"]
    for f in files:
        size_kb = f.stat().st_size / 1024
        lines.append(f"- {f.name} ({size_kb:.0f} KB)")
    lines.append(
        "→ You can load these with np.load() for stacking. "
        "If you have 3+ diverse OOF sets, consider building a stacking meta-learner."
    )
    return "\n".join(lines)


def _gpu_hint(work_dir: Path) -> str:
    """Detect available GPU and return a hint string for the coding agent."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return ""
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible_ids = set()
        if cuda_visible:
            visible_ids = {int(x.strip()) for x in cuda_visible.split(",") if x.strip().isdigit()}
        lines = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            idx = int(parts[0])
            if visible_ids and idx not in visible_ids:
                continue
            name, total, used = parts[1], float(parts[2]), float(parts[3])
            free = total - used
            lines.append(f"  GPU {idx}: {name}, {free:.0f}MB free / {total:.0f}MB total")
        if not lines:
            return ""
        return "GPU available:\n" + "\n".join(lines)
    except Exception:
        return ""


def _coding_system(plan: str, competition: Any, work_dir: Path, submission_name: str, metric_hint: str = "") -> str:
    desc = getattr(competition, "description", "")[:12000]
    metric_line = f"\nMetric: {metric_hint}\n" if metric_hint else ""
    gpu_info = _gpu_hint(work_dir)
    gpu_section = f"\n{gpu_info}\n" if gpu_info else ""
    return f"""You are the coding agent. Work in the Jupyter kernel (folder: {work_dir}).
Data is under ./input/ (symlink to competition data). Write submission to ./{submission_name}.

Competition description (excerpt):
{desc}
{metric_line}{gpu_section}
Plan from planning agent:
{plan}

You MUST use tools. Available tools: execute_cell, edit_cell, check_submission, read_file.

## Following the Plan
- The plan above is your primary objective for this round. Execute it faithfully.
- If the plan asks you to try a specific model family (e.g. transformer, tree-based),
  you MUST attempt it — do not skip it in favor of something you're more comfortable with.
- You may adjust implementation details (hyperparameters, exact features) based on what
  you observe, but do not substitute the plan's core goals with a different approach.
- If a planned approach fails (e.g. package error, timeout), fix the error and retry
  before falling back to simpler alternatives.

## Workflow
- Start by loading and briefly exploring the data.
- Build a working baseline that produces a valid {submission_name} as early as possible.
- Once you have a valid submission, work on the plan's goals to improve the score.
- Use check_submission to verify your output file before finishing.

## Package Installation
- If you need a library that is not installed (e.g. torch, transformers, xgboost,
  lightgbm, etc.), install it yourself by running `!pip install <package>` in an
  execute_cell call. Do NOT give up on an approach just because a package is missing.

## Bug Handling
- If a cell errors with ModuleNotFoundError or ImportError, install the missing
  package with `!pip install <package>` and retry.
- If a cell errors, read the traceback carefully and fix the specific issue.
- Use edit_cell to make small fixes to a previous cell instead of rewriting from scratch.
- If the same error keeps recurring, simplify your approach — use a simpler model or
  fewer features rather than retrying the same failing code.

## Iteration Strategy
- After getting a working model, try variations: regularization strength, learning rate,
  feature count, number of folds, different algorithms.
- Keep changes small and measurable — change one thing at a time when tuning.
- Always print your validation metric so you can track improvements.
- If stuck at the same score for 3+ consecutive steps, try a fundamentally different
  approach (different model family, different features) rather than more tuning.
- If a GPU is available, consider fine-tuning a pretrained transformer (e.g.
  bert-base-uncased, distilbert, roberta-base for NLP tasks) — this might beats
  traditional ML. Use HuggingFace transformers + a small learning rate
  (2e-5), few epochs (2-4), and K-fold CV. Save OOF predictions for stacking.

## Artifact Protocol (OOF for Stacking)
- After training each model with K-fold cross-validation, save the out-of-fold predictions:
  import os, numpy as np
  os.makedirs("./artifacts", exist_ok=True)
  np.save("./artifacts/<model_name>_oof_train.npy", oof_train_preds)
  np.save("./artifacts/<model_name>_oof_test.npy", test_preds)
- Save artifacts IMMEDIATELY after computing them — if a later step times out,
  earlier artifacts are preserved for future stacking.
- For stacking: load all saved OOF artifacts, stack as columns, train a meta-learner.
  If you have 4+ diverse OOF sources, prefer a non-linear meta-learner (XGBoost,
  LightGBM) over LR — it better captures interactions between base models.

## Calibration (context-dependent)
- For probability-based metrics (log-loss, Brier): calibration (isotonic/sigmoid)
  almost always helps. Apply CalibratedClassifierCV after stacking.
- For ranking/threshold metrics (AUC, accuracy, F1): calibration changes predicted
  probabilities but does NOT change ranking — it won't improve AUC or accuracy.
  Skip calibration unless your metric is probability-based.
- ALWAYS verify on CV that calibration actually improves YOUR specific metric
  before using calibrated predictions in the submission.

## API Compatibility
- scikit-learn >=1.5: LogisticRegression does NOT accept `multi_class` parameter (removed).
  Always use solver='lbfgs' for multiclass. CalibratedClassifierCV uses `estimator=`
  not `base_estimator=`.
- lightgbm >=4.0: Do NOT pass `early_stopping_rounds` or `verbose` to fit().
  Use callbacks: lgb.early_stopping(50), lgb.log_evaluation(100).
- xgboost >=2.0: Use `early_stopping_rounds` in constructor, not fit().
- scipy: Use `from scipy import sparse` then `sparse.hstack(...)`.

## Validation Rules (CRITICAL)
- NEVER evaluate a model on the same data it was trained on. This gives a fake score.
  Bad:  model.fit(X, y); score = metric(y, model.predict(X))
  Good: use cross_val_predict or manual K-fold to get out-of-fold predictions.
- When building a stacking meta-learner, the meta-model MUST be trained and
  evaluated using out-of-fold predictions from the base models, NOT the base
  models' training-set predictions. Use sklearn.model_selection.cross_val_predict
  on the OOF matrix, or nested K-fold.
- If your CV metric suddenly jumps by a large margin (e.g. AUC from 0.70 to 0.90),
  suspect an evaluation bug or data leakage before celebrating. Double-check that
  you are using proper OOF evaluation, not train-set resubstitution.

## Self-Assessment
- After each experiment, evaluate: did the validation metric improve? If not, why?
- Before trying something new, check if you have saved OOF artifacts for stacking.
- If the plan says to stack but you have fewer than 3 OOF sets, build more diverse
  base models first.
- If you just built a stacking model, check if calibration is appropriate for your
  metric (see Calibration section above). Don't calibrate blindly.

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
        tracker: Optional[TokenTracker] = None,
    ) -> None:
        self.coding_cfg = coding_cfg
        self.tracer = tracer
        self.agent_idx = agent_idx
        self.tracker = tracker
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
        llm = ToolCallingLLM(self.coding_cfg, tracker=self.tracker)
        lower = getattr(competition, "is_lower_better", False)
        metric_hint = f"lower is better: {lower}" if lower else "higher is better"
        llm.set_system(_coding_system(plan, competition, work_dir, submission_name, metric_hint=metric_hint))
        artifacts_hint = _discover_artifacts(work_dir)
        start_msg = (
            "Start the round. Use tools to implement the plan. "
            f"You have {max_steps} steps available."
        )
        if artifacts_hint:
            start_msg += f"\n\n{artifacts_hint}"
        llm.append_user(start_msg)
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
