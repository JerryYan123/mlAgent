"""Evaluation agent: writes evaluation code via LLM, runs it in an isolated kernel.

Flow:
  1. EvalAgent (LLM) inspects data + competition description → writes eval Python code
  2. EvalKernel runs that code in an isolated Jupyter kernel:
     - Splits train (85/15), overwrites train file with train_split
     - Holds holdout labels in kernel memory only (never on disk)
     - Defines evaluate(submission_path) → float
  3. Orchestrator calls EvalKernel.score() each round
"""

from __future__ import annotations

import ast
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

from mlagent.config import LLMConfig, JupyterConfig
from mlagent.holdout import find_train_file, find_sample_submission, get_target_info
from mlagent.jupyter_executor import JupyterExecutor
from mlagent.llm import ToolCallingLLM
from mlagent.utils import TokenTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM prompt for generating evaluation code
# ---------------------------------------------------------------------------

EVAL_SYSTEM = """\
You are an evaluation engineer. Write a self-contained Python script that:

1. Loads training data from the specified path.
2. Splits it into ~85% train_split and ~15% holdout using sklearn's train_test_split
   with random_state=42. If the target is categorical, use stratify.
3. Overwrites the original train file with train_split only.
4. Keeps holdout labels in memory as global variables (NEVER writes them to disk).
5. Defines: evaluate(submission_path) → float
   - Reads the submission CSV at submission_path
   - Aligns predictions with holdout by the ID column
   - Computes and returns the competition metric as a float

Requirements:
- Print "SPLIT_DONE" after overwriting the train file
- Save holdout features (WITHOUT labels but WITH the ID column and all other
  non-label columns) to input/holdout.csv so the coding agent can predict on it.
  The coding agent will save predictions as holdout_predictions.csv.
- Print "HOLDOUT_SIZE: <N>" with the holdout row count
- Define evaluate(submission_path) that reads the predictions CSV, aligns with
  holdout labels by the ID column, and returns the metric score as a float.
- Print "EVAL_READY" after defining evaluate()
- Use only: numpy, pandas, sklearn, scipy (no other packages)
- Handle edge cases: missing IDs, wrong columns, NaN values → return None
- NEVER write holdout labels to disk — keep them only in memory variables

Output ONLY the Python code, no explanations."""


class EvalAgent:
    """One-shot LLM agent that generates evaluation code."""

    def __init__(self, llm_cfg: LLMConfig, tracker: Optional[TokenTracker] = None):
        self.llm_cfg = llm_cfg
        self.tracker = tracker

    def generate_eval_code(self, competition: Any, data_dir: Path, prev_error: str = "") -> Optional[str]:
        """Ask LLM to write evaluation code. Returns code string or None."""
        import pandas as pd

        info = get_target_info(data_dir)
        if info is None:
            logger.warning("EvalAgent: could not discover target info")
            return None

        llm = ToolCallingLLM(self.llm_cfg, tracker=self.tracker)
        llm.set_system(EVAL_SYSTEM)

        # Build context
        desc = getattr(competition, "description", "")[:4000]
        lower = getattr(competition, "is_lower_better", False)
        metric_dir = "lower is better" if lower else "higher is better"

        # Data preview
        sample_sub_path = find_sample_submission(data_dir)
        sample_sub_text = ""
        if sample_sub_path:
            df = pd.read_csv(sample_sub_path, nrows=3)
            sample_sub_text = f"Sample submission ({sample_sub_path.name}):\n{df.to_string()}"

        train_rel = str(info["train_path"].relative_to(data_dir))

        prompt = (
            f"Competition: {desc[:2000]}\n\n"
            f"Metric: {metric_dir}\n\n"
            f"Train file: input/{train_rel} (format: {info['train_format']})\n"
            f"ID column: {info['id_col']}\n"
            f"Target columns in submission: {info['target_cols']}\n"
            f"Label columns in train: {info['train_label_cols']}\n\n"
            f"{sample_sub_text}\n\n"
        )

        if prev_error:
            prompt += (
                f"IMPORTANT: Your previous attempt failed with this error:\n"
                f"{prev_error}\n"
                f"Fix the issue and try again.\n\n"
            )

        prompt += "Write the evaluation code now."

        llm.append_user(prompt)
        response = llm.chat_no_tools()
        if not response:
            return None

        # Extract code from markdown blocks if present
        code = response
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        return code.strip()

    @staticmethod
    def verify_code(code: str) -> tuple[bool, str]:
        """Verify generated code is valid Python and defines evaluate()."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check that evaluate() is defined
        has_evaluate = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
                has_evaluate = True
                break

        if not has_evaluate:
            return False, "No 'evaluate' function defined"

        return True, "OK"


# ---------------------------------------------------------------------------
# EvalKernel — isolated Jupyter kernel for scoring
# ---------------------------------------------------------------------------

class EvalKernel:
    """Manages an isolated evaluation kernel with holdout data in memory."""

    def __init__(
        self,
        run_dir: Path,
        data_dir: Path,
        jupyter_cfg: JupyterConfig,
    ):
        self.run_dir = run_dir
        self.data_dir = data_dir
        self.eval_data_dir = run_dir / "_eval_data"
        self.eval_work_dir = run_dir / "_eval"
        self._ready = False
        self._jupyter: Optional[JupyterExecutor] = None
        self._jupyter_cfg = jupyter_cfg

    def prepare_data(self) -> bool:
        """Copy and clean competition data. Must be called before setup()."""
        try:
            if self.eval_data_dir.exists():
                shutil.rmtree(self.eval_data_dir)
            shutil.copytree(self.data_dir, self.eval_data_dir)
            logger.info("Eval: copied data to %s", self.eval_data_dir)

            # Step 1b: Remove duplicate train files to prevent agent from using unsplit copies.
            primary_train = find_train_file(self.eval_data_dir)
            if primary_train:
                primary_resolved = primary_train.resolve()
                removed = []
                for f in self.eval_data_dir.rglob("*"):
                    if not f.is_file():
                        continue
                    fname = f.stem.lower()
                    if "train" not in fname:
                        continue
                    # Keep the primary train file, sample submissions, and non-data files
                    if f.resolve() == primary_resolved:
                        continue
                    if "sample" in fname or "submission" in fname:
                        continue
                    if f.suffix in (".csv", ".json", ".jsonl", ".parquet", ".pkl"):
                        f.unlink()
                        removed.append(str(f.relative_to(self.eval_data_dir)))
                if removed:
                    logger.info("Eval: removed %d duplicate train files: %s", len(removed), removed[:5])

            # Step 1c: Convert JSON data files to CSV for easier agent consumption
            import pandas as pd
            for json_file in list(self.eval_data_dir.glob("*.json")):
                try:
                    import json as _json
                    with open(json_file) as f:
                        data = _json.load(f)
                    if isinstance(data, list) and data:
                        df = pd.DataFrame(data)
                        csv_path = json_file.with_suffix(".csv")
                        df.to_csv(csv_path, index=False)
                        json_file.unlink()  # remove the JSON
                        logger.info("Eval: converted %s → %s (%d rows)", json_file.name, csv_path.name, len(df))
                except Exception as e:
                    logger.warning("Eval: failed to convert %s: %s", json_file.name, e)

            return True
        except Exception as e:
            logger.error("Eval data preparation failed: %s", e)
            return False

    def setup(self, eval_code: str) -> bool:
        """Create eval kernel and run eval code. Call prepare_data() first."""
        try:
            # Step 2: Create eval kernel
            self.eval_work_dir.mkdir(parents=True, exist_ok=True)
            # Symlink input to our copy
            eval_input = self.eval_work_dir / "input"
            if eval_input.exists():
                eval_input.unlink()
            eval_input.symlink_to(self.eval_data_dir.resolve(), target_is_directory=True)

            self._jupyter = JupyterExecutor(
                work_dir=self.eval_work_dir,
                data_dir=self.eval_data_dir,
                jupyter_cfg=self._jupyter_cfg,
                env_name="eval_kernel",
            )

            # Step 3: Execute eval code
            result = self._jupyter.execute_cell(eval_code, "Setup evaluation")

            if not result.success:
                logger.error("Eval setup failed: %s", result.error[:500])
                return False

            if "SPLIT_DONE" not in result.output:
                logger.error("Eval: SPLIT_DONE marker not found")
                return False

            if "EVAL_READY" not in result.output:
                logger.error("Eval: EVAL_READY marker not found")
                return False

            # Log holdout size
            for line in result.output.split("\n"):
                if "HOLDOUT_SIZE:" in line:
                    logger.info("Eval: %s", line.strip())

            # Step 4: Sanity check — create dummy predictions and verify evaluate() works
            sanity_code = (
                "import pandas as pd, numpy as np, glob\n"
                "hf = pd.read_csv('input/holdout.csv')\n"
                "# Get ID and target columns from sample submission\n"
                "sub_files = glob.glob('input/sample_submission*.csv') + glob.glob('input/sampleSubmission*.csv')\n"
                "if sub_files:\n"
                "    sample = pd.read_csv(sub_files[0], nrows=1)\n"
                "    id_col = sample.columns[0]\n"
                "    target_cols = sample.columns[1:].tolist()\n"
                "else:\n"
                "    id_col = hf.columns[0]\n"
                "    target_cols = []\n"
                "dummy = {id_col: hf[id_col] if id_col in hf.columns else range(len(hf))}\n"
                "for tc in target_cols:\n"
                "    dummy[tc] = 0.5\n"
                "pd.DataFrame(dummy).to_csv('dummy_pred.csv', index=False)\n"
                "try:\n"
                "    s = evaluate('dummy_pred.csv')\n"
                "    print(f'SANITY_SCORE: {s}')\n"
                "except Exception as e:\n"
                "    print(f'SANITY_FAIL: {e}')\n"
            )
            sanity_result = self._jupyter.execute_cell(sanity_code, "Sanity check evaluate()")
            if "SANITY_FAIL" in sanity_result.output:
                logger.error("Eval sanity check failed: %s", sanity_result.output[:300])
                return False
            if "SANITY_SCORE:" not in sanity_result.output:
                logger.error("Eval sanity check: no score returned. Output: %s", sanity_result.output[:300])
                return False

            # Verify score is a finite number
            for line in sanity_result.output.split("\n"):
                if "SANITY_SCORE:" in line:
                    val = line.split("SANITY_SCORE:")[1].strip()
                    try:
                        sanity_score = float(val)
                        if not (sanity_score == sanity_score):  # NaN check
                            logger.error("Eval sanity: score is NaN")
                            return False
                        logger.info("Eval sanity score: %s (OK)", sanity_score)
                    except ValueError:
                        logger.error("Eval sanity: score not a number: %s", val)
                        return False

            self._ready = True
            logger.info("Eval kernel ready and verified")
            return True

        except Exception as e:
            logger.error("Eval setup exception: %s", e)
            return False

    def setup_data_isolation(self, agent_dirs: list[Path]) -> None:
        """Replace coding agent input/ symlinks to point to our modified data."""
        target = self.eval_data_dir.resolve()  # absolute path to avoid symlink issues
        for agent_dir in agent_dirs:
            inp = agent_dir / "input"
            if inp.is_symlink():
                inp.unlink()
            elif inp.exists():
                shutil.rmtree(inp)
            inp.symlink_to(target, target_is_directory=True)
            logger.info("Eval: re-symlinked %s → %s", inp, target)

    def score(self, submission_path: Path) -> Optional[float]:
        """Score a submission against the holdout. Returns score or None."""
        if not self._ready or not self._jupyter:
            return None

        if not submission_path.exists():
            return None

        try:
            # Copy submission to eval workspace
            eval_sub = self.eval_work_dir / "submission_to_eval.csv"
            shutil.copy(submission_path, eval_sub)

            code = (
                "try:\n"
                "    _score = evaluate('submission_to_eval.csv')\n"
                "    if _score is not None:\n"
                "        print(f'HOLDOUT_SCORE: {_score}')\n"
                "    else:\n"
                "        print('HOLDOUT_SCORE: NONE')\n"
                "except Exception as e:\n"
                "    print(f'EVAL_ERROR: {e}')\n"
            )
            result = self._jupyter.execute_cell(code, "Score submission")

            if not result.success:
                logger.warning("Eval scoring cell failed: %s", result.error[:200])
                return None

            for line in result.output.split("\n"):
                if "HOLDOUT_SCORE:" in line:
                    val = line.split("HOLDOUT_SCORE:")[1].strip()
                    if val == "NONE":
                        return None
                    try:
                        return float(val)
                    except (ValueError, IndexError):
                        pass
                if "EVAL_ERROR:" in line:
                    logger.warning("Eval error: %s", line)

            return None
        except Exception as e:
            logger.warning("Eval scoring exception: %s", e)
            return None

    @property
    def ready(self) -> bool:
        return self._ready

    def shutdown(self):
        if self._jupyter:
            self._jupyter.shutdown()
            self._jupyter = None


# ---------------------------------------------------------------------------
# Setup helper — used by orchestrators
# ---------------------------------------------------------------------------

def setup_eval(
    config: Any,
    competition: Any,
    run_dir: Path,
    agent_dirs: list[Path],
    tracker: Optional[TokenTracker] = None,
) -> Optional[EvalKernel]:
    """Set up evaluation kernel with LLM-generated code. Falls back to HoldoutValidator.

    Returns EvalKernel if successful, None otherwise.
    """
    if not getattr(config, "enable_holdout", True):
        logger.info("Eval: holdout disabled in config")
        return None

    eval_llm_cfg = getattr(config, "eval_llm", None) or config.planning_llm
    agent = EvalAgent(eval_llm_cfg, tracker=tracker)
    kernel = EvalKernel(run_dir, competition.data_dir, config.jupyter)

    # Prepare data first (copy + clean + JSON→CSV), then generate eval code
    if not kernel.prepare_data():
        logger.warning("Eval: data preparation failed")
        return None

    # LLM sees the prepared data (CSV files, not JSON)
    eval_data_dir = kernel.eval_data_dir
    last_error = ""
    for attempt in range(3):
        code = agent.generate_eval_code(competition, eval_data_dir, prev_error=last_error)
        if not code:
            last_error = "Code generation returned empty response."
            logger.warning("Eval: code generation returned None (attempt %d)", attempt + 1)
            continue

        ok, msg = EvalAgent.verify_code(code)
        if not ok:
            last_error = f"Code verification failed: {msg}"
            logger.warning("Eval: %s (attempt %d)", last_error, attempt + 1)
            continue

        if kernel.setup(code):
            kernel.setup_data_isolation(agent_dirs)
            return kernel
        else:
            last_error = "Eval kernel setup/sanity check failed. The evaluate() function may have a bug."
            logger.warning("Eval: kernel setup failed (attempt %d)", attempt + 1)

    # Fallback: try deterministic HoldoutValidator
    logger.info("Eval: LLM eval failed, trying deterministic fallback")
    try:
        from mlagent.holdout import HoldoutValidator
        hv = HoldoutValidator(
            holdout_fraction=getattr(config, "holdout_fraction", 0.15),
        )
        # Need to copy data first for isolation
        eval_data_dir = run_dir / "_eval_data"
        if not eval_data_dir.exists():
            shutil.copytree(competition.data_dir, eval_data_dir)
            # Remove duplicate train files — keep only the primary one
            primary_train = find_train_file(eval_data_dir)
            if primary_train:
                primary_resolved = primary_train.resolve()
                for f in eval_data_dir.rglob("*"):
                    if not f.is_file():
                        continue
                    fname = f.stem.lower()
                    if "train" not in fname:
                        continue
                    if f.resolve() == primary_resolved:
                        continue
                    if "sample" in fname or "submission" in fname:
                        continue
                    if f.suffix in (".csv", ".json", ".jsonl", ".parquet", ".pkl"):
                        f.unlink()
            # Convert JSON → CSV
            import pandas as pd
            for json_file in list(eval_data_dir.glob("*.json")):
                try:
                    import json as _json
                    with open(json_file) as jf:
                        data = _json.load(jf)
                    if isinstance(data, list) and data:
                        pd.DataFrame(data).to_csv(json_file.with_suffix(".csv"), index=False)
                        json_file.unlink()
                except Exception:
                    pass
        # Re-symlink agent inputs
        eval_data_abs = eval_data_dir.resolve()
        for agent_dir in agent_dirs:
            inp = agent_dir / "input"
            if inp.is_symlink():
                inp.unlink()
            elif inp.exists():
                shutil.rmtree(inp)
            inp.symlink_to(eval_data_abs, target_is_directory=True)

        if hv.setup(competition.data_dir, eval_data_dir, competition):
            logger.info("Eval: deterministic fallback succeeded")
            # Wrap HoldoutValidator in a duck-typed object matching EvalKernel interface
            return _HoldoutKernelAdapter(hv, agent_dirs)
    except Exception as e:
        logger.warning("Eval: deterministic fallback also failed: %s", e)

    logger.warning("Eval: all eval setup attempts failed, proceeding without holdout")
    return None


class _HoldoutKernelAdapter:
    """Wraps HoldoutValidator to match EvalKernel interface."""

    def __init__(self, hv, agent_dirs: list[Path]):
        self._hv = hv
        self._agent_dirs = agent_dirs

    def score(self, submission_path: Path) -> Optional[float]:
        # HoldoutValidator.score expects agent_dir not submission_path
        # We need to find the agent dir from the submission path
        agent_dir = submission_path.parent
        return self._hv.score(agent_dir)

    @property
    def ready(self) -> bool:
        return self._hv.active

    def shutdown(self):
        pass  # No kernel to shutdown

    def setup_data_isolation(self, agent_dirs: list[Path]) -> None:
        pass  # Already done in setup_eval
