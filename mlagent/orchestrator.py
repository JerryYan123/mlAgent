"""Task-based orchestrator: planner outputs tasks, each runs as a separate coding agent call."""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mlagent.coding_agent import CodingAgent
from mlagent.competition_loader import load_competition
from mlagent.config import AgentConfig
from mlagent.eval_agent import setup_eval
from mlagent.jupyter_executor import JupyterExecutor
from mlagent.planning_agent import PlanningAgent
from mlagent.utils import (
    ExperimentLog,
    PromptTracer,
    TokenTracker,
    extract_round_diagnostics,
    format_parallel_summary,
    setup_logging,
    auto_select_gpu,
)

logger = logging.getLogger(__name__)


def _ensure_cuda_visible(cfg) -> None:
    g = cfg.jupyter.gpu
    if g == "auto" or g is None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", auto_select_gpu())
    elif isinstance(g, str) and g.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = g.strip()


def _parse_tasks(plan: str) -> list[str]:
    """Split a plan into tasks by ### Task headers. Falls back to single task."""
    pattern = r"###\s*Task\s*\d+"
    splits = list(re.finditer(pattern, plan, re.IGNORECASE))
    if len(splits) < 2:
        return [plan]
    tasks = []
    for idx, match in enumerate(splits):
        start = match.start()
        end = splits[idx + 1].start() if idx + 1 < len(splits) else len(plan)
        tasks.append(plan[start:end].strip())
    return tasks


def run_experiment(config: AgentConfig) -> dict[str, Any]:
    setup_logging()
    _ensure_cuda_visible(config)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.working_dir) / f"run_{ts}_{config.competition_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    planner_trace_dir = run_dir / "debug_prompts" if config.trace_prompts else None
    planner_tracer = PromptTracer(planner_trace_dir, enabled=bool(config.trace_prompts))

    logger.info("=" * 60)
    logger.info("Loading competition: %s", config.competition_id)
    competition = load_competition(config.competition_id)
    lower = getattr(competition, "is_lower_better", False)

    logger.info(
        "Config: %d round(s), %d step(s)/round",
        config.max_rounds, config.max_steps_per_round,
    )

    tracker = TokenTracker()
    planner = PlanningAgent(config, competition, tracer=planner_tracer, tracker=tracker)
    exp_log = ExperimentLog(run_dir / "experiment_log.json")

    last_summary: Optional[str] = None
    best_score: Optional[float] = None
    start = time.time()

    # Persistent kernel + agent workspace
    agent_dir = run_dir / "agent_0"
    agent_dir.mkdir(parents=True, exist_ok=True)
    jupyter = JupyterExecutor(
        work_dir=agent_dir,
        data_dir=competition.data_dir,
        jupyter_cfg=config.jupyter,
    )
    agent_tracer = PromptTracer(
        agent_dir / "debug_prompts" if config.trace_prompts else None,
        enabled=bool(config.trace_prompts),
    )

    # Eval kernel (isolated)
    eval_kernel = None
    try:
        eval_kernel = setup_eval(config, competition, run_dir, [agent_dir], tracker)
        if eval_kernel:
            logger.info("Eval kernel ready=%s", eval_kernel.ready)
        else:
            logger.warning("Eval setup returned None")
    except Exception as e:
        logger.warning("Eval setup failed: %s — proceeding without holdout", e)

    def _eval_score_fn():
        if eval_kernel and eval_kernel.ready:
            pred = agent_dir / "holdout_predictions.csv"
            return eval_kernel.score(pred)
        return None

    try:
        for rnd in range(1, config.max_rounds + 1):
            if time.time() - start > config.total_time_limit:
                logger.info("Total time limit reached.")
                break

            elapsed_min = (time.time() - start) / 60
            logger.info("=" * 60)
            logger.info("ROUND %d/%d  (elapsed %.1f min)", rnd, config.max_rounds, elapsed_min)
            logger.info("-" * 60)

            snap_before_plan = tracker.snapshot()

            # Plan
            planner.set_notebook_context(jupyter.nb.cells, agent_dir)
            plan = planner.plan(rnd, last_summary)
            snap_after_plan = tracker.snapshot()

            # Parse plan into tasks
            tasks = _parse_tasks(plan)
            logger.info("Plan has %d task(s)", len(tasks))
            steps_per_task = config.max_steps_per_round

            # Run each task as a separate coding agent call
            task_results = []
            for task_idx, task_plan in enumerate(tasks):
                task_label = task_plan.split("\n")[0][:80]
                logger.info("  Task %d/%d: %s (%d steps)", task_idx + 1, len(tasks), task_label, steps_per_task)

                coder = CodingAgent(
                    config.coding_llm,
                    tracer=agent_tracer,
                    agent_idx=0,
                    tracker=tracker,
                    eval_score_fn=_eval_score_fn,
                )
                summary = coder.run_round(
                    plan=task_plan,
                    competition=competition,
                    jupyter=jupyter,
                    work_dir=agent_dir,
                    submission_name=config.submission_file,
                    max_steps=steps_per_task,
                )

                # Score after each task
                holdout_score = None
                if eval_kernel and eval_kernel.ready:
                    pred_path = agent_dir / "holdout_predictions.csv"
                    if pred_path.exists():
                        holdout_score = eval_kernel.score(pred_path)
                        logger.info("  Holdout score: %s", holdout_score)
                    else:
                        logger.info("  holdout_predictions.csv not found — skipping holdout scoring")
                else:
                    logger.info("  Eval kernel not ready (kernel=%s, ready=%s)",
                                eval_kernel is not None, getattr(eval_kernel, 'ready', None))

                task_results.append({
                    "task": task_label,
                    "summary": summary,
                    "holdout_score": holdout_score,
                })

                score_str = f", holdout={holdout_score:.5f}" if holdout_score is not None else ""
                logger.info("  Task %d done%s", task_idx + 1, score_str)

            logger.info("-" * 60)

            # Save submission
            sub_path = agent_dir / config.submission_file
            if sub_path.exists():
                shutil.copy(sub_path, run_dir / "best_submission.csv")
                shutil.copy(jupyter.get_notebook_path(), run_dir / "best_experiment.ipynb")

            # Build round summary for planner
            summary_parts = [f"=== After round {rnd} ==="]
            for tr in task_results:
                summary_parts.append(f"--- {tr['task']} ---\n{tr['summary']}")
                if tr["holdout_score"] is not None:
                    direction = "lower is better" if lower else "higher is better"
                    summary_parts.append(f"Holdout score after this task: {tr['holdout_score']:.5f} ({direction})")
            last_summary = "\n\n".join(summary_parts)

            # Hidden real grade (only in log)
            hidden_grade = None
            if sub_path.exists():
                try:
                    grade = competition.grade(sub_path)
                    hidden_grade = getattr(grade, "score", None)
                    logger.info("  [HIDDEN] Real grade: %s", hidden_grade)
                except Exception:
                    pass

            # Diagnostics
            nb_cells = jupyter.nb.cells
            diag = extract_round_diagnostics(
                nb_cells,
                round_start_cell=max(0, len(nb_cells) - config.max_steps_per_round - 5),
                artifacts_dir=agent_dir / "artifacts",
            )
            if diag:
                last_summary += f"\n\n--- Diagnostics ---\n{diag}"

            # Log
            snap_after_code = tracker.snapshot()
            plan_in = snap_after_plan[0] - snap_before_plan[0]
            plan_out = snap_after_plan[1] - snap_before_plan[1]
            code_in = snap_after_code[0] - snap_after_plan[0]
            code_out = snap_after_code[1] - snap_after_plan[1]
            round_cost = TokenTracker.estimate_cost(
                plan_in + code_in, plan_out + code_out, config.coding_llm.model_name,
            )
            logger.info("  Tokens: plan %d/%d, code %d/%d (est $%.4f)",
                        plan_in, plan_out, code_in, code_out, round_cost or 0)

            exp_log.log_round(rnd, plan[:1000], last_summary, None, extra={
                "num_tasks": len(tasks),
                "task_holdout_scores": [tr["holdout_score"] for tr in task_results],
                "hidden_grade": hidden_grade,
                "plan_tokens": {"input": plan_in, "output": plan_out},
                "code_tokens": {"input": code_in, "output": code_out},
                "estimated_cost_usd": round_cost,
            })

            elapsed_min = (time.time() - start) / 60
            logger.info("Round %d done (%.1f min total).", rnd, elapsed_min)

    finally:
        # Final grade
        best_sub = run_dir / "best_submission.csv"
        if best_sub.exists():
            final_sub = run_dir / config.submission_file
            shutil.copy(best_sub, final_sub)
            grade = competition.grade(final_sub)
            best_score = getattr(grade, "score", None)
            logger.info("Final grade: score=%s, valid=%s",
                        best_score, getattr(grade, "valid_submission", None))
            exp_log.log_round(0, "", "", grade, extra={"final_grade": True})
        jupyter.shutdown()
        if eval_kernel:
            eval_kernel.shutdown()

    total_min = (time.time() - start) / 60
    logger.info("=" * 60)
    logger.info("DONE  total=%.1f min  best_score=%s  run_dir=%s",
                total_min, best_score, run_dir)
    logger.info("=" * 60)

    return {"run_dir": str(run_dir), "best_score": best_score, "last_summary": last_summary}
