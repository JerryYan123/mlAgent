"""Planning <-> Coding round loop with support for 1..N parallel coding agents."""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mlagent.coding_agent import CodingAgent
from mlagent.competition_loader import load_competition
from mlagent.config import AgentConfig
from mlagent.jupyter_executor import JupyterExecutor
from mlagent.planning_agent import PlanningAgent
from mlagent.utils import (
    ExperimentLog,
    PromptTracer,
    TokenTracker,
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


def _run_single_agent(
    agent_idx: int,
    config: AgentConfig,
    plan: str,
    competition: Any,
    jupyter: JupyterExecutor,
    agent_dir: Path,
    tracer: PromptTracer,
    results: list[Optional[str]],
    tracker: Optional[TokenTracker] = None,
) -> None:
    """Run one CodingAgent in its own workspace. Thread-safe."""
    try:
        coder = CodingAgent(config.coding_llm, tracer=tracer, agent_idx=agent_idx, tracker=tracker)
        summary = coder.run_round(
            plan=plan,
            competition=competition,
            jupyter=jupyter,
            work_dir=agent_dir,
            submission_name=config.submission_file,
            max_steps=config.max_steps_per_round,
        )
        results[agent_idx] = summary
    except Exception:
        logger.exception("Agent %d crashed", agent_idx)
        results[agent_idx] = f"Agent {agent_idx} crashed with an exception."


def run_experiment(config: AgentConfig) -> dict[str, Any]:
    setup_logging()
    _ensure_cuda_visible(config)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.working_dir) / f"run_{ts}_{config.competition_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    planner_trace_dir = None
    if config.trace_prompts:
        planner_trace_dir = (
            Path(config.trace_prompts_dir)
            if config.trace_prompts_dir
            else run_dir / "debug_prompts"
        )
    planner_tracer = PromptTracer(planner_trace_dir, enabled=bool(config.trace_prompts))

    logger.info("=" * 60)
    logger.info("Loading competition: %s", config.competition_id)
    competition = load_competition(config.competition_id)

    N = config.num_coding_agents
    logger.info(
        "Config: %d round(s), %d step(s)/round, %d agent(s)/round",
        config.max_rounds, config.max_steps_per_round, N,
    )

    tracker = TokenTracker()
    planner = PlanningAgent(config, competition, tracer=planner_tracer, tracker=tracker)
    exp_log = ExperimentLog(run_dir / "experiment_log.json")

    last_summary: Optional[str] = None
    best_score: Optional[float] = None
    start = time.time()

    all_jupyters: list[JupyterExecutor] = []

    try:
        for rnd in range(1, config.max_rounds + 1):
            if time.time() - start > config.total_time_limit:
                logger.info("Total time limit reached.")
                break

            elapsed_min = (time.time() - start) / 60
            logger.info("=" * 60)
            logger.info(
                "ROUND %d/%d  (elapsed %.1f min, best_score=%s)",
                rnd, config.max_rounds, elapsed_min, best_score,
            )
            logger.info("-" * 60)
            snap_before_plan = tracker.snapshot()
            logger.info("Planning for %d agent(s)...", N)
            plans = planner.plan_parallel(rnd, last_summary, N)
            snap_after_plan = tracker.snapshot()
            for i, p in enumerate(plans):
                first_line = p.strip().split("\n")[0][:120]
                logger.info("  Agent %d plan: %s", i, first_line)

            agent_dirs: list[Path] = []
            jupyters: list[JupyterExecutor] = []
            agent_tracers: list[PromptTracer] = []

            for i in range(N):
                agent_dir = run_dir / f"agent_{i}"
                agent_dir.mkdir(parents=True, exist_ok=True)

                jup = JupyterExecutor(
                    work_dir=agent_dir,
                    data_dir=competition.data_dir,
                    jupyter_cfg=config.jupyter,
                )

                at_dir = agent_dir / "debug_prompts" if config.trace_prompts else None
                at = PromptTracer(at_dir, enabled=bool(config.trace_prompts))

                agent_dirs.append(agent_dir)
                jupyters.append(jup)
                agent_tracers.append(at)

            all_jupyters = jupyters

            results: list[Optional[str]] = [None] * N

            if N == 1:
                _run_single_agent(
                    0, config, plans[0], competition,
                    jupyters[0], agent_dirs[0], agent_tracers[0], results,
                    tracker=tracker,
                )
            else:
                threads: list[threading.Thread] = []
                for i in range(N):
                    t = threading.Thread(
                        target=_run_single_agent,
                        args=(
                            i, config, plans[i], competition,
                            jupyters[i], agent_dirs[i], agent_tracers[i], results,
                            tracker,
                        ),
                        name=f"agent-{i}",
                    )
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

            logger.info("-" * 60)
            logger.info("Grading round %d results...", rnd)
            grades = []
            lower = getattr(competition, "is_lower_better", False)
            for i in range(N):
                sub_path = agent_dirs[i] / config.submission_file
                grade = competition.grade(sub_path)
                grades.append(grade)

                score_val = getattr(grade, "score", None)
                valid = getattr(grade, "valid_submission", False)
                logger.info(
                    "  Agent %d: score=%s, valid=%s",
                    i, score_val, valid,
                )

                if valid and score_val is not None:
                    s = float(score_val)
                    is_better = False
                    if best_score is None:
                        is_better = True
                    else:
                        is_better = s < best_score if lower else s > best_score

                    if is_better:
                        best_score = s
                        shutil.copy(sub_path, run_dir / "best_submission.csv")
                        nb_path = jupyters[i].get_notebook_path()
                        shutil.copy(nb_path, run_dir / "best_experiment.ipynb")
                        logger.info("  ★ New best score: %s (agent %d)", best_score, i)

            summaries = [r or f"Agent {i} produced no summary." for i, r in enumerate(results)]
            last_summary = format_parallel_summary(summaries, grades, best_score, rnd)

            combined_plan = "\n---\n".join(
                f"Agent {i}: {plans[i][:500]}" for i in range(N)
            )
            snap_after_code = tracker.snapshot()
            plan_in = snap_after_plan[0] - snap_before_plan[0]
            plan_out = snap_after_plan[1] - snap_before_plan[1]
            code_in = snap_after_code[0] - snap_after_plan[0]
            code_out = snap_after_code[1] - snap_after_plan[1]
            round_cost = TokenTracker.estimate_cost(
                plan_in + code_in, plan_out + code_out, config.coding_llm.model_name,
            )
            logger.info(
                "  Tokens: plan %d/%d, code %d/%d (est $%.4f)",
                plan_in, plan_out, code_in, code_out, round_cost or 0,
            )
            exp_log.log_round(
                rnd,
                combined_plan,
                last_summary,
                grades[0],
                extra={
                    "num_agents": N,
                    "all_grades": [
                        {
                            "agent": i,
                            "score": getattr(g, "score", None),
                            "valid": getattr(g, "valid_submission", None),
                        }
                        for i, g in enumerate(grades)
                    ],
                    "plan_tokens": {"input": plan_in, "output": plan_out},
                    "code_tokens": {"input": code_in, "output": code_out},
                    "estimated_cost_usd": round_cost,
                },
            )
            elapsed_min = (time.time() - start) / 60
            logger.info(
                "Round %d done (%.1f min total). Best score: %s",
                rnd, elapsed_min, best_score,
            )

            for jup in jupyters:
                jup.shutdown()
            all_jupyters = []

    finally:
        best_sub = run_dir / "best_submission.csv"
        if best_sub.exists():
            final_sub = run_dir / config.submission_file
            shutil.copy(best_sub, final_sub)
        for jup in all_jupyters:
            jup.shutdown()

    total_min = (time.time() - start) / 60
    logger.info("=" * 60)
    logger.info("DONE  total=%.1f min  best_score=%s  run_dir=%s", total_min, best_score, run_dir)
    logger.info("=" * 60)

    return {
        "run_dir": str(run_dir),
        "best_score": best_score,
        "last_summary": last_summary,
    }
