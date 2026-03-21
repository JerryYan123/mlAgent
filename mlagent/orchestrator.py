"""Planning <-> Coding round loop."""

from __future__ import annotations

import logging
import os
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
    format_summary,
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


def run_experiment(config: AgentConfig) -> dict[str, Any]:
    setup_logging()
    _ensure_cuda_visible(config)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.working_dir) / f"run_{ts}_{config.competition_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    trace_dir = None
    if config.trace_prompts:
        trace_dir = Path(config.trace_prompts_dir) if config.trace_prompts_dir else run_dir / "debug_prompts"
    tracer = PromptTracer(trace_dir, enabled=bool(config.trace_prompts))

    logger.info("Loading competition %s", config.competition_id)
    competition = load_competition(config.competition_id)

    jupyter = JupyterExecutor(
        work_dir=run_dir,
        data_dir=competition.data_dir,
        jupyter_cfg=config.jupyter,
        monitor_llm=config.monitor_llm,
    )

    planner = PlanningAgent(config, competition, tracer=tracer)
    exp_log = ExperimentLog(run_dir / "experiment_log.json")

    last_summary: Optional[str] = None
    best_score: Optional[float] = None
    start = time.time()

    try:
        for rnd in range(1, config.max_rounds + 1):
            if time.time() - start > config.total_time_limit:
                logger.info("Total time limit reached.")
                break

            plan = planner.plan(rnd, last_summary)
            logger.info("Round %s plan:\n%s", rnd, plan[:2000])

            coder = CodingAgent(config.coding_llm, tracer=tracer)
            coding_summary = coder.run_round(
                plan=plan,
                competition=competition,
                jupyter=jupyter,
                work_dir=run_dir,
                submission_name=config.submission_file,
                max_steps=config.max_steps_per_round,
            )

            sub_path = run_dir / config.submission_file
            grade = competition.grade(sub_path)

            if getattr(grade, "valid_submission", False) and grade.score is not None:
                if best_score is None:
                    best_score = float(grade.score)
                else:
                    lower = getattr(competition, "is_lower_better", False)
                    s = float(grade.score)
                    if lower:
                        best_score = min(best_score, s)
                    else:
                        best_score = max(best_score, s)

            last_summary = format_summary(coding_summary, grade, best_score, rnd)
            exp_log.log_round(
                rnd,
                plan,
                coding_summary,
                grade,
                extra={"notebook": str(jupyter.get_notebook_path())},
            )
            logger.info("Round %s done. Grade: %s", rnd, getattr(grade, "score", None))

    finally:
        jupyter.shutdown()

    return {
        "run_dir": str(run_dir),
        "best_score": best_score,
        "last_summary": last_summary,
    }
