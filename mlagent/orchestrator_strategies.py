"""Strategy-based orchestrator: dispatches to baseline, board_replan, or codex_todo."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mlagent.coding_agent import CodingAgent, build_tools, _coding_system
from mlagent.competition_loader import load_competition
from mlagent.config import AgentConfig
from mlagent.jupyter_executor import JupyterExecutor
from mlagent.llm import ToolCallingLLM
from mlagent.utils import (
    ExperimentLog,
    PromptTracer,
    format_parallel_summary,
    setup_logging,
    auto_select_gpu,
)

logger = logging.getLogger(__name__)


def run_experiment(config: AgentConfig) -> dict[str, Any]:
    """Dispatch to the right strategy based on config.planning_strategy."""
    strategy = getattr(config, "planning_strategy", "baseline")
    if strategy == "baseline":
        from mlagent.orchestrator import run_experiment as _baseline
        return _baseline(config)
    if strategy == "board_replan":
        return _run_board_replan(config)
    if strategy == "codex_todo":
        return _run_codex_todo(config)
    raise ValueError(f"Unknown planning_strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ensure_cuda_visible(cfg: AgentConfig) -> None:
    g = cfg.jupyter.gpu
    if g == "auto" or g is None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", auto_select_gpu())
    elif isinstance(g, str) and g.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = g.strip()


def _setup_run_dir(config: AgentConfig, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.working_dir) / f"run_{ts}_{tag}_{config.competition_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _grade_and_track_best(
    competition: Any,
    sub_path: Path,
    nb_path: Path,
    run_dir: Path,
    best_score: float | None,
) -> float | None:
    grade = competition.grade(sub_path)
    score_val = getattr(grade, "score", None)
    valid = getattr(grade, "valid_submission", False)
    logger.info("  Grade: score=%s, valid=%s", score_val, valid)

    if valid and score_val is not None:
        s = float(score_val)
        lower = getattr(competition, "is_lower_better", False)
        is_better = best_score is None or (s < best_score if lower else s > best_score)
        if is_better:
            best_score = s
            shutil.copy(sub_path, run_dir / "best_submission.csv")
            shutil.copy(nb_path, run_dir / "best_experiment.ipynb")
            logger.info("  ★ New best score: %s", best_score)
    return best_score, grade


def _run_coding_step(
    llm: ToolCallingLLM,
    tools: list[dict],
    coder: CodingAgent,
    jupyter: JupyterExecutor,
    work_dir: Path,
    submission_name: str,
    step: int,
    max_steps: int,
    tracer: PromptTracer | None,
    tag: str,
) -> None:
    """Execute one tool-calling step (shared between board and todo strategies)."""
    if step > 0:
        remaining = max_steps - step
        status = f"[Status] Step {step + 1}/{max_steps} ({remaining} remaining)."
        if remaining <= 2:
            status += f" Running low on steps. Prioritize producing a valid {submission_name} now."
        llm.append_user(status)

    step_result = llm.complete_with_tools(tools)
    tool_calls = step_result.tool_calls

    if not tool_calls:
        logger.info("%s Step %d/%d — no tool call, prompting retry", tag, step + 1, max_steps)
        llm.append_assistant(step_result.content, None)
        llm.append_user("You must call one of the tools (execute_cell, edit_cell, check_submission, read_file).")
        return

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
        logger.info("%s Step %d/%d — %s: %s", tag, step + 1, max_steps, tc.name, goal)
        out = coder._dispatch_tool(tc.name, tc.arguments, jupyter, work_dir, submission_name)
        result_data = {}
        try:
            result_data = json.loads(out) if out.startswith("{") else {}
        except Exception:
            pass
        ok = result_data.get("success", "?")
        et = result_data.get("execution_time")
        et_str = f" ({et:.1f}s)" if isinstance(et, (int, float)) else ""
        logger.info("%s         → success=%s%s", tag, ok, et_str)
        if tracer:
            tracer.write(
                f"coding_tool_{tc.name}_s{step}",
                json.dumps({"args": tc.arguments}, indent=2),
                out[:8000],
            )
        llm.append_tool_result(tc.id, out)


# ---------------------------------------------------------------------------
# Strategy: board_replan
# ---------------------------------------------------------------------------

def _run_board_replan(config: AgentConfig) -> dict[str, Any]:
    from mlagent.board_planner import BoardPlanningAgent
    from mlagent.experiment_board import ExperimentBoard

    setup_logging()
    _ensure_cuda_visible(config)
    run_dir = _setup_run_dir(config, "board")
    tracer = PromptTracer(run_dir / "debug_prompts", enabled=bool(config.trace_prompts))
    competition = load_competition(config.competition_id)
    board = ExperimentBoard()

    logger.info("=" * 60)
    logger.info("Strategy: BOARD_REPLAN")
    logger.info("Loading competition: %s", config.competition_id)
    logger.info(
        "Config: %d round(s), %d step(s)/round, replan every %d steps",
        config.max_rounds, config.max_steps_per_round, config.replan_interval,
    )

    planner = BoardPlanningAgent(config, competition, board, tracer=tracer)
    exp_log = ExperimentLog(run_dir / "experiment_log.json")

    last_summary: Optional[str] = None
    best_score: Optional[float] = None
    start = time.time()

    agent_dir = run_dir / "agent_0"

    try:
        for rnd in range(1, config.max_rounds + 1):
            if time.time() - start > config.total_time_limit:
                logger.info("Total time limit reached.")
                break

            elapsed_min = (time.time() - start) / 60
            logger.info("=" * 60)
            logger.info("ROUND %d/%d  (elapsed %.1f min, best_score=%s)", rnd, config.max_rounds, elapsed_min, best_score)
            logger.info("-" * 60)

            plan = planner.plan(rnd, last_summary)
            logger.info("Plan generated (board has %d entries)", len(board.entries))

            agent_dir.mkdir(parents=True, exist_ok=True)
            jupyter = JupyterExecutor(
                work_dir=agent_dir,
                data_dir=competition.data_dir,
                jupyter_cfg=config.jupyter,
            )
            at = PromptTracer(agent_dir / "debug_prompts", enabled=bool(config.trace_prompts))
            coder = CodingAgent(config.coding_llm, tracer=at, agent_idx=0)
            tag = "[Agent 0]"

            llm = ToolCallingLLM(config.coding_llm)
            llm.set_system(_coding_system(plan, competition, agent_dir, config.submission_file))
            llm.append_user(
                f"Start the round. Use tools to implement the plan. "
                f"You have {config.max_steps_per_round} steps. "
                f"The strategist will check in every {config.replan_interval} steps."
            )
            tools = build_tools(config.submission_file)
            max_steps = config.max_steps_per_round
            interval = config.replan_interval

            try:
                for step in range(max_steps):
                    _run_coding_step(llm, tools, coder, jupyter, agent_dir, config.submission_file, step, max_steps, at, tag)

                    if (step + 1) % interval == 0 and (step + 1) < max_steps:
                        logger.info("%s Checkpoint at step %d — requesting mid-summary...", tag, step + 1)
                        llm.append_user(
                            "Pause. Briefly summarize what you've done so far, any scores/metrics "
                            "you've observed, and what you plan to do next."
                        )
                        mid_summary = llm.chat_no_tools()
                        board.add("result", mid_summary[:2000], rnd, step + 1)

                        logger.info("%s Replanning...", tag)
                        revised = planner.replan(rnd, step + 1, mid_summary)
                        llm.append_user(f"[Strategist Update]\n{revised}")
                        logger.info("%s Replan injected, continuing...", tag)

                logger.info("%s All %d steps done, generating summary...", tag, max_steps)
                llm.append_user(
                    "The coding round is over. Provide a concise summary: "
                    "what was tried, what worked, what errors occurred, current metrics, "
                    "and suggestions for the next round."
                )
                summary = llm.chat_no_tools() or "No summary generated."
                board.add("result", f"Round {rnd} final: {summary[:2000]}", rnd)

            finally:
                jupyter.shutdown()

            sub_path = agent_dir / config.submission_file
            nb_path = jupyter.get_notebook_path()
            best_score, grade = _grade_and_track_best(competition, sub_path, nb_path, run_dir, best_score)
            last_summary = format_parallel_summary([summary], [grade], best_score, rnd)

            exp_log.log_round(rnd, plan[:1000], last_summary, grade, extra={"strategy": "board_replan", "board_size": len(board.entries)})
            board.save(run_dir / "experiment_board.json")

            elapsed_min = (time.time() - start) / 60
            logger.info("Round %d done (%.1f min total). Best score: %s", rnd, elapsed_min, best_score)

    finally:
        best_sub = run_dir / "best_submission.csv"
        if best_sub.exists():
            shutil.copy(best_sub, run_dir / config.submission_file)
        board.save(run_dir / "experiment_board.json")

    total_min = (time.time() - start) / 60
    logger.info("=" * 60)
    logger.info("DONE [board_replan]  total=%.1f min  best_score=%s  run_dir=%s", total_min, best_score, run_dir)
    logger.info("=" * 60)

    return {"run_dir": str(run_dir), "best_score": best_score, "last_summary": last_summary}


# ---------------------------------------------------------------------------
# Strategy: codex_todo
# ---------------------------------------------------------------------------

def _todo_coding_system(todo_str: str, competition: Any, work_dir: Path, submission_name: str) -> str:
    desc = getattr(competition, "description", "")[:12000]
    return f"""You are the coding agent. Work in the Jupyter kernel (folder: {work_dir}).
Data is under ./input/ (symlink to competition data). Write submission to ./{submission_name}.

Competition description (excerpt):
{desc}

## Your Todo List
{todo_str}

Work through the todo items in order. For each item:
1. Execute the necessary code using execute_cell.
2. Report the result (scores, observations) clearly in your output.

You MUST use tools. Available tools: execute_cell, edit_cell, check_submission, read_file.

Rules:
- Use relative paths; data is under input/
- Each execute_cell should be focused on one task
- The kernel state persists: variables from earlier cells are available in later ones
- Prioritize having a valid {submission_name} over complex approaches
- If a cell errors, read the traceback carefully and fix the specific issue
- Always print metrics so the strategist can track progress"""


def _run_codex_todo(config: AgentConfig) -> dict[str, Any]:
    from mlagent.todo_planner import TodoPlanningAgent
    from mlagent.experiment_board import TodoList

    setup_logging()
    _ensure_cuda_visible(config)
    run_dir = _setup_run_dir(config, "todo")
    tracer = PromptTracer(run_dir / "debug_prompts", enabled=bool(config.trace_prompts))
    competition = load_competition(config.competition_id)
    todo = TodoList()

    logger.info("=" * 60)
    logger.info("Strategy: CODEX_TODO")
    logger.info("Loading competition: %s", config.competition_id)
    logger.info(
        "Config: %d round(s), %d step(s)/round, revise every %d steps",
        config.max_rounds, config.max_steps_per_round, config.replan_interval,
    )

    planner = TodoPlanningAgent(config, competition, todo, tracer=tracer)
    exp_log = ExperimentLog(run_dir / "experiment_log.json")

    last_summary: Optional[str] = None
    best_score: Optional[float] = None
    start = time.time()

    agent_dir = run_dir / "agent_0"

    try:
        for rnd in range(1, config.max_rounds + 1):
            if time.time() - start > config.total_time_limit:
                logger.info("Total time limit reached.")
                break

            elapsed_min = (time.time() - start) / 60
            logger.info("=" * 60)
            logger.info("ROUND %d/%d  (elapsed %.1f min, best_score=%s)", rnd, config.max_rounds, elapsed_min, best_score)
            logger.info("-" * 60)

            todo_str = planner.plan(rnd, last_summary)
            logger.info("Todo list (%d items):\n%s", len(todo.items), todo_str)

            agent_dir.mkdir(parents=True, exist_ok=True)
            jupyter = JupyterExecutor(
                work_dir=agent_dir,
                data_dir=competition.data_dir,
                jupyter_cfg=config.jupyter,
            )
            at = PromptTracer(agent_dir / "debug_prompts", enabled=bool(config.trace_prompts))
            coder = CodingAgent(config.coding_llm, tracer=at, agent_idx=0)
            tag = "[Agent 0]"

            llm = ToolCallingLLM(config.coding_llm)
            llm.set_system(_todo_coding_system(todo_str, competition, agent_dir, config.submission_file))
            llm.append_user(
                f"Start working through the todo list. You have {config.max_steps_per_round} steps. "
                f"Focus on completing items in order. Report scores and observations clearly."
            )
            tools = build_tools(config.submission_file)
            max_steps = config.max_steps_per_round
            interval = config.replan_interval

            try:
                for step in range(max_steps):
                    _run_coding_step(llm, tools, coder, jupyter, agent_dir, config.submission_file, step, max_steps, at, tag)

                    if (step + 1) % interval == 0 and (step + 1) < max_steps:
                        logger.info("%s Checkpoint at step %d — requesting progress report...", tag, step + 1)
                        llm.append_user(
                            "Pause. Report which todo items you've completed, any scores/metrics, "
                            "and what you plan to work on next."
                        )
                        mid_summary = llm.chat_no_tools()

                        logger.info("%s Revising todo list...", tag)
                        revised_str = planner.revise(rnd, step + 1, mid_summary)
                        llm.append_user(
                            f"[Todo List Updated]\n{revised_str}\n\n"
                            f"Continue working through the remaining items."
                        )
                        logger.info("%s Todo revised (%d items), continuing...", tag, len(todo.items))

                logger.info("%s All %d steps done, generating summary...", tag, max_steps)
                llm.append_user(
                    "The coding round is over. Provide a concise summary: "
                    "what was tried, what worked, what errors occurred, current metrics, "
                    "and suggestions for the next round."
                )
                summary = llm.chat_no_tools() or "No summary generated."

            finally:
                jupyter.shutdown()

            sub_path = agent_dir / config.submission_file
            nb_path = jupyter.get_notebook_path()
            best_score, grade = _grade_and_track_best(competition, sub_path, nb_path, run_dir, best_score)
            last_summary = format_parallel_summary([summary], [grade], best_score, rnd)

            exp_log.log_round(rnd, todo_str, last_summary, grade, extra={"strategy": "codex_todo", "todo_count": len(todo.items)})
            todo.save(run_dir / "todo_list.json")

            elapsed_min = (time.time() - start) / 60
            logger.info("Round %d done (%.1f min total). Best score: %s", rnd, elapsed_min, best_score)

    finally:
        best_sub = run_dir / "best_submission.csv"
        if best_sub.exists():
            shutil.copy(best_sub, run_dir / config.submission_file)
        todo.save(run_dir / "todo_list.json")

    total_min = (time.time() - start) / 60
    logger.info("=" * 60)
    logger.info("DONE [codex_todo]  total=%.1f min  best_score=%s  run_dir=%s", total_min, best_score, run_dir)
    logger.info("=" * 60)

    return {"run_dir": str(run_dir), "best_score": best_score, "last_summary": last_summary}
