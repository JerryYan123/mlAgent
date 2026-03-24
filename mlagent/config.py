"""YAML config via OmegaConf."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf


@dataclass
class LLMConfig:
    model_name: str = "gpt-5.4-mini"
    api_key: str = ""
    temperature: float = 1.0
    max_tokens: int = 100000
    timeout: int = 600
    max_retries: int = 3
    keep_history: bool = True
    native_function_calling: Optional[bool] = None


@dataclass
class JupyterConfig:
    kernel_name: str = "python3"
    cell_timeout: int = 900
    """Max seconds for a single cell before interrupt (15 min)."""
    monitor_interval: int = 120
    """Seconds between kernel health checks during a long cell."""
    gpu: Optional[str] = "auto"
    max_cpu_cores: int = 8
    max_gpu_count: int = 1


@dataclass
class AgentConfig:
    competition_id: str = "spooky-author-identification"
    working_dir: str = "./workspace"
    submission_file: str = "submission.csv"

    planning_llm: LLMConfig = field(default_factory=LLMConfig)
    coding_llm: LLMConfig = field(default_factory=LLMConfig)

    jupyter: JupyterConfig = field(default_factory=JupyterConfig)

    planning_strategy: str = "baseline"
    replan_interval: int = 5
    num_coding_agents: int = 1
    max_rounds: int = 10
    max_steps_per_round: int = 20
    total_time_limit: int = 14400

    trace_prompts: bool = True
    trace_prompts_dir: Optional[str] = None


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> AgentConfig:
    schema = OmegaConf.structured(AgentConfig)
    file_cfg = OmegaConf.load(config_path)
    merged = OmegaConf.merge(schema, file_cfg)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_object(merged)
