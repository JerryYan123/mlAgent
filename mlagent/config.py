"""YAML config via OmegaConf."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf


@dataclass
class LLMConfig:
    model_name: str = "gpt-5.1-codex-mini"
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
    cell_timeout: int = 600
    monitor_interval: int = 300
    """Seconds between LLM execution-health checks during a long cell."""
    execution_timeout: int = 7200
    """Max seconds for a single cell (also used as cap in monitor prompts)."""
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
    monitor_llm: LLMConfig = field(default_factory=LLMConfig)
    """Used for Jupyter execution monitoring; defaults same as coding if unset in YAML."""

    jupyter: JupyterConfig = field(default_factory=JupyterConfig)

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
