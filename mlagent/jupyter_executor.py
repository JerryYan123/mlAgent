"""Persistent Jupyter kernel execution (adapted from comind-new jupyter_session)."""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nbformat
from jupyter_client import BlockingKernelClient, KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import litellm
import psutil

from mlagent.config import JupyterConfig, LLMConfig
from mlagent.utils import extract_xml_tag, process_backspace_chars

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    llm_terminated: bool
    timeout: bool
    output: str
    error: str
    execution_time: float


def _monitor_llm_decision(
    monitor_cfg: LLMConfig,
    code: str,
    goal: str,
    output: str,
    run_time: float,
    execution_timeout: int,
) -> tuple[bool, str]:
    """Return (should_continue, explanation)."""
    lines = output.splitlines()
    if len(lines) > 500:
        output = "\n".join(lines[:250] + ["... (truncated) ..."] + lines[-250:])

    remaining = max(0.0, execution_timeout - run_time)
    user = f"""Code:
```python
{code}
```
Goal: {goal}
Runtime: {run_time:.1f}s / {execution_timeout}s max. Remaining: {remaining:.1f}s.

Output:
```
{process_backspace_chars(output)}
```

Respond with:
<action>CONTINUE</action> or <action>STOP</action>
<explanation>Brief reason.</explanation>
"""
    messages = [
        {
            "role": "system",
            "content": "You monitor long-running ML code. Say CONTINUE if training looks healthy, STOP if stuck/broken or cannot finish in time.",
        },
        {"role": "user", "content": user},
    ]
    for _ in range(monitor_cfg.max_retries):
        try:
            resp = litellm.completion(
                model=monitor_cfg.model_name,
                messages=messages,
                api_key=monitor_cfg.api_key or os.environ.get("OPENAI_API_KEY"),
                max_tokens=monitor_cfg.max_tokens,
                temperature=monitor_cfg.temperature,
                timeout=monitor_cfg.timeout,
            )
            text = resp.choices[0].message.content or ""
            action = extract_xml_tag(text, "action") or ""
            expl = extract_xml_tag(text, "explanation") or text
            should_continue = "continue" in action.lower() and "stop" not in action.lower()
            if "stop" in action.lower():
                should_continue = False
            return should_continue, expl
        except Exception as e:
            logger.warning("monitor llm error: %s", e)
    return True, "monitor failed; default continue"


class JupyterExecutor:
    def __init__(
        self,
        work_dir: str | Path,
        data_dir: str | Path,
        jupyter_cfg: JupyterConfig,
        monitor_llm: LLMConfig,
        env_name: str | None = None,
    ) -> None:
        self.work_dir = Path(work_dir).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.cfg = jupyter_cfg
        self.monitor_llm = monitor_llm
        self.work_dir.mkdir(parents=True, exist_ok=True)

        inp = self.work_dir / "input"
        if not inp.exists():
            try:
                inp.symlink_to(self.data_dir, target_is_directory=True)
            except OSError:
                import shutil

                shutil.copytree(self.data_dir, inp)

        self.notebook_path = self.work_dir / "experiment.ipynb"
        self.nb = nbformat.v4.new_notebook()
        nbformat.write(self.nb, self.notebook_path)

        self.selected_gpu_ids, self.selected_cpu_cores = self._select_optimal_resources()
        kernel_name = env_name or hashlib.md5(str(self.work_dir).encode()).hexdigest()[:12]
        self._kernel_spec_name = self._install_kernel_spec(kernel_name)

        self.km = KernelManager(kernel_name=self._kernel_spec_name)
        self.km.start_kernel(cwd=str(self.work_dir))
        self.kc: BlockingKernelClient = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=120)
        self._closed = False
        atexit.register(self._safe_shutdown)

    def _select_optimal_resources(self) -> tuple[list[int], list[int]]:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        available_gpus: list[int] = []
        if cuda_visible:
            try:
                available_gpus = [int(x.strip()) for x in cuda_visible.split(",") if x.strip().isdigit()]
            except ValueError:
                pass

        gpu_usage = self._gpu_memory_usage()
        selected_gpu_ids: list[int] = []
        if gpu_usage:
            if available_gpus:
                avail = [(g, u) for g, u in gpu_usage if g in available_gpus]
            else:
                avail = gpu_usage
            if avail:
                avail.sort(key=lambda x: x[1])
                selected_gpu_ids = [g for g, _ in avail[: self.cfg.max_gpu_count]]

        cpu_usage = self._cpu_per_core()
        cpu_usage.sort(key=lambda x: x[1])
        selected_cores = [c for c, _ in cpu_usage[: self.cfg.max_cpu_cores]]
        logger.info("Jupyter kernel GPUs %s cores %s", selected_gpu_ids, selected_cores)
        return selected_gpu_ids, selected_cores

    def _gpu_memory_usage(self) -> list[tuple[int, float]]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            if result.returncode != 0:
                return []
            out = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                gid = int(parts[0])
                used, total = float(parts[1]), float(parts[2])
                pct = (used / total) * 100 if total else 0.0
                out.append((gid, pct))
            return out
        except Exception:
            return []

    def _cpu_per_core(self) -> list[tuple[int, float]]:
        try:
            pct = psutil.cpu_percent(interval=0.5, percpu=True)
            return [(i, u) for i, u in enumerate(pct)]
        except Exception:
            return [(0, 0.0)]

    def _install_kernel_spec(self, kname: str) -> str:
        ksm = KernelSpecManager()
        kernel_env: dict[str, str] = {}
        if self.selected_gpu_ids:
            kernel_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.selected_gpu_ids))
        else:
            kernel_env["CUDA_VISIBLE_DEVICES"] = ""

        kernel_argv = ["python", "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        if self.selected_cpu_cores and os.name == "posix":
            try:
                subprocess.run(["taskset", "--version"], capture_output=True, check=True)
                mask = ",".join(map(str, self.selected_cpu_cores))
                kernel_argv = ["taskset", "-c", mask] + kernel_argv
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            spec: dict[str, Any] = {
                "argv": kernel_argv,
                "display_name": kname,
                "language": "python",
            }
            if kernel_env:
                spec["env"] = kernel_env
            (td_path / "kernel.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
            ksm.install_kernel_spec(str(td_path), kernel_name=kname, user=True, replace=True)
        return kname

    def _safe_shutdown(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self.kc.stop_channels()
        except Exception:
            pass
        try:
            self.km.shutdown_kernel(now=False)
        except Exception:
            try:
                self.km.shutdown_kernel(now=True)
            except Exception:
                pass

    def shutdown(self) -> None:
        self._safe_shutdown()

    def restart_kernel(self) -> None:
        self._safe_shutdown()
        self._closed = False
        self.km = KernelManager(kernel_name=self._kernel_spec_name)
        self.km.start_kernel(cwd=str(self.work_dir))
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=120)

    def get_notebook_path(self) -> Path:
        return self.notebook_path

    def _save_nb(self) -> None:
        nbformat.write(self.nb, self.notebook_path)

    def _collect_iopub(
        self,
        msg_id: str,
        code: str,
        goal: str,
        cell_timeout: int,
    ) -> tuple[ExecutionResult, list[dict[str, Any]]]:
        stdout_buf: list[str] = []
        stderr_buf: list[str] = []
        nb_outputs: list[dict[str, Any]] = []
        error_text = ""
        success, llm_term, timed_out = True, False, False
        start = time.time()
        last_check = start
        exec_cap = min(self.cfg.execution_timeout, cell_timeout * 20)

        while True:
            now = time.time()
            elapsed = now - start
            if now - last_check >= self.cfg.monitor_interval:
                last_check = now
                out_so_far = "".join(stdout_buf) + "".join(stderr_buf)
                cont, expl = _monitor_llm_decision(
                    self.monitor_llm,
                    code,
                    goal,
                    out_so_far,
                    elapsed,
                    exec_cap,
                )
                if not cont:
                    llm_term = True
                    error_text = expl
                    success = False
                    break

            if elapsed > cell_timeout:
                timed_out = True
                success = False
                break

            try:
                msg = self.kc.get_iopub_msg(timeout=10)
            except Exception:
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            mtype = msg.get("msg_type")
            content = msg.get("content", {})

            if mtype == "status":
                if content.get("execution_state") == "idle":
                    break
            elif mtype == "stream":
                name = content.get("name", "stdout")
                text = content.get("text", "")
                if name == "stderr":
                    stderr_buf.append(text)
                else:
                    stdout_buf.append(text)
                nb_outputs.append(
                    {"output_type": "stream", "name": name, "text": text}
                )
            elif mtype == "error":
                success = False
                tb_lines = content.get("traceback", [])
                error_text = "\n".join(tb_lines)
                nb_outputs.append(
                    {
                        "output_type": "error",
                        "ename": content.get("ename", "Error"),
                        "evalue": content.get("evalue", ""),
                        "traceback": tb_lines,
                    }
                )

        if timed_out or llm_term:
            try:
                self.km.interrupt_kernel()
            except Exception:
                pass

        out = process_backspace_chars("".join(stdout_buf))
        err = process_backspace_chars("".join(stderr_buf) + ("\n" + error_text if error_text else ""))
        if err and not out:
            out = err
        elif err:
            out = out + "\n" + err

        et = time.time() - start
        ok = success and not timed_out and not llm_term
        return (
            ExecutionResult(
                success=ok,
                llm_terminated=llm_term,
                timeout=timed_out,
                output=out,
                error=error_text if llm_term else (err if not ok else ""),
                execution_time=et,
            ),
            nb_outputs,
        )

    def execute_cell(self, code: str, goal: str, timeout: int | None = None) -> ExecutionResult:
        cell_timeout = timeout if timeout is not None else self.cfg.cell_timeout
        msg_id = self.kc.execute(code, allow_stdin=False, store_history=True, stop_on_error=True)
        result, nb_outputs = self._collect_iopub(msg_id, code, goal, cell_timeout)
        cell = nbformat.v4.new_code_cell(source=code)
        cell.outputs = nb_outputs
        cell.execution_count = len(self.nb.cells) + 1
        self.nb.cells.append(cell)
        self._save_nb()
        return result
