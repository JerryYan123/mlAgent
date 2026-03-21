"""Textual preview of competition data (for planning context)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_CODE_EXTS = {".py", ".sh", ".yaml", ".yml", ".md", ".html", ".xml", ".rst"}
_TEXT_EXTS = {".txt", ".csv", ".json", ".tsv"} | _CODE_EXTS
_MAX_LEN = 6000


def _file_tree(path: Path, depth: int = 0) -> str:
    files = sorted(p for p in path.iterdir() if not p.is_dir())
    dirs = sorted(p for p in path.iterdir() if p.is_dir())
    lines: list[str] = []
    indent = "  " * depth
    max_show = 4 if len(files) > 30 else 8
    for f in files[:max_show]:
        if f.suffix in _TEXT_EXTS:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                size = f"{sum(1 for _ in fh)} lines"
        else:
            size = f"{f.stat().st_size} bytes"
        lines.append(f"{indent}{f.name} ({size})")
    if len(files) > max_show:
        lines.append(f"{indent}... and {len(files) - max_show} more files")
    for d in dirs:
        lines.append(f"{indent}{d.name}/")
        lines.append(_file_tree(d, depth + 1))
    return "\n".join(lines)


def _preview_csv(path: Path, name: str) -> str:
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    cols_str = ", ".join(cols[:15])
    if len(cols) > 15:
        cols_str += f"... and {len(cols) - 15} more"
    return f"-> {name} has {df.shape[0]} rows and {df.shape[1]} columns.\n  Columns: {cols_str}"


def generate_preview(base_path: str | Path) -> str:
    """Generate a textual preview of a data directory."""
    base = Path(base_path)
    parts = [f"```\n{_file_tree(base)}\n```"]
    for f in sorted(base.rglob("*")):
        if f.is_dir():
            continue
        name = str(f.relative_to(base))
        if f.suffix == ".csv":
            parts.append(_preview_csv(f, name))
        elif f.suffix == ".json":
            try:
                with open(f, encoding="utf-8") as fh:
                    data = json.load(fh)
                schema_hint = type(data).__name__
                if isinstance(data, list) and data:
                    if isinstance(data[0], dict):
                        first_keys = list(data[0].keys())
                    else:
                        first_keys = type(data[0]).__name__
                    schema_hint = f"list[{len(data)}], first keys: {first_keys}"
                elif isinstance(data, dict):
                    keys = list(data.keys())[:10]
                    schema_hint = f"dict with keys: {keys}"
                parts.append(f"-> {name}: {schema_hint}")
            except Exception:
                pass
        elif f.suffix in _TEXT_EXTS and f.stat().st_size < 2000:
            content = f.read_text(encoding="utf-8", errors="ignore")
            parts.append(f"-> {name}:\n{content}")

    result = "\n\n".join(parts)
    if len(result) > _MAX_LEN:
        result = result[:_MAX_LEN] + "\n... (truncated)"
    return result
