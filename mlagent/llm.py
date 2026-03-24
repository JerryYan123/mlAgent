"""LiteLLM-based client with multi-tool OpenAI-style calling."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any

from litellm import completion

from mlagent.config import LLMConfig


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMStepResult:
    """One model turn."""

    content: str | None
    tool_calls: list[ToolCall]
    raw_message: Any


class ToolCallingLLM:
    """Conversation + tools via LiteLLM (same path for Codex and other models)."""

    def __init__(self, config: LLMConfig, **overrides: Any) -> None:
        self.config = deepcopy(config)
        for k, v in overrides.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        native = self.config.native_function_calling
        if native is None:
            native = True
        self.use_native_fc = bool(native)
        self.messages: list[dict[str, Any]] = []
        self._call = partial(
            completion,
            model=self.config.model_name,
            api_key=self.config.api_key or os.environ.get("OPENAI_API_KEY"),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=self.config.timeout,
        )

    def reset(self) -> None:
        self.messages = []

    def set_system(self, text: str) -> None:
        self.messages = [{"role": "system", "content": text}]

    def append_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def append_assistant(self, content: str | None, tool_calls: Any) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def append_tool_result(self, tool_call_id: str, content: str) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )

    def complete_with_tools(
        self,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMStepResult:
        if not self.use_native_fc:
            raise RuntimeError(
                "Tool calling is disabled: set coding_llm.native_function_calling to true in config."
            )
        for attempt in range(self.config.max_retries):
            try:
                resp = self._call(
                    messages=self.messages,
                    tools=tools,
                    tool_choice=tool_choice,
                )
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                continue
            choice = resp.choices[0]
            msg = choice.message
            tcs: list[ToolCall] = []
            raw_tcs = getattr(msg, "tool_calls", None) or []
            for tc in raw_tcs:
                if isinstance(tc, dict):
                    tid = tc.get("id", "")
                    fn = tc.get("function") or {}
                    name = fn.get("name", "")
                    args_raw = fn.get("arguments", "{}")
                else:
                    tid = getattr(tc, "id", "")
                    fn = getattr(tc, "function", None)
                    name = getattr(fn, "name", "") if fn else ""
                    args_raw = getattr(fn, "arguments", "{}") if fn else "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {}
                tcs.append(
                    ToolCall(
                        id=tid,
                        name=name,
                        arguments=args if isinstance(args, dict) else {},
                    )
                )
            return LLMStepResult(
                content=msg.content,
                tool_calls=tcs,
                raw_message=msg,
            )
        raise RuntimeError("complete_with_tools failed after retries")

    def chat_no_tools(self) -> str:
        """Plain text completion without tool definitions (for summarization)."""
        for attempt in range(self.config.max_retries):
            try:
                resp = self._call(messages=self.messages)
            except Exception:
                if attempt == self.config.max_retries - 1:
                    return ""
                continue
            content = resp.choices[0].message.content or ""
            if content:
                self.messages.append({"role": "assistant", "content": content})
                return content
        return ""


class PlanningLLM:
    """Simple chat with history for planning (no tools)."""

    def __init__(self, config: LLMConfig, **overrides: Any) -> None:
        self.config = deepcopy(config)
        for k, v in overrides.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.messages: list[dict[str, Any]] = []
        self._call = partial(
            completion,
            model=self.config.model_name,
            api_key=self.config.api_key or os.environ.get("OPENAI_API_KEY"),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=self.config.timeout,
        )

    def append_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def chat(self) -> str:
        for _ in range(self.config.max_retries):
            resp = self._call(messages=self.messages)
            content = resp.choices[0].message.content or ""
            if content:
                self.messages.append({"role": "assistant", "content": content})
                return content
        return ""
