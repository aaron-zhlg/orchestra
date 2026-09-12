"""LLM client + tool-calling conversation loop (Chat Completions compatible).

This is the low-level engine every agent in :mod:`orchestra` runs on. It speaks
the OpenAI-style *Chat Completions API* (``POST {base_url}/chat/completions``),
which is the most widely supported LLM interface — OpenAI, DeepSeek, Together,
Groq, vLLM, Ollama, and most gateways all implement it — so the same code drives
any of them by pointing ``base_url`` at the right host.

Only the Python standard library is used, so this module (and the whole
framework) can be dropped into any project with no dependencies to install.

    # A provider key picks the endpoint + a default (rolling) model for you:
    export DEEPSEEK_API_KEY=sk-...           # -> https://api.deepseek.com, deepseek-flash
    export OPENAI_API_KEY=sk-...             # -> https://api.openai.com/v1, gpt-5
    # Or bring your own endpoint (required with the generic ORCHESTRA_API_KEY):
    export ORCHESTRA_API_KEY=sk-...
    export ORCHESTRA_BASE_URL=https://my-host/v1
    export ORCHESTRA_MODEL=my-model

Usage::

    from orchestra.client import Conversation, ChatClient

    def get_weather(city: str) -> str:
        '''Look up the current weather.

        Args:
            city: Name of the city, e.g. "Hangzhou".
        '''
        return f"{city}: 24C, sunny"

    chat = Conversation(ChatClient(), tools=[get_weather])
    print(chat.ask("What's the weather in Hangzhou?"))
    print(chat.ask("And how about tomorrow?"))
"""

from __future__ import annotations

import inspect
import json
import os
import re
import time
import types
import typing
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

# A provider-specific key both authenticates *and* tells us which endpoint and
# default model to use — key and endpoint are bound together, so you can never end
# up sending, say, an OpenAI key to the DeepSeek host. Order = detection priority.
#
# The default models are deliberately *version-less rolling aliases* (e.g.
# ``deepseek-flash`` always points at the latest Flash), so this table does not go
# stale as providers ship new versions. They are only a convenience fallback: set
# ``ORCHESTRA_MODEL`` / ``model=`` (or a snapshot id) to pin an exact version.
_PROVIDERS: dict[str, tuple[str, str]] = {
    # key env var:        (base_url,                    default model alias)
    "DEEPSEEK_API_KEY": ("https://api.deepseek.com", "deepseek-flash"),
    "OPENAI_API_KEY": ("https://api.openai.com/v1", "gpt-5"),
}
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """An API request failed, or the tool loop could not be completed."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


def resolve_endpoint(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> tuple[str, str, str]:
    """Resolve ``(api_key, base_url, model)`` from arguments and the environment.

    Resolution keeps the key bound to its endpoint so they can never mismatch:

    1. Explicit arguments always win.
    2. A provider-specific key (``DEEPSEEK_API_KEY`` / ``OPENAI_API_KEY``) implies a
       *coherent* ``(base_url, model)`` pair. It fills the base when none is given,
       and fills the model only when the resolved base is that provider's host —
       so a **custom** ``base_url`` never silently borrows another host's model.
    3. Whenever a custom ``base_url`` is set without a ``model``, that is an error:
       we cannot know which model a foreign endpoint serves, so you must set one.
    4. The generic ``ORCHESTRA_API_KEY`` carries no provider hint, so it MUST be
       paired with ``ORCHESTRA_BASE_URL`` + ``ORCHESTRA_MODEL`` (or explicit args).

    Raises ``ValueError`` with an actionable message rather than guessing wrong.
    """
    base = base_url or os.environ.get("ORCHESTRA_BASE_URL")
    model = model or os.environ.get("ORCHESTRA_MODEL")

    provider_key: str | None = None
    preset_base: str | None = None
    preset_model: str | None = None
    for env_name, (pb, pm) in _PROVIDERS.items():
        value = os.environ.get(env_name)
        if value:
            provider_key, preset_base, preset_model = value, pb, pm
            break

    key = api_key or os.environ.get("ORCHESTRA_API_KEY") or provider_key
    if not key:
        raise ValueError(
            "no API key: set a provider key (DEEPSEEK_API_KEY or OPENAI_API_KEY), or "
            "ORCHESTRA_API_KEY together with ORCHESTRA_BASE_URL + ORCHESTRA_MODEL, or "
            "pass api_key= explicitly."
        )

    # Adopt the provider preset only where it stays coherent: take its base if none
    # was chosen, and take its default model ONLY when the base is that same host.
    if base is None:
        base = preset_base
    if model is None and preset_base and base and base.rstrip("/") == preset_base.rstrip("/"):
        model = preset_model

    if not base:
        raise ValueError(
            "no endpoint: use a provider key (DEEPSEEK_API_KEY / OPENAI_API_KEY), set "
            "ORCHESTRA_BASE_URL, or pass base_url= explicitly."
        )
    if not model:
        raise ValueError(
            f"a base_url is set ({base!r}) but no model — a custom endpoint's model "
            "cannot be inferred (and a provider preset's model is never borrowed for a "
            "different host). Set model= (or ORCHESTRA_MODEL) for this endpoint."
        )
    return key, base.rstrip("/"), model


# --------------------------------------------------------------------------- #
# Function tools
# --------------------------------------------------------------------------- #

_SCALAR_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

_ARG_LINE = re.compile(r"^(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
_SECTION = re.compile(r"^(args|arguments|parameters|returns|raises|yields|examples?|notes?)\s*:$", re.I)


def _json_schema_for(annotation: Any) -> dict[str, Any]:
    if annotation is None or annotation is Any or annotation is inspect.Parameter.empty:
        return {}
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Literal:
        return {"enum": list(args)}
    if origin in (typing.Union, types.UnionType):
        variants = [a for a in args if a is not type(None)]
        if len(variants) == 1:
            return _json_schema_for(variants[0])
        return {"anyOf": [_json_schema_for(a) for a in variants]}
    if origin in (list, set, frozenset, tuple):
        return {"type": "array", "items": _json_schema_for(args[0]) if args else {}}
    if origin is dict or annotation is dict:
        return {"type": "object"}
    if annotation in _SCALAR_TYPES:
        return {"type": _SCALAR_TYPES[annotation]}
    return {}


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a docstring into its summary and Google-style ``Args:`` entries."""
    if not doc:
        return "", {}

    summary: list[str] = []
    params: dict[str, str] = {}
    in_args = False
    current = ""

    for line in inspect.cleandoc(doc).splitlines():
        stripped = line.strip()
        section = _SECTION.match(stripped)
        if section:
            in_args = section.group(1).lower() in ("args", "arguments", "parameters")
            current = ""
            continue
        if not in_args:
            summary.append(stripped)
            continue
        if not stripped:
            continue
        match = _ARG_LINE.match(stripped)
        if match:
            current = match.group(1).lstrip("*")
            params[current] = match.group(2).strip()
        elif current:
            params[current] = f"{params[current]} {stripped}".strip()

    return "\n".join(summary).strip(), params


def _schema_from_signature(fn: Callable[..., Any], descriptions: dict[str, str]) -> dict[str, Any]:
    signature = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # unresolvable forward refs should not break tool registration
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        schema = _json_schema_for(hints.get(name))
        if name in descriptions:
            schema = {**schema, "description": descriptions[name]}
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    schema["additionalProperties"] = False
    return schema


@dataclass(frozen=True)
class Tool:
    """A Python callable exposed to the model as a Chat Completions ``function`` tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> Tool:
        if isinstance(fn, Tool):
            return fn
        declared = getattr(fn, "__orchestra_tool__", {})
        summary, arg_docs = _parse_docstring(inspect.getdoc(fn))
        resolved = cls(
            name=name or declared.get("name") or fn.__name__,
            description=description or declared.get("description") or summary,
            parameters=(
                parameters
                or declared.get("parameters")
                or _schema_from_signature(fn, arg_docs)
            ),
            fn=fn,
        )
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", resolved.name):
            raise ValueError(f"invalid tool name {resolved.name!r}: must match ^[a-zA-Z0-9_-]{{1,128}}$")
        return resolved

    def spec(self) -> dict[str, Any]:
        """The Chat Completions tool schema: ``{"type": "function", "function": {...}}``."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def call(self, arguments: str) -> str:
        """Run the tool. Failures come back as text so the model can recover."""
        try:
            kwargs = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return f"ERROR: arguments were not valid JSON ({exc})"
        if not isinstance(kwargs, dict):
            return "ERROR: arguments must be a JSON object"
        try:
            result = self.fn(**kwargs)
        except TypeError as exc:
            return f"ERROR: bad arguments for {self.name} ({exc})"
        except Exception as exc:
            return f"ERROR: {self.name} raised {type(exc).__name__}: {exc}"
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> Any:
    """Attach tool metadata to a function, overriding what is inferred from it.

    The function stays directly callable; pass it to ``Conversation(tools=[...])``.
    """

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        target.__orchestra_tool__ = {  # type: ignore[attr-defined]
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        return target

    return decorate(fn) if fn is not None else decorate


# --------------------------------------------------------------------------- #
# Response helpers
# --------------------------------------------------------------------------- #


def _first_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if not choices:
        return {}
    return choices[0].get("message") or {}


def message_text(response: dict[str, Any]) -> str:
    """The assistant text content of a (non-streamed) chat completion."""
    return _first_message(response).get("content") or ""


def message_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``tool_calls`` array of a (non-streamed) chat completion, if any."""
    return _first_message(response).get("tool_calls") or []


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #


class ChatClient:
    """Thin wrapper over ``POST {base_url}/chat/completions``.

    Credentials, endpoint, and model are resolved together by
    :func:`resolve_endpoint` so a key is always paired with the right host/model.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        self.api_key, self.base_url, self.model = resolve_endpoint(api_key, base_url, model)
        self.timeout = timeout
        self.max_retries = max_retries

    def create(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Iterable[Any] | None = None,
        tool_choice: Any = None,
        model: str | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> Any:
        """Create a chat completion. Returns the response dict, or an event iterator if streaming."""
        if not messages:
            raise ValueError("messages is required")

        payload: dict[str, Any] = {"model": model or self.model, "messages": list(messages)}
        if tools:
            payload["tools"] = [
                t.spec() if isinstance(t, Tool) else t if isinstance(t, dict) else Tool.from_function(t).spec()
                for t in tools
            ]
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        payload.update({k: v for k, v in extra.items() if v is not None})

        if stream:
            payload["stream"] = True
            return self._stream(payload)
        return self._post(payload)

    def _request(self, payload: dict[str, Any], *, stream: bool):
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if stream else "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return urllib.request.urlopen(request, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                if exc.code in RETRY_STATUS and attempt < self.max_retries:
                    last_error = exc
                    time.sleep(2**attempt)
                    continue
                raise _http_error(exc.code, body) from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    last_error = exc
                    time.sleep(2**attempt)
                    continue
                raise LLMError(f"request to {self.base_url} failed: {exc.reason}") from exc
        raise LLMError(f"request failed after {self.max_retries + 1} attempts: {last_error}")

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._request(payload, stream=False) as response:
            body = json.loads(response.read().decode())
        if isinstance(body, dict) and body.get("error"):
            error = body.get("error") or {}
            raise LLMError(error.get("message", "response failed"), code=error.get("code"))
        return body

    def _stream(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        with self._request(payload, stream=True) as response:
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                if isinstance(event, dict) and event.get("error"):
                    error = event.get("error") or {}
                    raise LLMError(error.get("message", "response failed"), code=error.get("code"))
                yield event


def _http_error(status: int, body: str) -> LLMError:
    message, code = body.strip(), None
    try:
        parsed = json.loads(body).get("error") or {}
        message = parsed.get("message", message)
        code = parsed.get("code")
    except (json.JSONDecodeError, AttributeError):
        pass
    return LLMError(f"HTTP {status}: {message}", status=status, code=code)


# --------------------------------------------------------------------------- #
# Multi-turn conversation
# --------------------------------------------------------------------------- #


class Conversation:
    """Multi-turn chat that runs registered tools until the model replies with text.

    The whole ``messages`` list is kept locally and resent on every request. The
    ``instructions`` (if any) are sent as a leading ``system`` message.
    """

    def __init__(
        self,
        client: ChatClient | None = None,
        *,
        instructions: str | None = None,
        tools: Iterable[Any] = (),
        model: str | None = None,
        max_tool_rounds: int = 8,
        on_tool_call: Callable[[str, str, str], None] | None = None,
        **params: Any,
    ):
        self.client = client or ChatClient()
        self.instructions = instructions
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.on_tool_call = on_tool_call
        self.params = params
        self.history: list[dict[str, Any]] = []
        self.last_response: dict[str, Any] | None = None
        self.tools: dict[str, Tool] = {}
        for item in tools:
            self.register(item)

    def register(self, fn: Any, **kwargs: Any) -> Tool:
        resolved = Tool.from_function(fn, **kwargs)
        self.tools[resolved.name] = resolved
        return resolved

    def reset(self) -> None:
        self.history.clear()
        self.last_response = None

    # -- request assembly --------------------------------------------------- #

    def _messages(self) -> list[dict[str, Any]]:
        prefix = [{"role": "system", "content": self.instructions}] if self.instructions else []
        return prefix + self.history

    def _tool_specs(self) -> list[dict[str, Any]] | None:
        return [t.spec() for t in self.tools.values()] or None

    # -- public loop -------------------------------------------------------- #

    def ask(self, message: str | dict[str, Any] | None = None) -> str:
        """Send a user message, resolve any tool calls, and return the reply text."""
        if message is not None:
            self.history.append(_user_item(message))

        for _ in range(self.max_tool_rounds + 1):
            response = self.client.create(
                messages=self._messages(),
                tools=self._tool_specs(),
                model=self.model,
                **self.params,
            )
            self.last_response = response
            assistant = _first_message(response)
            self.history.append(_clean_assistant(assistant))
            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                return assistant.get("content") or ""
            self._run_tool_calls(tool_calls)

        raise LLMError(f"tool loop did not settle within {self.max_tool_rounds} rounds")

    def ask_stream(self, message: str | dict[str, Any] | None = None) -> Iterator[str]:
        """Same as :meth:`ask`, yielding text deltas. Tool rounds are silent."""
        if message is not None:
            self.history.append(_user_item(message))

        for _ in range(self.max_tool_rounds + 1):
            content_parts: list[str] = []
            acc: dict[int, dict[str, str]] = {}
            for event in self.client.create(
                messages=self._messages(),
                tools=self._tool_specs(),
                model=self.model,
                stream=True,
                **self.params,
            ):
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                if text:
                    content_parts.append(text)
                    yield text
                for tc in delta.get("tool_calls") or []:
                    slot = acc.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]

            tool_calls = [
                {
                    "id": slot["id"] or f"call-{idx}",
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": slot["arguments"]},
                }
                for idx, slot in sorted(acc.items())
            ]
            assistant: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts) or None}
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            self.history.append(assistant)
            if not tool_calls:
                return
            self._run_tool_calls(tool_calls)

        raise LLMError(f"tool loop did not settle within {self.max_tool_rounds} rounds")

    def _run_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        """Execute each tool call and append its result as a ``tool`` message."""
        for call in tool_calls:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            arguments = fn.get("arguments", "") or "{}"
            target = self.tools.get(name)
            result = (
                target.call(arguments)
                if target is not None
                else f"ERROR: unknown tool {name!r}; available tools: {', '.join(self.tools) or 'none'}"
            )
            if self.on_tool_call is not None:
                self.on_tool_call(name, arguments, result)
            self.history.append(
                {"role": "tool", "tool_call_id": call.get("id", ""), "content": result}
            )


def _clean_assistant(message: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields we must resend for a valid assistant turn."""
    cleaned: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        cleaned["tool_calls"] = message["tool_calls"]
    return cleaned


def _user_item(message: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    return {"role": "user", "content": message}
