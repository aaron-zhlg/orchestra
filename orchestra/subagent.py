"""The ``SubAgent`` base class and its result/spec types.

A subagent is a narrow specialist: an LLM autonomously using a small set of tools
in a loop to accomplish one self-contained objective, then returning a condensed,
cited result to the orchestrator.

Writing a new subagent is meant to be trivial — subclass :class:`SubAgent` and
supply four things:

* ``name``        — a short id the orchestrator uses for routing.
* ``description`` — one or two sentences on what this subagent is good for (this
  is what the orchestrator reads when deciding whether/how to use it).
* ``instructions``— the system prompt that steers the subagent's tool loop.
* tools           — override :meth:`SubAgent.create_tools` to return them.

Example::

    from orchestra import SubAgent

    class CalculatorAgent(SubAgent):
        name = "calculator"
        description = "Evaluates arithmetic expressions precisely."
        instructions = "You are a precise calculator. Use the tools to compute."

        def create_tools(self):
            def add(a: float, b: float) -> float:
                "Add two numbers."
                return a + b
            return [add]

Nothing here is tied to a domain: tools are just Python callables, and the prompt
is just text. See :mod:`orchestra.examples` for richer, real-world subagents.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import Any

from orchestra.client import ChatClient, Conversation
from orchestra.prompts import SUBAGENT_INSTRUCTIONS

#: Signature of a per-run trajectory sink: ``(tool_name, arguments_json, result)``.
ToolCallSink = typing.Callable[[str, str, str], None]


@dataclass
class SubAgentResult:
    """The condensed output a subagent returns to the orchestrator."""

    subagent: str
    objective: str
    findings: str
    sources: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def __str__(self) -> str:  # convenient for print()
        return self.findings


@typing.runtime_checkable
class Toolset(typing.Protocol):
    """Optional richer tool container: a bundle of tools that also tracks sources.

    Any object with ``as_tools()`` works; if it also exposes ``sources()`` the
    subagent will report those as the run's citations automatically. This is handy
    for stateful tool bundles (e.g. an API client that records every id it fetched).
    """

    def as_tools(self) -> list[typing.Callable[..., Any]]: ...


class SubAgent:
    """A focused, tool-using worker. Subclass and customize prompt + tools.

    A fresh instance is meant to be created for each objective, so any per-run
    state (conversation history, a stateful toolset) starts clean. The orchestrator
    does exactly this via :class:`SubAgentSpec`.
    """

    #: Short routing id. Override in subclasses.
    name: str = "subagent"
    #: What this subagent is good for; the orchestrator reads this to route work.
    description: str = "A general-purpose autonomous subagent."
    #: Default system prompt. Override in subclasses or pass ``instructions=``.
    instructions: str = SUBAGENT_INSTRUCTIONS

    def __init__(
        self,
        *,
        client: ChatClient | None = None,
        instructions: str | None = None,
        tools: typing.Sequence[typing.Callable[..., Any]] | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tool_rounds: int = 16,
        verbose: bool = False,
        reasoning_effort: str | None = None,
        **conversation_params: Any,
    ):
        # ``instructions`` falls back to the class attribute, which itself falls
        # back to the framework default — so a bare SubAgent still works.
        if instructions is not None:
            self.instructions = instructions
        self._client = client
        self._tools_override = list(tools) if tools is not None else None
        # Per-agent LLM endpoint. ``model`` alone can ride on a shared client
        # (passed per request); a per-agent ``base_url``/``api_key`` needs its own
        # client, built lazily in :meth:`_make_client`.
        self.model = model
        self._base_url = base_url
        self._api_key = api_key
        self.max_tool_rounds = max_tool_rounds
        self.verbose = verbose
        self.reasoning_effort = reasoning_effort
        self._conversation_params = conversation_params

    # -- endpoint ----------------------------------------------------------- #

    def _make_client(self) -> ChatClient:
        """The LLM client this agent talks to.

        Reuses a shared/injected ``client`` unless this agent overrides the
        endpoint itself (``base_url``/``api_key``), in which case it builds its own
        so its URL is independent of the lead or its siblings.
        """
        if self._client is not None and self._base_url is None and self._api_key is None:
            return self._client
        return ChatClient(api_key=self._api_key, base_url=self._base_url, model=self.model)

    # -- customization hooks ----------------------------------------------- #

    def create_tools(self) -> typing.Sequence[typing.Callable[..., Any]] | Toolset:
        """Return this run's tools. Override in subclasses.

        May return either a plain list of callables, or a *toolset* object exposing
        ``as_tools()`` (and, optionally, ``sources()`` to auto-report citations).
        Called once per :meth:`run`, so returning fresh, stateful objects is fine.
        """
        if self._tools_override is not None:
            return self._tools_override
        return []

    def extract_sources(self, toolset: Any) -> list[str]:
        """Return the source ids this run touched (for citations).

        By default this reads ``toolset.sources()`` if present. Override for custom
        source tracking.
        """
        collector = getattr(toolset, "sources", None)
        if callable(collector):
            try:
                return [str(s) for s in collector()]
            except Exception:
                return []
        return []

    # -- execution --------------------------------------------------------- #

    @staticmethod
    def _resolve_callables(toolset: Any) -> list[typing.Callable[..., Any]]:
        as_tools = getattr(toolset, "as_tools", None)
        if callable(as_tools):
            return list(as_tools())
        return list(toolset)

    def run(self, objective: str, on_tool_call: ToolCallSink | None = None) -> SubAgentResult:
        """Execute the tool loop for ``objective`` and return a :class:`SubAgentResult`.

        Args:
            objective: The self-contained task to accomplish.
            on_tool_call: Optional sink invoked as ``(name, arguments, result)`` for
                every tool call, e.g. to stream the trajectory to a log file.
        """
        toolset = self.create_tools()
        callables = self._resolve_callables(toolset)

        recorded: list[dict[str, str]] = []

        def sink(name: str, arguments: str, result: str) -> None:
            recorded.append({"name": name, "arguments": arguments, "result": result})
            if self.verbose:
                preview = result if len(result) <= 500 else result[:500] + " …"
                print(f"\n[{self.name} tool] {name}({arguments})\n     -> {preview}\n", flush=True)
            if on_tool_call is not None:
                on_tool_call(name, arguments, result)

        params = dict(self._conversation_params)
        if self.reasoning_effort:
            params.setdefault("reasoning_effort", self.reasoning_effort)

        conversation = Conversation(
            self._make_client(),
            instructions=self.instructions,
            tools=callables,
            model=self.model,
            max_tool_rounds=self.max_tool_rounds,
            on_tool_call=sink,
            **params,
        )

        try:
            summary = conversation.ask(objective)
        except Exception as exc:  # a failing subagent must not sink the whole run
            return SubAgentResult(self.name, objective, "", tool_calls=recorded, error=str(exc))

        return SubAgentResult(
            subagent=self.name,
            objective=objective,
            findings=summary,
            sources=self.extract_sources(toolset),
            tool_calls=recorded,
        )

    def run_stream(self, objective: str) -> typing.Iterator[str]:
        """Same as :meth:`run` but yields the final answer as streaming text deltas."""
        toolset = self.create_tools()
        params = dict(self._conversation_params)
        if self.reasoning_effort:
            params.setdefault("reasoning_effort", self.reasoning_effort)
        conversation = Conversation(
            self._make_client(),
            instructions=self.instructions,
            tools=self._resolve_callables(toolset),
            model=self.model,
            max_tool_rounds=self.max_tool_rounds,
            **params,
        )
        yield from conversation.ask_stream(objective)

    # -- registration helpers ---------------------------------------------- #

    @classmethod
    def spec(cls, **kwargs: Any) -> "SubAgentSpec":
        """Build a :class:`SubAgentSpec` that instantiates this subclass per task.

        ``kwargs`` are forwarded to the constructor of every instance the
        orchestrator creates (e.g. ``model=``, ``verbose=``).
        """
        return SubAgentSpec(cls.name, cls.description, lambda: cls(**kwargs))


@dataclass
class SubAgentSpec:
    """A *type* of subagent the orchestrator can instantiate on demand.

    The orchestrator sees ``name`` + ``description`` to decide routing, then calls
    :meth:`create` **once per task**. That is what lets it spin up the same type
    several times (one independent instance, with its own context window and fresh
    tool state, per sub-question) or skip a type entirely. A spec is a lightweight
    factory; nothing is instantiated until work is actually assigned.
    """

    name: str
    description: str
    factory: typing.Callable[[], SubAgent]

    def create(self) -> SubAgent:
        return self.factory()


def to_spec(item: "SubAgentSpec | type[SubAgent] | SubAgent", **kwargs: Any) -> SubAgentSpec:
    """Normalize anything registrable into a :class:`SubAgentSpec`.

    Accepts a spec (returned as-is), a :class:`SubAgent` subclass (wrapped so a
    fresh instance is built per task), or a ready instance (reused across tasks —
    convenient, but note it will not get a clean context between tasks).
    """
    if isinstance(item, SubAgentSpec):
        return item
    if isinstance(item, type) and issubclass(item, SubAgent):
        return item.spec(**kwargs)
    if isinstance(item, SubAgent):
        return SubAgentSpec(item.name, item.description, lambda: item)
    raise TypeError(f"cannot register {item!r} as a subagent")
