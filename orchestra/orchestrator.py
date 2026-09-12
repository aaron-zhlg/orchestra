"""The orchestrator (lead agent): plan → dispatch → evaluate → synthesize.

An orchestrator-worker architecture, generalized to any domain and built on the
tool-calling loop in :mod:`orchestra.client`. It is intelligence-driven: the LLM
decides the decomposition, routing, and when to stop — there is no static graph
or hardcoded control flow, only safety caps (``max_rounds`` / ``max_parallel``).

The :class:`Orchestrator` is the *main agent*. It:

1. **plans** — decomposes a goal into subtasks and routes each to a subagent type,
2. **dispatches** — creates a fresh subagent instance per task and runs them in
   parallel, each with its own context window and tools,
3. **evaluates** — inspects the returned findings and may *dynamically spawn* more
   subagents to close a newly discovered gap, and
4. **synthesizes** — writes the final answer (optionally followed by a citation pass).

Every phase's instructions are injectable at startup (with sensible defaults in
:mod:`orchestra.prompts`), and a ``preamble`` lets you prepend a project- or
persona-level system prompt to all of them.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import sys
import threading
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orchestra.client import ChatClient, message_text
from orchestra.prompts import (
    CITATION_INSTRUCTIONS,
    EVALUATOR_INSTRUCTIONS,
    PLANNER_INSTRUCTIONS,
    SYNTHESIZER_INSTRUCTIONS,
)
from orchestra.subagent import SubAgent, SubAgentResult, SubAgentSpec, ToolCallSink, to_spec


# --------------------------------------------------------------------------- #
# Plan / report data
# --------------------------------------------------------------------------- #


@dataclass
class Assignment:
    """One subtask the lead hands to a specific subagent.

    An assignment carries not just an objective but also an explicit
    ``output_format`` describing what the worker should return, which reduces
    misinterpretation and duplicated work.
    """

    subagent: str
    objective: str
    output_format: str = ""

    def task_prompt(self) -> str:
        """The full instruction handed to the worker (objective + output format)."""
        if self.output_format:
            return f"{self.objective}\n\nRequired output format:\n{self.output_format}"
        return self.objective


@dataclass
class OrchestratorReport:
    """The end product of an :class:`Orchestrator` run."""

    goal: str
    answer: str
    complexity: str = "unknown"
    rounds: int = 0
    results: list[SubAgentResult] = field(default_factory=list)

    @property
    def sources(self) -> list[str]:
        seen: list[str] = []
        for r in self.results:
            for s in r.sources:
                if s not in seen:
                    seen.append(s)
        return seen

    def __str__(self) -> str:
        return self.answer


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ts() -> str:
    """A compact local timestamp prefix for log lines."""
    return _dt.datetime.now().strftime("%H:%M:%S")


def _extract_json(text: str) -> Any:
    """Best-effort parse of a JSON object/array embedded in model output."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"could not parse JSON from model output: {text[:200]!r}")


def _render_findings(results: list[SubAgentResult]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        header = f"### Finding {i} — subagent: {r.subagent}\nObjective: {r.objective}"
        if r.error:
            blocks.append(f"{header}\n[FAILED: {r.error}]")
        else:
            src = ", ".join(r.sources) if r.sources else "(none)"
            blocks.append(f"{header}\nSources: {src}\n{r.findings}")
    return "\n\n".join(blocks) if blocks else "(no findings)"


# --------------------------------------------------------------------------- #
# The orchestrator
# --------------------------------------------------------------------------- #


class Orchestrator:
    """Main agent implementing the plan → dispatch → evaluate → synthesize loop.

    The lead autonomously decides which subagent *types* to instantiate, how many
    of each, and when to stop. After each wave of parallel subagents it inspects
    the collected findings and may dynamically spawn more subagents to pursue a
    newly discovered sub-goal — repeating until it judges the evidence sufficient
    or the ``max_rounds`` safety cap is hit.

    All four phase prompts are injectable. ``preamble`` is prepended to each of
    them, which is the natural place to inject a project/persona system prompt at
    startup (e.g. "You coordinate a legal-research assistant. Be conservative...").
    """

    def __init__(
        self,
        subagents: typing.Sequence["SubAgentSpec | type[SubAgent] | SubAgent"] | None = None,
        *,
        client: ChatClient | None = None,
        lead_model: str | None = None,
        lead_base_url: str | None = None,
        lead_api_key: str | None = None,
        max_rounds: int = 3,
        max_parallel: int = 5,
        add_citations: bool = True,
        verbose: bool = True,
        reasoning_effort: str | None = None,
        preamble: str = "",
        planner_instructions: str = PLANNER_INSTRUCTIONS,
        evaluator_instructions: str = EVALUATOR_INSTRUCTIONS,
        synthesizer_instructions: str = SYNTHESIZER_INSTRUCTIONS,
        citation_instructions: str = CITATION_INSTRUCTIONS,
        subagent_kwargs: dict[str, Any] | None = None,
        log_dir: str | Path | None = None,
    ):
        # The lead's own LLM endpoint. ``lead_base_url``/``lead_api_key`` build a
        # dedicated client so the lead can run on a different URL than the
        # subagents; otherwise use the injected ``client`` (or the env default).
        if lead_base_url is not None or lead_api_key is not None:
            self.client = ChatClient(api_key=lead_api_key, base_url=lead_base_url, model=lead_model)
        else:
            self.client = client or ChatClient()

        # Registry of subagent *types* (specs). The lead instantiates them on
        # demand, so nothing here is a live agent until work is assigned. By
        # default subagents reuse the lead's HTTP client (each still gets its own
        # fresh Conversation + tools). To give subagents a different endpoint,
        # pass ``subagent_kwargs={"base_url": ..., "model": ..., "api_key": ...}``
        # (applies to all), or register per-type instances/specs with their own.
        shared = dict(subagent_kwargs or {})
        shared.setdefault("client", self.client)
        self.specs: dict[str, SubAgentSpec] = {}
        for item in subagents or []:
            spec = to_spec(item, **shared)
            self.specs[spec.name] = spec
        if not self.specs:
            raise ValueError("at least one subagent is required")
        self.lead_model = lead_model
        self.max_rounds = max_rounds
        self.max_parallel = max_parallel
        self.add_citations = add_citations
        self.verbose = verbose
        self.reasoning_effort = reasoning_effort

        self.preamble = preamble.strip()
        self.planner_instructions = planner_instructions
        self.evaluator_instructions = evaluator_instructions
        self.synthesizer_instructions = synthesizer_instructions
        self.citation_instructions = citation_instructions

        # When set, the lead writes logs/lead.log and each subagent instance
        # streams its full trajectory to logs/<type>_<NNN>.log, where NNN is a
        # per-type instance counter (the lead may init several of one type).
        self.log_dir = Path(log_dir) if log_dir else None
        self._lead_log_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._instance_counts: dict[str, int] = {}
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    # -- prompt assembly --------------------------------------------------- #

    def _with_preamble(self, instructions: str) -> str:
        if self.preamble:
            return f"{self.preamble}\n\n{instructions}"
        return instructions

    # -- observability ----------------------------------------------------- #

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr, flush=True)
        if self.log_dir:
            with self._lead_log_lock:
                with (self.log_dir / "lead.log").open("a", encoding="utf-8") as fh:
                    fh.write(f"{_ts()} {message}\n")

    # -- lead-agent LLM calls ---------------------------------------------- #

    @property
    def _roster(self) -> str:
        return "\n".join(f"- {spec.name}: {spec.description}" for spec in self.specs.values())

    def _think(self, instructions: str, prompt: str) -> str:
        params: dict[str, Any] = {}
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        messages = [
            {"role": "system", "content": self._with_preamble(instructions)},
            {"role": "user", "content": prompt},
        ]
        response = self.client.create(messages=messages, model=self.lead_model, **params)
        return message_text(response)

    def _plan(self, goal: str) -> tuple[str, list[Assignment]]:
        instructions = self.planner_instructions.format(roster=self._roster)
        raw = self._think(instructions, f"Goal:\n{goal}")
        data = _extract_json(raw)
        complexity = str(data.get("complexity", "unknown"))
        assignments = self._coerce_assignments(data.get("assignments", []))
        if not assignments:  # never leave the lead with nothing to do
            assignments = [Assignment(next(iter(self.specs)), goal)]
        return complexity, assignments

    def _evaluate(self, goal: str, results: list[SubAgentResult]) -> list[Assignment]:
        instructions = self.evaluator_instructions.format(roster=self._roster)
        prompt = f"Goal:\n{goal}\n\nFindings so far:\n{_render_findings(results)}"
        try:
            data = _extract_json(self._think(instructions, prompt))
        except ValueError:
            return []  # unparseable => treat as complete
        if data.get("complete", True):
            return []
        return self._coerce_assignments(data.get("follow_up", []))

    def _coerce_assignments(self, items: Any) -> list[Assignment]:
        out: list[Assignment] = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("subagent", "")).strip()
            objective = str(item.get("objective", "")).strip()
            output_format = str(item.get("output_format", "")).strip()
            if not objective:
                continue
            if name not in self.specs:  # route unknown names to the first spec
                name = next(iter(self.specs))
            out.append(Assignment(name, objective, output_format))
        return out[: self.max_parallel]

    def _synthesize(self, goal: str, results: list[SubAgentResult]) -> str:
        prompt = f"User goal:\n{goal}\n\nSubagent findings:\n{_render_findings(results)}"
        return self._think(self.synthesizer_instructions, prompt)

    def _cite(self, goal: str, draft: str, results: list[SubAgentResult]) -> str:
        allowed = sorted({s for r in results for s in r.sources})
        prompt = (
            f"User goal:\n{goal}\n\nAllowed sources (only these may be cited):\n"
            + ("\n".join(allowed) if allowed else "(none)")
            + f"\n\nDraft answer:\n{draft}"
        )
        return self._think(self.citation_instructions, prompt)

    # -- dispatch (parallel workers) --------------------------------------- #

    def _trajectory_sink(
        self, round_idx: int, a: Assignment
    ) -> tuple[ToolCallSink | None, typing.Callable[[SubAgentResult], None]]:
        """Return (per-tool-call sink, finalizer) that stream one subagent
        instance's full trajectory to ``logs/<type>_<NNN>.log``, where NNN counts
        instances of that type across the whole run. No-ops without ``log_dir``."""
        if not self.log_dir:
            return None, lambda result: None

        with self._counter_lock:
            self._instance_counts[a.subagent] = self._instance_counts.get(a.subagent, 0) + 1
            instance_no = self._instance_counts[a.subagent]

        path = self.log_dir / f"{a.subagent}_{instance_no:03d}.log"
        fh = path.open("w", encoding="utf-8")
        fh.write(f"{_ts()} === subagent: {a.subagent} #{instance_no:03d} (round {round_idx}) ===\n")
        fh.write(f"{_ts()} OBJECTIVE:\n{a.objective}\n\n")
        fh.flush()
        lock = threading.Lock()
        step = {"n": 0}

        def sink(name: str, arguments: str, result: str) -> None:
            with lock:
                step["n"] += 1
                fh.write(f"{_ts()} [tool #{step['n']}] {name}({arguments})\n")
                fh.write(f"{_ts()}   -> {result}\n\n")
                fh.flush()

        def closer(result: SubAgentResult) -> None:
            with lock:
                if result.error:
                    fh.write(f"{_ts()} [FAILED] {result.error}\n")
                else:
                    fh.write(f"{_ts()} [DONE] {step['n']} tool call(s), {len(result.sources)} source(s)\n")
                    fh.write(f"{_ts()} SOURCES: {', '.join(result.sources) or '(none)'}\n\n")
                    fh.write(f"{_ts()} FINDINGS:\n{result.findings}\n")
                fh.close()

        return sink, closer

    def _dispatch(self, assignments: list[Assignment], round_idx: int = 1) -> list[SubAgentResult]:
        results: list[SubAgentResult] = [None] * len(assignments)  # type: ignore[list-item]

        def work(index: int, a: Assignment) -> tuple[int, SubAgentResult]:
            # Instantiate a fresh worker for this task (this is the lead "init"-ing
            # a subagent on demand — same type can be spun up many times).
            worker = self.specs[a.subagent].create()
            self._log(f"   → init [{a.subagent}] {a.objective}")
            sink, closer = self._trajectory_sink(round_idx, a)
            try:
                result = worker.run(a.task_prompt(), on_tool_call=sink)
            except Exception as exc:  # a failing worker must not sink the run
                result = SubAgentResult(a.subagent, a.objective, "", error=str(exc))
            closer(result)
            return index, result

        workers = min(self.max_parallel, len(assignments)) or 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, i, a) for i, a in enumerate(assignments)]
            for fut in as_completed(futures):
                index, result = fut.result()
                results[index] = result
                status = "ok" if result.ok else f"ERROR: {result.error}"
                self._log(f"   ← [{result.subagent}] done ({len(result.sources)} sources, {status})")
        return results

    # -- public API -------------------------------------------------------- #

    def _reset_logs(self) -> None:
        """Clear historical logs so each run starts a clean trajectory."""
        with self._counter_lock:
            self._instance_counts.clear()
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            for old in self.log_dir.glob("*.log"):
                try:
                    old.unlink()
                except OSError:
                    pass

    def run(self, goal: str) -> OrchestratorReport:
        """Run the full multi-agent loop for ``goal`` and return the final report."""
        self._reset_logs()
        self._log(f"\n[lead] planning: {goal}")
        complexity, assignments = self._plan(goal)
        self._log(f"[lead] complexity={complexity}; {len(assignments)} initial task(s)")

        all_results: list[SubAgentResult] = []
        rounds = 0
        for rnd in range(1, self.max_rounds + 1):
            rounds = rnd
            self._log(f"[lead] round {rnd}: dispatching {len(assignments)} subagent(s) in parallel")
            all_results.extend(self._dispatch(assignments, rnd))

            if rnd >= self.max_rounds:
                self._log(f"[lead] reached max_rounds={self.max_rounds}; proceeding to synthesis")
                break
            follow_up = self._evaluate(goal, all_results)
            if not follow_up:
                self._log("[lead] evaluation: evidence sufficient, proceeding to synthesis")
                break
            spawned = ", ".join(a.subagent for a in follow_up)
            self._log(f"[lead] evaluation: dynamically spawning {len(follow_up)} subagent(s): {spawned}")
            assignments = follow_up

        self._log("[lead] synthesizing final answer")
        draft = self._synthesize(goal, all_results)
        answer = self._cite(goal, draft, all_results) if self.add_citations else draft

        return OrchestratorReport(
            goal=goal, answer=answer, complexity=complexity, rounds=rounds, results=all_results
        )


def orchestrate(goal: str, subagents: typing.Sequence[Any], **kwargs: Any) -> OrchestratorReport:
    """One-shot convenience wrapper around :class:`Orchestrator`."""
    return Orchestrator(subagents, **kwargs).run(goal)
