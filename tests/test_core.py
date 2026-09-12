"""Unit tests for the orchestra core, driven by a scripted fake client.

No network or API key required: :class:`FakeClient` mimics the Chat Completions
API and returns canned ``choices`` items, exercising the full plan → dispatch →
evaluate → synthesize loop deterministically.
"""

from __future__ import annotations

import json
import typing

import pytest

from orchestra import (
    Conversation,
    Orchestrator,
    SubAgent,
    Tool,
    message_text,
    resolve_endpoint,
)


# --------------------------------------------------------------------------- #
# Response builders + a scriptable fake client
# --------------------------------------------------------------------------- #


def _assistant(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


def _tool_call(name: str, arguments: dict, call_id: str = "call-1") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


def _system_content(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "system":
            return m.get("content") or ""
    return ""


class FakeClient:
    """A stand-in for :class:`orchestra.ChatClient`.

    Routing is by inspecting the system message (which phase) and ``tools``
    (subagent turns carry tools). Subagent turns emit one tool call, then a final
    message once the tool result is in the history. Every system message seen is
    recorded.
    """

    def __init__(self, plan: dict, evaluations: list[dict], subagent_reply: str = "SUBAGENT DONE"):
        self._plan = plan
        self._evaluations = list(evaluations)
        self._subagent_reply = subagent_reply
        self.instructions_seen: list[str] = []

    def create(self, *, messages, tools=None, tool_choice=None, model=None, stream=False, **extra):
        system = _system_content(messages)
        self.instructions_seen.append(system)

        # Subagent turns are the only ones that carry tools.
        if tools:
            already_called = any(m.get("role") == "tool" for m in messages)
            if not already_called:
                name = tools[0]["function"]["name"]
                return _tool_call(name, {"x": 1})
            return _assistant(self._subagent_reply)

        if "delegation plan" in system or "turn the user's goal" in system:
            return _assistant(json.dumps(self._plan))
        if "received findings" in system:
            nxt = self._evaluations.pop(0) if self._evaluations else {"complete": True, "follow_up": []}
            return _assistant(json.dumps(nxt))
        if "write the final answer" in system:
            return _assistant("FINAL DRAFT [S1]")
        if "citation checker" in system:
            return _assistant("FINAL DRAFT [S1]\n\nReferences\n- S1")
        return _assistant("(unrecognized phase)")


# --------------------------------------------------------------------------- #
# Toy toolset + subagent
# --------------------------------------------------------------------------- #


class CountingTools:
    def __init__(self):
        self.calls = 0

    def ping(self, x: int = 0) -> dict:
        """A trivial tool.

        Args:
            x: any integer.
        """
        self.calls += 1
        return {"pong": x}

    def as_tools(self) -> list[typing.Callable]:
        return [self.ping]

    def sources(self) -> list[str]:
        return ["S1"]


class ToyAgent(SubAgent):
    name = "toy"
    description = "A toy subagent used in tests."
    instructions = "You are a toy agent. Call the tool once, then report done."

    def create_tools(self):
        return CountingTools()


# --------------------------------------------------------------------------- #
# Tool schema inference (Chat Completions shape)
# --------------------------------------------------------------------------- #


def test_tool_schema_from_signature_and_docstring():
    def sample(city: str, unit: str = "c") -> str:
        """Look something up.

        Args:
            city: the city name.
            unit: the unit.
        """
        return city

    spec = Tool.from_function(sample).spec()
    assert spec["type"] == "function"
    fn = spec["function"]
    assert fn["name"] == "sample"
    assert fn["description"] == "Look something up."
    params = fn["parameters"]
    assert params["properties"]["city"] == {"type": "string", "description": "the city name."}
    assert params["required"] == ["city"]  # unit has a default
    assert params["additionalProperties"] is False


def test_invalid_tool_name_rejected():
    def bad():
        return None

    with pytest.raises(ValueError):
        Tool.from_function(bad, name="not a valid name!")


# --------------------------------------------------------------------------- #
# Conversation loop
# --------------------------------------------------------------------------- #


def test_conversation_runs_tool_then_returns_text():
    tools = CountingTools()
    seen: list[tuple[str, str, str]] = []
    client = FakeClient(plan={}, evaluations=[])
    chat = Conversation(
        client,
        instructions="ignored",
        tools=tools.as_tools(),
        on_tool_call=lambda n, a, r: seen.append((n, a, r)),
    )
    reply = chat.ask("do it")
    assert reply == "SUBAGENT DONE"
    assert tools.calls == 1
    assert seen and seen[0][0] == "ping"
    # A tool-result message must have been recorded for the API.
    assert any(m.get("role") == "tool" for m in chat.history)


# --------------------------------------------------------------------------- #
# SubAgent
# --------------------------------------------------------------------------- #


def test_subagent_run_collects_sources_and_tool_calls():
    client = FakeClient(plan={}, evaluations=[])
    agent = ToyAgent(client=client)
    result = agent.run("some objective")
    assert result.ok
    assert result.findings == "SUBAGENT DONE"
    assert result.sources == ["S1"]
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "ping"


def test_subagent_instructions_override():
    agent = ToyAgent(instructions="custom prompt")
    assert agent.instructions == "custom prompt"
    assert ToyAgent(client=FakeClient({}, [])).instructions.startswith("You are a toy agent")


def test_bare_subagent_with_injected_tools():
    def echo(text: str = "hi") -> str:
        """Echo text."""
        return text

    client = FakeClient(plan={}, evaluations=[])
    agent = SubAgent(client=client, tools=[echo], instructions="bare")
    result = agent.run("go")
    assert result.findings == "SUBAGENT DONE"


# --------------------------------------------------------------------------- #
# Orchestrator end to end
# --------------------------------------------------------------------------- #


def test_orchestrator_end_to_end():
    plan = {
        "complexity": "simple",
        "reasoning": "one angle",
        "assignments": [
            {"subagent": "toy", "objective": "do the thing", "output_format": "bullets"}
        ],
    }
    client = FakeClient(plan=plan, evaluations=[{"complete": True, "follow_up": []}])
    lead = Orchestrator([ToyAgent], client=client, verbose=False)
    report = lead.run("achieve the goal")

    assert report.complexity == "simple"
    assert report.rounds == 1
    assert len(report.results) == 1
    assert report.results[0].subagent == "toy"
    assert report.sources == ["S1"]
    assert "FINAL DRAFT" in report.answer
    assert "References" in report.answer  # citation pass ran


def test_orchestrator_dynamic_followup_round():
    plan = {
        "complexity": "moderate",
        "assignments": [{"subagent": "toy", "objective": "first", "output_format": ""}],
    }
    evaluations = [
        {"complete": False, "follow_up": [{"subagent": "toy", "objective": "gap", "output_format": ""}]},
    ]
    client = FakeClient(plan=plan, evaluations=evaluations)
    lead = Orchestrator([ToyAgent], client=client, max_rounds=2, verbose=False)
    report = lead.run("goal")

    assert report.rounds == 2
    assert len(report.results) == 2  # initial + one dynamically spawned


def test_preamble_is_injected_into_phase_instructions():
    plan = {"complexity": "simple", "assignments": [{"subagent": "toy", "objective": "x"}]}
    client = FakeClient(plan=plan, evaluations=[{"complete": True, "follow_up": []}])
    lead = Orchestrator([ToyAgent], client=client, preamble="MISSION-XYZ", verbose=False)
    lead.run("goal")
    lead_phase_instructions = [t for t in client.instructions_seen if "MISSION-XYZ" in t]
    assert lead_phase_instructions, "preamble was not injected into any lead instruction"


def test_orchestrator_requires_a_subagent():
    with pytest.raises(ValueError):
        Orchestrator([])


def test_message_text_helper():
    assert message_text(_assistant("hello")) == "hello"


# --------------------------------------------------------------------------- #
# Endpoint / credential resolution (key stays bound to endpoint)
# --------------------------------------------------------------------------- #


@pytest.fixture
def clean_env(monkeypatch):
    for var in (
        "ORCHESTRA_API_KEY",
        "ORCHESTRA_BASE_URL",
        "ORCHESTRA_MODEL",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_deepseek_key_selects_deepseek_endpoint(clean_env):
    clean_env.setenv("DEEPSEEK_API_KEY", "dk")
    key, base, model = resolve_endpoint()
    assert (key, base, model) == ("dk", "https://api.deepseek.com", "deepseek-flash")


def test_openai_key_selects_openai_endpoint(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "ok")
    key, base, model = resolve_endpoint()
    assert base == "https://api.openai.com/v1"
    assert model == "gpt-5"
    assert key == "ok"


def test_generic_key_without_endpoint_is_an_error(clean_env):
    clean_env.setenv("ORCHESTRA_API_KEY", "gk")
    with pytest.raises(ValueError):
        resolve_endpoint()


def test_generic_key_with_endpoint_and_model_ok(clean_env):
    clean_env.setenv("ORCHESTRA_API_KEY", "gk")
    clean_env.setenv("ORCHESTRA_BASE_URL", "https://my-host/v1")
    clean_env.setenv("ORCHESTRA_MODEL", "my-model")
    assert resolve_endpoint() == ("gk", "https://my-host/v1", "my-model")


def test_env_model_overrides_provider_preset(clean_env):
    clean_env.setenv("DEEPSEEK_API_KEY", "dk")
    clean_env.setenv("ORCHESTRA_MODEL", "custom-pinned-model")
    _, base, model = resolve_endpoint()
    assert base == "https://api.deepseek.com"  # preset base kept
    assert model == "custom-pinned-model"  # env model wins


def test_explicit_args_win(clean_env):
    key, base, model = resolve_endpoint(api_key="x", base_url="https://h/v1", model="m")
    assert (key, base, model) == ("x", "https://h/v1", "m")


def test_custom_base_url_without_model_is_an_error_even_with_provider_key(clean_env):
    # A provider key is present, but the base_url points at a *different* host, so
    # the preset's model must NOT be borrowed — this must fail, not silently mix.
    clean_env.setenv("DEEPSEEK_API_KEY", "dk")
    with pytest.raises(ValueError, match="no model"):
        resolve_endpoint(base_url="https://my-vllm:8000/v1")


def test_custom_base_url_with_model_is_ok(clean_env):
    clean_env.setenv("DEEPSEEK_API_KEY", "dk")  # used only for auth
    key, base, model = resolve_endpoint(base_url="https://my-vllm:8000/v1", model="qwen2.5")
    assert base == "https://my-vllm:8000/v1"
    assert model == "qwen2.5"
    assert key == "dk"


def test_no_key_anywhere_is_an_error(clean_env):
    with pytest.raises(ValueError):
        resolve_endpoint()


# --------------------------------------------------------------------------- #
# Per-agent endpoint (url + model) — lead and subagents independently
# --------------------------------------------------------------------------- #


def test_subagent_reuses_shared_client_without_override():
    fc = FakeClient(plan={}, evaluations=[])
    agent = SubAgent(client=fc, instructions="x")
    assert agent._make_client() is fc


def test_subagent_builds_own_client_when_endpoint_overridden(clean_env):
    fc = FakeClient(plan={}, evaluations=[])
    agent = SubAgent(
        client=fc,  # shared client is ignored because this agent sets its own URL
        api_key="k",
        base_url="https://sub-host/v1",
        model="sub-model",
        instructions="x",
    )
    own = agent._make_client()
    assert own is not fc
    assert own.base_url == "https://sub-host/v1"
    assert own.model == "sub-model"


def test_lead_can_set_its_own_url_and_model(clean_env):
    lead = Orchestrator(
        [ToyAgent],
        lead_api_key="k",
        lead_base_url="https://lead-host/v1",
        lead_model="lead-model",
        verbose=False,
    )
    assert lead.client.base_url == "https://lead-host/v1"
    assert lead.client.model == "lead-model"


def test_subagents_can_get_a_different_endpoint_than_lead(clean_env):
    lead = Orchestrator(
        [ToyAgent],
        lead_api_key="k",
        lead_base_url="https://lead-host/v1",
        lead_model="lead-model",
        subagent_kwargs={"api_key": "k2", "base_url": "https://sub-host/v1", "model": "sub-model"},
        verbose=False,
    )
    sub = lead.specs["toy"].create()
    sub_client = sub._make_client()
    assert sub_client.base_url == "https://sub-host/v1"
    assert sub_client.model == "sub-model"
    assert lead.client.base_url == "https://lead-host/v1"
