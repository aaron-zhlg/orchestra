"""orchestra — a minimal, dependency-free orchestrator-worker multi-agent framework.

Three moving parts:

* :class:`~orchestra.client.Conversation` — the tool-calling loop every agent runs on.
* :class:`~orchestra.subagent.SubAgent`   — subclass it, customize prompt + tools.
* :class:`~orchestra.orchestrator.Orchestrator` — the main agent: it takes a goal,
  plans, dynamically spins up subagents in parallel, evaluates, and synthesizes.

Quick start::

    from orchestra import Orchestrator, SubAgent

    class EchoAgent(SubAgent):
        name = "echo"
        description = "Repeats things back, loudly."
        instructions = "You are an echo. Use the tool to shout the input."

        def create_tools(self):
            def shout(text: str) -> str:
                "Return the text in upper case."
                return text.upper()
            return [shout]

    report = Orchestrator([EchoAgent]).run("say hello")
    print(report.answer)

The core (client + subagent + orchestrator) uses only the standard library.
"""

from orchestra.client import (
    ChatClient,
    Conversation,
    LLMError,
    Tool,
    message_text,
    message_tool_calls,
    resolve_endpoint,
    tool,
)
from orchestra.orchestrator import (
    Assignment,
    Orchestrator,
    OrchestratorReport,
    orchestrate,
)
from orchestra.subagent import (
    SubAgent,
    SubAgentResult,
    SubAgentSpec,
    Toolset,
    to_spec,
)

__version__ = "0.1.0"

__all__ = [
    # client / loop
    "Conversation",
    "ChatClient",
    "Tool",
    "tool",
    "message_text",
    "message_tool_calls",
    "resolve_endpoint",
    "LLMError",
    # subagents
    "SubAgent",
    "SubAgentResult",
    "SubAgentSpec",
    "Toolset",
    "to_spec",
    # orchestrator
    "Orchestrator",
    "OrchestratorReport",
    "Assignment",
    "orchestrate",
]
