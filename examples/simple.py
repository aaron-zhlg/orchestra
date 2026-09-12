"""The smallest possible illustration: two toy subagents under the orchestrator.

Neither subagent needs the network — their tools are pure Python — so this is the
quickest way to see the plan → dispatch → synthesize loop end to end (you still
need an LLM API key for the lead + subagent reasoning).

    export ORCHESTRA_API_KEY=sk-...   # or DEEPSEEK_API_KEY / OPENAI_API_KEY
    python -m examples.simple "What is (12 * 8) and shout the word 'done'?"
"""

from __future__ import annotations

import sys

from orchestra import Orchestrator, SubAgent


class CalculatorAgent(SubAgent):
    """A subagent is: a name, a description, a prompt, and some tools."""

    name = "calculator"
    description = "Evaluates arithmetic precisely. Use for any numeric computation."
    instructions = (
        "You are a precise calculator. Use the tools to compute results exactly, "
        "then report the final number and how you got it."
    )

    def create_tools(self):
        def add(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        def multiply(a: float, b: float) -> float:
            """Multiply two numbers."""
            return a * b

        return [add, multiply]


class ShoutAgent(SubAgent):
    """A second toy subagent, to show routing between multiple types."""

    name = "shouter"
    description = "Transforms text to emphatic upper case. Use for formatting words loudly."
    instructions = "You format text loudly. Use the tool, then return exactly what it produced."

    def create_tools(self):
        def shout(text: str) -> str:
            """Return the text in upper case with exclamation marks."""
            return f"{text.upper()}!!!"

        return [shout]


def main() -> None:
    goal = " ".join(sys.argv[1:]) or "Compute 12 * 8, then shout the word 'done'."
    lead = Orchestrator([CalculatorAgent, ShoutAgent], max_rounds=1)
    report = lead.run(goal)
    print("\n" + "=" * 80)
    print(report.answer)


if __name__ == "__main__":
    main()
