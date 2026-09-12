"""Run the medical multi-agent system from the command line.

    export ORCHESTRA_API_KEY=sk-...   # or DEEPSEEK_API_KEY / OPENAI_API_KEY
    python -m examples.medical.run "Do GLP-1 agonists reduce MACE in type 2 diabetes?"

This demonstrates the orchestrator taking a goal, decomposing it, dynamically
spinning up the literature and trials subagents (in parallel, possibly several
instances each), evaluating, and synthesizing a cited answer.
"""

from __future__ import annotations

import sys

from orchestra import LLMError, Orchestrator
from examples.medical import ClinicalTrialsAgent, LiteratureAgent

# A project-level system prompt, injected into every phase of the lead agent.
MISSION = """\
You coordinate a medical research assistant. Be rigorous and conservative: weigh
the level of evidence, prefer high-quality sources, and never overstate findings.
"""


def build(**kwargs) -> Orchestrator:
    return Orchestrator(
        [LiteratureAgent, ClinicalTrialsAgent],
        preamble=MISSION,
        **kwargs,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Multi-agent medical research (orchestra example).")
    parser.add_argument("goal", nargs="*", help="Research goal; omit for an interactive session.")
    parser.add_argument("--lead-model", default=None, help="Model for the lead/orchestrator.")
    parser.add_argument("--max-rounds", type=int, default=2, help="Max research rounds.")
    parser.add_argument("--effort", default=None, help="Lead thinking effort: none/low/medium/high/max.")
    parser.add_argument("--no-citations", action="store_true", help="Skip the citation pass.")
    parser.add_argument("--quiet", action="store_true", help="Suppress orchestration logs.")
    parser.add_argument("--log-dir", default=None, help="Write full trajectories here.")
    args = parser.parse_args()

    def answer(goal: str) -> None:
        try:
            report = build(
                lead_model=args.lead_model,
                max_rounds=args.max_rounds,
                add_citations=not args.no_citations,
                verbose=not args.quiet,
                reasoning_effort=args.effort,
                log_dir=args.log_dir,
            ).run(goal)
        except LLMError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return
        print("\n" + "=" * 80)
        print(report.answer)
        print("=" * 80)
        print(f"[complexity={report.complexity}, rounds={report.rounds}, sources={len(report.sources)}]")

    if args.goal:
        answer(" ".join(args.goal))
        return

    print("Multi-agent medical research. Ctrl-C or an empty line to quit.")
    while True:
        try:
            goal = input("\ngoal > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not goal:
            return
        answer(goal)


if __name__ == "__main__":
    main()
