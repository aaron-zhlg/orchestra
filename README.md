# orchestra

A minimal, **dependency-free** orchestrator-worker framework for building
multi-agent systems. The core is pure Python standard library — no SDKs, no
frameworks to learn — and it speaks the OpenAI-style *Chat Completions API*, so
it runs against OpenAI, DeepSeek, vLLM, Ollama, or any compatible endpoint by
changing one env var.

It's **intelligence-driven, not rule-driven**: you hand the lead agent a goal and
let the LLM plan, spawn tool-using subagents, and synthesize — there are no
graphs, nodes, or hardcoded control flow. The design is a plain orchestrator-worker
loop:

- a **main agent (orchestrator)** takes a goal, plans, and delegates;
- **subagents (workers)** are narrow specialists that use tools in a loop;
- the orchestrator runs them **in parallel**, **evaluates** their findings, and
  can **dynamically spawn** more subagents before **synthesizing** the answer.

## Three moving parts

| Piece | File | What it is |
| --- | --- | --- |
| `Conversation` | `orchestra/client.py` | The tool-calling loop every agent runs on (Chat Completions + function tools + streaming). |
| `SubAgent` | `orchestra/subagent.py` | The base class you subclass. Customize a prompt + tools; get a tool loop for free. |
| `Orchestrator` | `orchestra/orchestrator.py` | The main agent: goal → plan → dispatch (parallel) → evaluate → synthesize. |

## Install

```bash
pip install -e .            # editable install of the `orchestra` package
```

Then pick **one** of the two ways to configure the endpoint. A **provider key**
selects the host and a default model automatically (key and endpoint stay bound,
so they can't mismatch):

```bash
export DEEPSEEK_API_KEY=sk-...   # -> https://api.deepseek.com, model deepseek-flash
# or
export OPENAI_API_KEY=sk-...     # -> https://api.openai.com/v1, model gpt-5
```

The default models are **version-less rolling aliases** (`deepseek-flash`, `gpt-5`)
so they don't rot as providers ship new versions; set `ORCHESTRA_MODEL` (or a
per-agent `model=`) to pin an exact snapshot.

Or **bring your own endpoint** (any OpenAI-compatible server — vLLM, Ollama, a
gateway). The generic `ORCHESTRA_API_KEY` carries no provider hint, so it must be
paired with a base URL and model:

```bash
export ORCHESTRA_API_KEY=sk-...
export ORCHESTRA_BASE_URL=https://my-host/v1
export ORCHESTRA_MODEL=my-model
```

`ORCHESTRA_BASE_URL` / `ORCHESTRA_MODEL` also override a provider preset if you
want, e.g. to pin a specific DeepSeek/OpenAI model. If the model/endpoint can't
be determined, the client raises a clear error instead of guessing.

## Define a subagent — just prompt + tools

```python
from orchestra import SubAgent

class CalculatorAgent(SubAgent):
    name = "calculator"
    description = "Evaluates arithmetic precisely. Use for any numeric computation."
    instructions = "You are a precise calculator. Use the tools to compute exactly."

    def create_tools(self):
        def add(a: float, b: float) -> float:
            "Add two numbers."
            return a + b
        return [add]
```

Tool schemas are inferred from type hints and the Google-style docstring — you
just write normal Python functions. For stateful toolsets (e.g. an API client
that tracks which ids it fetched), return an object exposing `as_tools()` and
optionally `sources()`; the base class wires the tools in and reports the sources
as citations automatically. See `examples/medical/` for real ones.

## Register subagents and run a goal

```python
from orchestra import Orchestrator

lead = Orchestrator(
    [CalculatorAgent, ShoutAgent],       # subagent *types* the lead may instantiate
    preamble="You coordinate a helpful assistant. Be concise.",  # injected system prompt
    max_rounds=2,
)
report = lead.run("Compute 12 * 8, then shout the word 'done'.")
print(report.answer)
print(report.sources)
```

The orchestrator decides the complexity, decomposes the goal into subtasks,
routes each to the best subagent type (spinning up several instances of the same
type when needed, or skipping types the goal doesn't need), runs them in
parallel, and may spawn follow-up subagents based on what came back.

## Per-agent LLM endpoint (URL + model)

Every agent — the lead and each subagent — can run on its own URL and model. By
default subagents reuse the lead's client, but any of them can point elsewhere
(e.g. a big model for planning, a cheap/fast one for the workers, or a local
server for some tools):

```python
lead = Orchestrator(
    [CalculatorAgent, ShoutAgent],
    # the lead agent's endpoint (a strong model for planning):
    lead_base_url="https://api.openai.com/v1",
    lead_model="gpt-5",
    lead_api_key="sk-...",              # optional; falls back to env
    # a cheaper/faster endpoint for all the worker subagents:
    subagent_kwargs={
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",
        "api_key": "sk-...",
    },
)
```

To point **one** subagent at its own endpoint while the others keep the shared
client, register it with `.spec(...)` instead of the bare class (mix freely):

```python
lead = Orchestrator(
    [
        CalculatorAgent,                       # uses the lead's shared client
        ShoutAgent.spec(                        # this one gets its own endpoint
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
            model="gpt-5-mini",
        ),
    ],
)
```

`.spec(**kwargs)` builds a fresh instance per task with those kwargs. Equivalently,
hand it a ready client (handy when several subagents should share one non-lead
endpoint): `ShoutAgent.spec(client=ChatClient(api_key=..., base_url=..., model=...))`.

You can also bake the endpoint into the class itself:

```python
class LocalCalculator(SubAgent):
    name = "calculator"
    # ...
    def __init__(self, **kw):
        super().__init__(base_url="http://localhost:11434/v1", model="qwen2.5", **kw)
```

Rules: `model` alone rides on the shared client (sent per request); a per-agent
`base_url`/`api_key` makes that agent build its own client, so its URL is
independent of everyone else. Setting a custom `base_url` **requires** a `model`
(a foreign endpoint's model can't be inferred, and a provider preset's model is
never borrowed for a different host) — otherwise you get a clear error, not a
silent mismatch.

## Design

1. **Main agent with a default system instruction, injectable at startup.**
   `Orchestrator` ships default prompts for every phase (`orchestra/prompts.py`)
   and takes `planner_instructions` / `evaluator_instructions` /
   `synthesizer_instructions` / `citation_instructions` overrides, plus a
   `preamble` that is prepended to all of them — the natural hook for a
   project/persona-level system prompt.
2. **A subagent super class you register and customize.** `SubAgent` is that base
   class. Subclass it, set `name` / `description` / `instructions`, and override
   `create_tools()`. Register subclasses directly with the orchestrator.
3. **Goal decomposition, dynamic subagent creation, long-running tasks.**
   `Orchestrator.run(goal)` runs the full plan → dispatch → evaluate → synthesize
   loop, creating fresh subagent instances on demand and spawning more across
   rounds until the evidence is sufficient (or `max_rounds` is hit).

## Examples

```bash
# Tiny, no-network illustration (still needs an LLM key for reasoning):
python -m examples.simple "Compute 12 * 8, then shout the word 'done'."

# A two-subagent medical research system (PubMed + ClinicalTrials.gov):
python -m examples.medical.run "Do GLP-1 agonists reduce MACE in type 2 diabetes?"

# Run a single subagent on its own:
python -m examples.medical.literature "Recent RCTs on semaglutide for weight loss"
```

## Observability

Pass `log_dir=` to the orchestrator to stream the lead's decisions to
`logs/lead.log` and each subagent instance's full tool-by-tool trajectory to
`logs/<type>_<NNN>.log`.

## Development

```bash
pip install -e ".[dev]"
pytest        # the suite uses a scripted fake client — no network or API key needed
```

## License

MIT — see [LICENSE](LICENSE).
