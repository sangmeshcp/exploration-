# shadowtrace

**Find out which of your LLM calls a cheaper model could handle, prove it on
your own traffic, and route accordingly.**

shadowtrace runs on your machine. It records the LLM traffic you already
generate, mainly Claude Code sessions plus anything that calls the Anthropic or
OpenAI API. It groups that traffic into recurring kinds of task, then re-runs
samples of each kind against cheaper or local models with a strict budget. It
grades the answers against what your frontier model actually said, and
recommends a routing change only when the evidence is statistically
significant. It also tracks how each model performs over time and alerts you
when one degrades or your workload shifts.

```
 your Claude Code sessions ──┐
 API scripts via the proxy ──┼─▶ capture ─▶ cluster ─▶ shadow replay ─▶ grade ─▶ recommend ─▶ apply
                             │   (redacted)  (task      (budget-capped,   (5 scorers   (SPRT     (router
                             │               types)      local first)     + judge)     verdicts)  config)
                             └──────────────────────── monitor: drift · data quality · SLOs · alerts
```

> **Status: alpha.** The full pipeline works and has extensive automated tests
> (237 tests, ~93% coverage). Calls to real model providers have only been
> tested against mocked endpoints so far. See [Limitations](#limitations).

## Contents

- [Why](#why)
- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Command reference](#command-reference)
- [Configuration](#configuration)
- [Dashboard, metrics and alerts](#dashboard-metrics-and-alerts)
- [Privacy and security](#privacy-and-security)
- [Development](#development)
- [Project layout](#project-layout)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

## Why

Most people send everything to their most capable model, because checking
whether a cheaper one would do is tedious and generic benchmarks don't predict
how a model handles *your* work. shadowtrace answers that question with your
own data:

- **On a subscription** (Claude Pro/Max), savings show up as *quota headroom*:
  "send these two kinds of task to Haiku or a local model and free up ~N% of
  your usage limit."
- **On API keys**, savings show up as dollars per month.

## Features

- **Zero-config capture.** Reads Claude Code's own session transcripts from
  `~/.claude/projects/`, with no proxy and no change to how you log in. An
  optional pass-through proxy covers API-key scripts and other tools.
- **Never gets in your way.** If any capture component fails, your session
  carries on untouched. Writes fall back to a spill file when the database is
  unavailable.
- **Secrets stay out.** API keys, tokens, JWTs and private keys are scrubbed
  before anything is stored. High-entropy strings are quarantined and excluded
  from replay rather than deleted.
- **Task clustering.** Groups prompts into stable "archetypes" (for example
  "Python debugging" or "SQL queries") that keep their identity when you
  re-cluster.
- **Budget-capped shadow replay.** Re-runs sampled traces against a ladder of
  candidate models, cheapest first: free local models (Ollama, vLLM), then
  cloud models (Anthropic, OpenRouter). The run stops at your hard spend cap,
  and an interrupted run resumes without paying twice. Multi-turn sessions can
  be replayed as whole chains.
- **Evaluation built for LLM and agent output.** Five scorers: output format,
  task completion, tool-use trajectory, grounding (a hallucination check) and
  an LLM judge. A human review queue measures how far the judge can be trusted.
- **Statistically sound decisions.** A sequential probability ratio test
  (SPRT) gives each archetype × model pair a pass, fail or inconclusive
  verdict, so you spend only as many samples as the evidence needs.
- **Apply with undo.** Shows a diff, asks for confirmation, then writes the
  archetype → model mapping into a LiteLLM config, a NanoClaw-style rules file
  or a Claude Code settings file. `shadow revert` restores the file byte for
  byte.
- **Monitoring.** Week-over-week performance per model, drift in what you ask
  for, data-quality checks, SLOs with error budgets, and alerts.
- **Standard interfaces.** OpenTelemetry traces (using the GenAI semantic
  conventions), Prometheus `/metrics`, and a JSON API with a React dashboard.

## Installation

Requires Python 3.11+. [uv](https://docs.astral.sh/uv/) is recommended.

```bash
git clone https://github.com/sangmeshcp/exploration-.git shadowtrace
cd shadowtrace
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

Optional extras:

| Extra | Adds | Install |
|---|---|---|
| `judge` | LLM-as-judge via the Anthropic SDK | `uv pip install -e '.[judge]'` |
| `ml` | sentence-transformers embeddings + HDBSCAN clustering (pulls in PyTorch) | `uv pip install -e '.[ml]'` |
| `otel` | OTLP export of traces to your own collector | `uv pip install -e '.[otel]'` |
| `dev` | test, lint and type-check tooling | `uv pip install -e '.[dev]'` |

Without `ml`, shadowtrace uses a lightweight built-in embedder and clustering,
so the base install needs no model downloads.

To build the web dashboard (optional; needs Node.js 18+):

```bash
cd shadowtrace/dashboard/web && npm ci && npm run build
```

## Quick start

```bash
# 1. Import the Claude Code history you already have
shadow ingest

# 2. Group it into task archetypes, and look at what you have
shadow cluster
shadow report

# 3. Replay samples against cheaper models, spending at most $3
export SHADOWTRACE_REPLAY_ANTHROPIC_API_KEY=sk-ant-...   # a metered API key, not your subscription
shadow run --budget 3.00

# 4. See what can safely move to a cheaper model
shadow recommend

# 5. Apply one: shows a diff and asks before writing anything
shadow apply <recommendation-id> --writer litellm --output-path ./litellm-router.yaml

# Changed your mind? Restore the file exactly as it was
shadow revert <policy-id> --output-path ./litellm-router.yaml
```

From then on, keep shadowtrace running in the background:

```bash
shadow up        # capture + periodic sync + alerting + dashboard on http://127.0.0.1:8788
```

With no API key, replay can still run entirely on local models through
[Ollama](https://ollama.com) or vLLM. Grading then falls back to a simple
local heuristic instead of the LLM judge.

## How it works

### 1. Capture

There are two ways traffic gets in. Both write to the same local SQLite store.

- **Claude Code transcripts (primary).** shadowtrace watches
  `~/.claude/projects/**/*.jsonl` read-only and picks up new messages within
  seconds while a session is still running. `shadow ingest` reads your whole
  existing history in one go; it's safe to re-run, because duplicates are
  ignored.

  Because duplicates are skipped, a transcript-parser fix in a newer
  shadowtrace version won't touch traces you already captured.
  `shadow ingest --reset` deletes both local stores and re-reads everything
  from the original transcripts, which Claude Code still has. This also deletes
  everything derived from them, including archetype names, replay results,
  recommendations, applied-policy history, human reviews and SLO history, so
  run `shadow cluster` again afterwards. Your config files and the files
  `shadow apply` changed are not touched.
- **Proxy (optional).** `shadow up` starts a pass-through proxy on
  `127.0.0.1:8787` for `/v1/messages` (Anthropic) and `/v1/chat/completions`
  (OpenAI-compatible). Point a tool at it to capture its traffic:

  ```bash
  export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
  ```

  Streaming responses pass through unmodified. Upstream errors are returned
  as-is and never recorded as successful traces. To group calls into a
  session, send an `x-shadowtrace-session` header.

`shadow pause` stops recording (the proxy keeps forwarding) and `shadow resume`
turns it back on. Data is synced into a DuckDB analytics store every 5 minutes,
or straight away with `shadow sync`.

### 2. Cluster

`shadow cluster` embeds the user-visible text of each prompt and groups similar
prompts into archetypes. Each archetype gets an automatic label. Archetypes are
matched across runs by their most central example, so their ids, and any
recommendations attached to them, survive re-clustering.

### 3. Shadow replay

`shadow run --budget <usd>` samples recent traces from each archetype and
replays them against every candidate in the ladder:

- Candidates receive the original conversation, system prompt and tool
  definitions. For transcripts, which don't record tool definitions, the tools
  are inferred from the calls the original model made.
- Candidates are tried cheapest first for each sample. Local candidates cost
  nothing; cloud candidates are charged against the budget. The run stops
  within one call of the cap and writes a checkpoint after every call. Rate
  limits are retried with backoff.
- With `--chains` (the default), whole multi-turn sessions from the two busiest
  archetypes are also replayed, feeding each candidate's own answers forward.
  Each chain has its own cap (`--chain-budget`).

### 4. Grade

Each candidate answer is compared with the stored answer from your original
model:

| Scorer | Decides pass/fail? | What it checks |
|---|---|---|
| `format` | yes | If the original answer was JSON, a diff or a Python code block, the candidate's must be too, and must parse |
| `completion` | yes | The candidate actually answered: not empty, and not a refusal where the original answered |
| `trajectory` | yes | For tool use: the same tools, in the same order, with the same arguments |
| `grounding` | no (recorded only) | File paths, identifiers, numbers and URLs in the answer should appear in the request or the original answer. A cheap hallucination signal |
| `judge` | yes | An LLM judge (`claude-haiku-5-5` by default) decides whether the candidate is an acceptable substitute. If the judge call fails, the candidate fails |

Judge costs count against `--budget`. Every score is stored, so `shadow evals`
can break results down by archetype, model and scorer.

Judges make mistakes, so `shadow review` shows you a sample of judge-decided
results side by side and asks for your verdict. It reports how often you and
the judge agree, Cohen's kappa, and the judge's *false-pass rate* (answers it
passed that you rejected). An alert fires if that rate goes above 10%.

### 5. Recommend and apply

For every archetype × candidate pair, an SPRT checks whether the candidate's
pass rate is above your floor (`--floor`, default 0.90). `shadow recommend`
turns passing verdicts into recommendations. Each one has a pass rate with a
95% confidence interval, an estimated saving, and an expiry date 45 days out,
because model prices and releases change.

`shadow apply` previews the change as a diff, asks for confirmation, writes the
file and prints a policy id. `shadow revert <policy-id>` puts the file back
exactly as it was. Available writers:

| Writer | Writes |
|---|---|
| `litellm` | `router_settings.routing_rules.<archetype>: <model>` in a [LiteLLM](https://docs.litellm.ai) config YAML |
| `nanoclaw` | An archetype → model rule in a NanoClaw-style complexity-router JSON file |
| `claude_code` | `model_routing.<archetype>: <model>` in a Claude Code settings JSON file |

None of these tools classifies requests into shadowtrace archetypes on its
own. The written mapping is the decision record; acting on it needs a hook,
router plugin or wrapper that recognises the archetype and picks the model.

### 6. Monitor

`shadow monitor` reports:

- **Performance:** each model's pass rate this week against last week, per
  archetype. A significant drop flags the pair as degraded.
- **Drift:** how your task mix, model mix and prompt sizes this week compare
  with the previous four weeks, measured with the Population Stability Index.
  A growing share of traffic outside every archetype means it's time to
  re-cluster.
- **Data quality:** missing token counts, empty responses, unknown models,
  quarantine rate, unclustered traffic and clock skew.

SLOs and alert rules are covered in
[Dashboard, metrics and alerts](#dashboard-metrics-and-alerts).

## Command reference

Run `shadow <command> --help` for every option.

| Command | Does |
|---|---|
| `shadow ingest [--reset]` | Import existing Claude Code transcripts, then sync. `--reset` deletes all local data first and re-imports from scratch |
| `shadow up` | Run everything continuously: transcript watcher, proxy (`:8787`), sync, alert evaluation, dashboard (`:8788`) |
| `shadow pause` / `resume` | Stop / restart recording without stopping the proxy |
| `shadow sync` | Copy newly captured traces into the analytics store now |
| `shadow cluster` | Embed prompts and (re)build archetypes |
| `shadow report [--raw]` | Trace counts, overall and per archetype |
| `shadow run --budget USD` | Budget-capped shadow replay and grading |
| `shadow recommend` | Turn passing verdicts into recommendations (`--access-mode subscription\|api_key`) |
| `shadow apply ID --writer W --output-path P` | Preview, confirm and write a routing change (`--yes` skips the prompt) |
| `shadow revert POLICY_ID --output-path P` | Restore a file to exactly what it was before `apply` |
| `shadow evals` | Results per scorer, plus judge reliability |
| `shadow review` | Label judge verdicts by hand (`--stats` only prints agreement) |
| `shadow monitor [--json]` | Performance, drift and data quality |
| `shadow slo [--notify] [--json]` | Evaluate SLOs and alerts once; exits with status 1 if anything is breached |
| `shadow dashboard [--port]` | Serve the dashboard alone (`up` already does this) |
| `shadow metrics` | Print this process's metrics as JSON |

## Configuration

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SHADOWTRACE_HOME` | `~/.shadowtrace` | Where all state is kept |
| `SHADOWTRACE_CLAUDE_PROJECTS_DIR` | `~/.claude/projects` | Claude Code transcript directory to watch |
| `SHADOWTRACE_REPLAY_ANTHROPIC_API_KEY` | falls back to `ANTHROPIC_API_KEY` | Metered key for Anthropic replay candidates. Subscription logins can't be used for replay |
| `SHADOWTRACE_JUDGE_ANTHROPIC_API_KEY` | falls back to the replay key | Key for the LLM judge (needs the `judge` extra). Without any key, a local heuristic judges |
| `SHADOWTRACE_JUDGE_MODEL` | `claude-haiku-5-5` | Judge model |
| `SHADOWTRACE_OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `SHADOWTRACE_VLLM_URL` | `http://localhost:8000` | vLLM (OpenAI-compatible) server |
| `OPENROUTER_API_KEY` | none | Needed for OpenRouter candidates |
| `SHADOWTRACE_OTLP_ENDPOINT` | none (local file only) | Also send traces to an OTLP/HTTP collector (needs the `otel` extra) |

### Candidate ladder

The models to replay against are defined in
[`shadowtrace/replay/ladder.yaml`](shadowtrace/replay/ladder.yaml). To use
your own list, put a file in the same format at `~/.shadowtrace/ladder.yaml`:

```yaml
candidates:
  - name: ollama-llama3.1-8b     # the name used in recommendations
    provider: ollama             # anthropic | ollama | vllm | openrouter
    model: llama3.1:8b           # the provider's model id (defaults to name)
    local: true                  # local candidates are free and tried first
    price_per_mtok_in: 0.0       # USD per million tokens, for budget tracking
    price_per_mtok_out: 0.0
```

### SLOs and alert rules

Defaults live in
[`shadowtrace/monitor/slo_rules.yaml`](shadowtrace/monitor/slo_rules.yaml).
Copy it to `~/.shadowtrace/slo_rules.yaml` to change thresholds or add rules;
your copy replaces the defaults.

### State directory

Everything lives under `SHADOWTRACE_HOME`. To back it up, copy the directory;
to reset, delete it.

```
~/.shadowtrace/
├── capture.sqlite3     # raw captured traces (write-optimised)
├── analytics.duckdb    # synced copy plus archetypes, replay results, scores, recommendations
├── spill/              # fallback JSONL written when SQLite is unavailable
├── checkpoints/        # resumable replay-run state
├── logs/traces.jsonl   # OpenTelemetry spans
├── PAUSED              # present while recording is paused
├── ladder.yaml         # optional: your candidate list
└── slo_rules.yaml      # optional: your SLOs and alerts
```

## Dashboard, metrics and alerts

`shadow up` serves a web dashboard on <http://127.0.0.1:8788> (after you've
built it; see [Installation](#installation)). It has an overview of your
traffic, archetypes with trace drill-down, recommendations, a **Health** tab
(SLOs, error budgets, firing alerts, drift, data quality, model performance and
eval results) and a system tab.

The same server exposes:

| Endpoint | Content |
|---|---|
| `/metrics` | All metrics in Prometheus text format, ready for Prometheus, Grafana Agent, the Datadog agent or an OpenTelemetry collector |
| `/healthz`, `/readyz` | Liveness, and readiness (store present, state directory writable, at least 2 GB free) |
| `/api/slo`, `/api/monitor`, `/api/evals` | The data behind the Health tab, as JSON |

**Alerts.** `shadow up` evaluates every rule once a minute. Alerts are logged
when they start and when they resolve, rather than on every check. Add
`--notify` for desktop notifications on critical alerts (`notify-send` on
Linux, `osascript` on macOS). Default rules cover capture lag, proxy overhead,
sync freshness, judge reliability, pipeline errors, disk space, model
degradation, drift, archetype coverage and expired recommendations.

**Traces.** Replay runs, judge calls, transcript ingestion and proxied requests
are traced with OpenTelemetry and written to `~/.shadowtrace/logs/traces.jsonl`.
Set `SHADOWTRACE_OTLP_ENDPOINT` to also send them to Jaeger, Grafana Tempo,
Honeycomb, Langfuse, Arize Phoenix or any other OTLP backend. Model-call spans
use the OpenTelemetry GenAI attributes, so LLM-aware tools display them as
model calls.

Some metrics, such as capture lag, proxy latency and fail-open events, exist
only inside the running `shadow up` process. A one-shot `shadow slo` reports
those as `no_data`; the dashboard served by `shadow up` has them.

More detail on the evaluation and monitoring design, including how to add your
own scorers and indicators, is in
[`docs/observability-and-evals.md`](docs/observability-and-evals.md).

## Privacy and security

- **Local by default.** All data stays in `SHADOWTRACE_HOME`. shadowtrace makes
  network calls only to forward proxied requests to their original provider,
  to replay candidates you configured, to the LLM judge if you set a key, and
  to an OTLP endpoint if you set one. A test suite blocks every other outbound
  connection to enforce this.
- **Read-only on Claude Code's files.** The transcript watcher never writes to
  or locks anything under `~/.claude/`.
- **Redaction before storage.** Known secret formats (AWS, GCP, GitHub, Slack,
  OpenAI and Anthropic keys, JWTs, PEM blocks, `password=`-style pairs) are
  replaced before a trace is stored. The same redaction applies to log output.
  Traces with other high-entropy strings are flagged as quarantined and never
  replayed.
- **Replay sends prompts to the candidates you choose.** If a trace contains
  something that shouldn't reach a third-party provider, keep that provider out
  of your ladder or use local models only.
- Not yet implemented: encryption at rest for the capture store.

## Development

```bash
uv pip install -e '.[dev]'

python -m pytest -m "not e2e and not perf"   # unit + integration, with the 85% coverage gate under --cov
python -m pytest -m e2e                      # end-to-end pipeline scenario
python -m pytest -m perf                     # proxy latency budget
ruff check . && ruff format --check .
mypy shadowtrace                             # strict mode
```

Use `python -m pytest` rather than bare `pytest`, so the virtual environment's
pytest is the one that runs.

Tests never touch the real network. Providers are mocked with `respx`, a fake
Anthropic server lives in `tests/fakes/`, and the synthetic trace corpus in
`tests/fixtures/` can be regenerated with `tests/fixtures/generate_traces.py`.
CI (`.github/workflows/ci.yml`) runs all of the above on every push and pull
request.

For dashboard development, run `npm run dev` in `shadowtrace/dashboard/web`.
It proxies API calls to a backend on `:8788` (`shadow up` or
`shadow dashboard`).

## Project layout

```
shadowtrace/
├── cli.py            # the `shadow` command
├── ingest/           # Claude Code transcript watcher, claude.ai export importer, SQLite→DuckDB sync
├── proxy/            # pass-through proxy, capture writer, redaction
├── mining/           # embeddings, clustering, archetype identity
├── replay/           # sampling, candidate ladder, providers, runner, SPRT
│   └── graders/      # scorers, LLM judge, human review
├── recommend/        # recommendation engine and config writers
├── monitor/          # drift, data quality, performance, SLOs and alerts
├── dashboard/        # FastAPI backend and React frontend (web/)
├── db/               # schemas and migrations
└── common/           # config, logging, metrics, tracing, instrumentation
tests/                # unit/, integration/, e2e/, perf/, fakes/, fixtures/
docs/                 # design notes
```

[`plan.md`](plan.md) is the original design document, kept for background on
why things are built the way they are.

## Limitations

- **Real provider calls are untested.** Calls to Anthropic, Ollama, vLLM and
  OpenRouter, and the LLM judge, are only tested against mocked endpoints. File
  an issue if a real provider rejects a request.
- **Claude Code's transcript format isn't a published spec.** The parser is
  defensive: it skips and counts lines it doesn't recognise rather than
  crashing. Format changes may still need parser updates; the
  `unknown_model_ratio` data-quality check is an early warning.
- **No automatic regression watch yet.** The logic to revert an applied
  change when its spot-check pass rate falls below the floor exists
  (`recommend/engine.py`), but nothing schedules spot-check replays or runs it.
  For now, watch `shadow monitor` or the Health tab and use `shadow revert`.
- **No CLI command for the claude.ai export importer** (`ingest/claude_export.py`)
  yet.
- **No check of capture completeness against Claude Code's own telemetry** yet.
- **The grounding scorer is advisory** until it has been validated against
  human review.
- **No encryption at rest.**

## Contributing

Issues and pull requests are welcome. Before opening a pull request, run the
test, lint and type-check commands from [Development](#development); CI runs
the same checks. Include a test for any behaviour change. For anything
touching capture, keep the fail-open guarantee: a shadowtrace failure must
never break or slow the user's own session.

## License

MIT, as declared in [`pyproject.toml`](pyproject.toml).
