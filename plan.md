# shadowtrace: design and roadmap

This document explains how shadowtrace is designed and why, and tracks what is
built and what is left. For installation and usage, see the
[README](README.md). For evaluation, monitoring and SLO internals, see
[docs/observability-and-evals.md](docs/observability-and-evals.md).

Section and item IDs (§3.3, M3.6, D1, R4, E10, ...) are stable. Code comments
refer to them, so keep them when editing.

**Status labels:** **Done** means it's built and tested. **Partial** means the
core works but something listed is missing. **Not started** means just that.

---

## 0. Goals and key decisions

### Goals

- Tell a single user which of their LLM workloads a cheaper or local model can
  handle, using evidence from their own traffic rather than public benchmarks.
- Turn that evidence into a routing change they can review, apply and undo.
- Keep watching afterwards: performance over time, workload drift and pipeline
  health.

### Non-goals

- Multi-user or hosted deployment. shadowtrace is a local, single-user tool.
- Routing requests itself. It writes routing decisions for other tools to
  enforce.
- General-purpose benchmarking on curated datasets.

### Decisions

| # | Decision | Why |
|---|---|---|
| D1 | Claude Code transcripts are the primary capture source; the proxy is secondary | Subscription (Pro/Max) logins can't be routed through a custom base URL reliably, but Claude Code writes complete session transcripts to `~/.claude/projects/`. Reading those captures the main tool with no proxy and no risk to login. |
| D1.3 | Savings are measured in quota headroom on a subscription, and in dollars only on API keys | A subscription is flat-rate, so the useful number is how much of the usage limit a routing change frees up. |
| D1.4 | Replay uses a separate metered API key | Subscription credentials can't be used for programmatic calls. `--budget` caps spend on that key. |
| D2 | FastAPI backend with a React + Vite + Tailwind dashboard | One process serves the API, `/metrics` and the UI. |
| D3 | Local models (Ollama, vLLM) are first-class replay candidates | They cost nothing, so they're worth trying before any paid model. |
| D4 | A small custom proxy instead of an off-the-shelf gateway | The proxy must be byte-transparent and fail open; a few hundred lines are easier to guarantee than a general gateway. |
| D5 | Local embeddings for clustering | Prompts never leave the machine for clustering. A dependency-free embedder is the default; sentence-transformers is optional. |

---

## 1. Architecture

```
 ~/.claude/projects/*.jsonl ──▶ transcript watcher ─┐
 API clients ──▶ proxy :8787 ──▶ upstream provider  │ (copy, redacted)
                      └───────────────────────────┐ │
                                                  ▼ ▼
                                     SQLite capture store (WAL, append-only)
                                                  │  sync every 5 min
                                                  ▼
                                      DuckDB analytics store
        ┌──────────────┬─────────────┬────────────┼─────────────┬──────────────┐
     cluster        replay +       recommend     apply /     monitor + SLOs   dashboard
   (archetypes)   grade + SPRT                  revert      (drift, quality,   :8788 +
                                                             performance)     /metrics
```

**Storage.** Capture writes go to SQLite in WAL mode, because many small
concurrent writes on the hot path suit it. Analysis runs on DuckDB, populated
by an idempotent sync. Keeping them apart means a slow dashboard query can
never block capture (R2). Both stores live under `~/.shadowtrace/` and carry
versioned schemas with in-place migrations (`db/migrations.py`).

**Trace rows** keep the full request and response JSON, including system
prompts and `tool_use`/`tool_result` blocks, not flattened text. Graders and
chain replay need that structure (R1). Main tables:

| Table | Holds |
|---|---|
| `traces` | Captured calls: model, request/response JSON, tokens, cost (API lane only), session id, redaction flags, quarantine flag |
| `archetypes`, `archetype_assignments` | Task clusters with stable ids, and trace membership |
| `replay_results`, `eval_scores` | Candidate answers, verdicts, cost and latency, and one row per scorer |
| `sprt_verdicts` | Pass, fail or inconclusive per archetype × candidate per run |
| `recommendations`, `applied_policies` | Recommendations with expiry, and every applied change with the previous file content |
| `spot_checks`, `human_reviews`, `slo_evaluations` | Online checks, human labels for judge verdicts, and stored SLO evaluations |

**Code layout** mirrors the pipeline: `ingest/`, `proxy/`, `mining/`,
`replay/` (with `graders/`), `recommend/` (with `apply/`), `monitor/`,
`dashboard/`, `db/` and `common/`. See the README's project layout.

---

## 2. Roadmap

### M0: Skeleton

| Item | Status | Notes |
|---|---|---|
| Repo, CI (ruff, mypy strict, pytest, 85% coverage gate) | Done | `.github/workflows/ci.yml` |
| `shadow up` / `shadow pause` | Done | `up` also runs sync, the alert loop and the dashboard |

### M1: Capture

| Item | Status | Notes |
|---|---|---|
| M1.1 Transcript watcher | Done | Polls every 2s, reads incrementally, read-only, skips and counts malformed lines. `shadow ingest` backfills history; `--reset` re-imports after a parser fix |
| M1.2 Streaming pass-through proxy (Anthropic + OpenAI-compatible) | Done | Byte-identical streaming; capture is assembled asynchronously |
| M1.3 Fail-open behaviour | Done | Capture errors never reach the user; JSONL spill when SQLite is unavailable |
| M1.4 Redaction | Partial | Regex pack and entropy quarantine work. Quarantined rows are **not encrypted** yet |
| M1.5 Cost and quota estimation | Partial | Dollar cost from a price table for the API lane. Quota headroom is a flat tokens ÷ quota-period estimate, not modelled on published limits |
| M1.6 OTel receiver for Claude Code telemetry | Partial | `ingest/otel_sink.py` exists, but no command starts it, so there's no token-count cross-check |
| M1.7 claude.ai export importer | Partial | Idempotent importer exists; no CLI command |
| M1.8 SQLite → DuckDB sync | Done | Periodic in `shadow up`, on demand with `shadow sync` |

### M2: Task mining and dashboard v1

| Item | Status | Notes |
|---|---|---|
| M2.1 Embeddings | Done | Hashing embedder by default; sentence-transformers optional and not yet verified in CI |
| M2.2 Clustering and labels | Partial | HDBSCAN optional (its test runs only when the `ml` extra is installed, which CI doesn't do), built-in fallback otherwise. Labels are keyword-based; the LLM labelling hook isn't wired to any command |
| M2.3 Archetype rename / merge / pin | Partial | Library functions work and survive re-clustering; no CLI command or API endpoint |
| M2.4 Dashboard v1 | Partial | Overview, tool split, model mix, archetype list, trace drill-down. Missing: usage over time, archetype treemap |

### M3: Shadow replay

| Item | Status | Notes |
|---|---|---|
| M3.1 Stratified sampler | Done | Per archetype, most recent first, excludes quarantined traces, deterministic per seed |
| M3.2 Candidate ladder | Done | Ollama, vLLM, Anthropic, OpenRouter; user override at `~/.shadowtrace/ladder.yaml`. Candidates run cheapest-first per sample, not "all local before any cloud" |
| M3.3 Budget-capped runner | Partial | Hard cap, checkpoints, resume without double-spend, backoff, judge cost counted. Missing: a cost preview before spending |
| M3.4 Graders | Done | Format, completion, trajectory, grounding (advisory), LLM judge. Code checks use Python's parser, not tree-sitter; diffs are checked for shape, not applied |
| M3.5 SPRT | Done | α = β = 0.05, p0 = floor (default 0.90), p1 = floor − 0.10 |
| M3.6 Chain replay | Done | Top-2 archetypes by volume, candidate outputs carried forward, per-chain cap (R6) |

### M4: Recommend, apply, watch

| Item | Status | Notes |
|---|---|---|
| M4.1 Objective-aware engine | Done | Quota headroom vs dollars per D1.3; latest verdict per pair only; idempotent |
| M4.2 Recommendation details | Partial | Pass rate with Wilson 95% interval, saving, 45-day expiry. Latency delta is always empty |
| M4.3 Apply writers | Done | LiteLLM, NanoClaw-style, Claude Code; diff → confirm → write → record; `shadow revert`. None of the target tools reads these mappings natively, so enforcing them needs a hook or wrapper |
| M4.4 Regression watch | Partial | Revert-on-breach logic and the week-over-week degradation alert exist. Nothing schedules spot-check replays of live traffic, and the dashboard has no revert button |
| M4.5 Human review queue | Done | `shadow review`: agreement, Cohen's kappa, false-pass rate; feeds an SLO and an alert |
| M4.6 Dashboard v2 | Partial | Recommendations tab and Health tab (SLOs, alerts, drift, quality, performance, evals). Missing: savings/quota tracker, replay leaderboard view, apply and revert buttons |

### Beyond the original milestones

Built while closing the gaps above:

- Instrumentation helper (`@instrument`), Prometheus `/metrics`,
  OpenTelemetry GenAI span attributes, process-wide tracer registration.
- Drift (PSI and JS divergence), data-quality checks and week-over-week
  performance (`monitor/`).
- Declarative SLOs and alerts with error budgets and desktop notifications
  (`monitor/slo.py`, `monitor/slo_rules.yaml`).
- Replay sends the real conversation, system prompt and tools (inferred from
  the frontier's calls for transcript traces).

### What's next

In rough priority order:

1. **Scheduled spot-checks** for applied policies, wired to the existing
   revert-on-breach logic, plus a dashboard revert button (finishes M4.4).
2. **A smoke test against real providers** (one cheap Haiku call, one Ollama
   call) to run before releases. Every provider is currently tested only
   against mocks.
3. **Encryption at rest** for quarantined rows and the stores (M1.4, §3.5).
4. **CLI commands for archetype rename/merge/pin and the claude.ai importer**
   (M2.3, M1.7).
5. **Start the OTel receiver** in `shadow up` and add a capture-completeness
   SLO (M1.6, §4.5).
6. **Cost preview** before `shadow run` spends (M3.3).
7. **Dashboard:** usage over time, treemap, savings tracker, leaderboard
   (M2.4, M4.6).
8. **Smaller items:** latency delta on recommendations (M4.2), LLM cluster
   labelling (M2.2), per-component log files with rotation (§4.5), a `LICENSE`
   file.

---

## 3. Testing

Tests never call real providers. Upstreams are a FastAPI fake
(`tests/fakes/anthropic_server.py`) or `respx` mocks.

### 3.1 Unit

Redaction (true secrets scrubbed, high-entropy code quarantined rather than
scrubbed, property tests), SPRT (exact boundaries plus 4,000-trial
simulations of both error rates), sampler, ladder pricing, capture batching and spill,
sync idempotency, cluster stability, recommender currency and expiry, scorers,
judge, human review, drift, quality, performance and SLO maths.

### 3.2 Integration

Proxy against the fake upstream (byte-identical streaming, writer killed
mid-stream, upstream errors passed through, tool-use structure intact), the
transcript watcher on files being appended, the replay runner (budget cap,
resume, backoff), the CLI end to end, and the dashboard API.

### 3.3 End-to-end scenarios

Implemented as targeted pytest suites against fakes and real local mechanisms
(real SQLite locks, real sockets, a real local HTTP server), not a
docker-compose harness.

| # | Scenario | Status |
|---|---|---|
| E1 | Golden path: transcripts → capture → sync → cluster → replay → SPRT → recommend → apply | Done |
| E2 | Live session tail, incremental, multi-session | Done |
| E3 | Malformed transcript lines skipped and counted | Done |
| E4 | Proxy golden path, streaming and tool use | Done |
| E5 | Failure injection: writer killed, SQLite locked, spill directory unwritable | Done |
| E6 | No seeded secret appears in any store or log; quarantined rows excluded from replay | Done |
| E7 | Budget cap, checkpoint, resume without double-spend | Done |
| E8 | SPRT verdicts match known pass rates | Done |
| E9 | Chain replay carries candidate outputs forward and respects its sub-cap | Done |
| E10 | Regression below floor → revert restores config byte for byte | Done (logic); not scheduled, see M4.4 |
| E11 | claude.ai export re-import is idempotent | Done |
| E12 | Cold start with an empty database | Done |
| E13 | Egress containment: only allowed hosts reachable | Done |
| E14 | Migration on a populated database keeps data intact | Done |

Not set up: a nightly full-matrix run, and a live smoke test before releases.

### 3.4 Performance

Proxy overhead budget: p50 < 5 ms, p99 < 15 ms (`tests/perf/`, runs in CI).
The 50 requests/s capture-burst check is not written yet.

### 3.5 Security and privacy

Done: the secret-leak audit (E6), quarantine exclusion and redacted logs. Not
started: encryption at rest.

### 3.6 Test data

304 synthetic traces across 8 archetypes in `tests/fixtures/traces/`. They
include tool-use chains, non-English, long-context and near-duplicate cases,
and can be regenerated with `tests/fixtures/generate_traces.py`.

---

## 4. Non-functional requirements

- **Fail open.** shadowtrace must never break or slow the user's own work. Any
  violation is the highest-priority bug.
- **One directory of state.** Everything lives under `~/.shadowtrace/`. To back
  up, copy it; to reset, delete it.
- **No unexpected network calls.** Outbound traffic goes only to proxied
  upstreams, configured replay candidates, the judge (if a key is set) and an
  OTLP endpoint (if set). This is enforced by the E13 test.

### 4.5 Observability

shadowtrace instruments itself. Full details are in
[docs/observability-and-evals.md](docs/observability-and-evals.md).

| Area | Status |
|---|---|
| Metrics (Prometheus naming, `/metrics`) | Done |
| Tracing (local JSONL, optional OTLP, GenAI attributes; replay, judge, ingest and proxy traced) | Done |
| Logs: structured JSON, redacted | Partial: written to stderr; per-component files with rotation exist but aren't enabled |
| `/healthz`, `/readyz` (store, writable state dir, disk headroom) | Done |
| Alert rules and desktop notifications | Done |
| SLOs: capture lag, proxy overhead, sync freshness, judge agreement | Done |
| SLO: capture completeness against Claude Code telemetry | Not started (needs M1.6) |

---

## 5. Risks and mitigations

| # | Risk | Mitigation | Status |
|---|---|---|---|
| R1 | Flattening responses to text loses tool calls and system prompts | Store full request/response JSON | Done |
| R2 | DuckDB is a poor fit for concurrent hot-path writes | SQLite on the hot path, DuckDB via sync | Done |
| R3 | Entropy-based redaction flags large amounts of code, gutting the dataset | Quarantine instead of delete; exclude from replay | Partial: not encrypted |
| R4 | The judge shares biases with the frontier baseline | Human review queue with agreement and false-pass alerts; judge model configurable | Done |
| R5 | Subscription logins may bypass a custom base URL, so the proxy sees nothing | Read Claude Code transcripts directly (D1) | Done |
| R6 | Chain replay cost can explode as candidate outputs change later calls | Top-2 archetypes only, per-chain spend cap | Done |
| R7 | Recommendations go stale as prices and models change | 45-day expiry plus a stale-policy alert | Done |
| R8 | Re-importing a claude.ai export duplicates traces | Idempotent upsert keyed on message id | Done (importer not in CLI yet) |
| R9 | Claude Code's transcript format is undocumented and may change | Defensive parser, `unknown_model_ratio` data-quality alert, `shadow ingest --reset` to re-import after a fix | Done |
| R10 | Routing mappings are written but not enforced by the target tools | Documented; enforcement needs a hook or wrapper | Open |
