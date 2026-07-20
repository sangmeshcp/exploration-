# shadowtrace

Personal Shadow Tracing & Model Recommender — a local-first, single-user
tool that captures your own LLM traffic, clusters it into task
archetypes, budget-caps a shadow replay against cheaper/local candidate
models, and recommends (and can apply) a router config that reclaims
subscription quota headroom or API spend.

Design doc: [`plan.md`](./plan.md).

## Status

This is a from-scratch implementation of the full `plan.md` — every
module in the architecture (proxy, transcript ingestion, redaction, ETL,
mining, shadow replay + SPRT + chain replay, recommender + regression
watch, actuation writers, CLI, dashboard, OTel tracing, schema
migrations) is real, working code with tests, not a stub. **Read
[Scope and what's simplified](#scope-and-whats-simplified) below** before
assuming any single piece matches the plan's every detail — a handful of
things remain deliberately scoped down (mainly: no literal
docker-compose harness, and the heaviest ML dependency — real
sentence-transformers — isn't installed in this environment).

```
176 tests passing · ruff clean · mypy --strict clean · ~89% coverage
```

## Quick start

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Use `python -m pytest`, not bare `pytest` — on some shells (zsh command-hash
# caching is a common culprit) a bare `pytest` can silently resolve to an
# unrelated pytest from before the venv was activated, missing every dev
# dependency below. `python -m pytest` always uses the active venv's pytest.
python -m pytest -m "not e2e and not perf"   # unit + integration (fast)
python -m pytest -m e2e                      # golden-path end-to-end scenario
python -m pytest -m perf                     # proxy latency budget check

shadow up                    # start the proxy (API-key lane) + transcript
                              # watcher (primary lane) + periodic ETL
shadow pause / shadow resume # toggle the capture bypass flag
shadow sync                  # force an immediate SQLite -> DuckDB ETL run
shadow cluster                # embed + (re-)cluster prompt text into archetypes
shadow run --budget 3.00      # budgeted shadow replay + top-2-archetype chain replay
shadow report [--raw]         # summarize captured traffic
shadow apply <rec-id> --writer claude_code --output-path ~/.claude/settings.json
```

Local sentence-transformers / HDBSCAN are optional (`pip install '.[ml]'`)
— without them, `mining/embed.py` and `mining/cluster.py` fall back to a
dependency-free hashing embedder and a cosine-threshold clustering
implementation, so the base install and CI never need to download a
model. HDBSCAN specifically (numpy/scipy/scikit-learn, no torch, ~65MB)
*was* installed and its real backend verified in this session — see
`tests/unit/test_cluster.py::test_real_hdbscan_backend_separates_topics`
(auto-skips if `hdbscan` isn't installed). Sentence-transformers pulls in
torch and was left uninstalled/unverified here on size/time grounds; the
fallback hashing embedder is what's actually exercised everywhere else.

OTel span export is on by default and fully local: every replay run,
transcript ingest, and proxy request is traced to
`~/.shadowtrace/logs/traces.jsonl` as JSON lines (no collector required).
Real OTLP export to your own collector is opt-in via
`SHADOWTRACE_OTLP_ENDPOINT` and needs `pip install '.[otel]'`.

### Dashboard

```bash
cd shadowtrace/dashboard/web
npm install && npm run build   # or `npm run dev` for hot-reload against
                                # a `shadow up`-style backend on :8788
```

`shadowtrace/dashboard/api.py` serves `/api/*` over DuckDB plus
`/healthz`/`/readyz`, and mounts `dashboard/web/dist/` if present. Verified
end-to-end in this session: built the React bundle, seeded a DuckDB
analytics store, ran the FastAPI backend, and confirmed both `/api/summary`
returned real seeded data and `/` served the built `index.html` with
correctly-referenced asset paths. No real browser was available in this
environment, so the pixels themselves weren't eyeballed — worth a manual
`npm run dev` + look before you rely on it for anything visual.

## Repo layout

Mirrors `plan.md` §1 exactly: `shadowtrace/{proxy,ingest,mining,replay,
recommend,dashboard,db}/`, `shadowtrace/cli.py`, `tests/{unit,
integration,e2e,perf,fakes,fixtures}/`.

## What's covered, scenario by scenario

The plan's §3.3 test matrix (E1–E14) is implemented as a targeted pytest
suite rather than a literal docker-compose harness (see below for why).
Coverage against that matrix:

| # | Scenario | Where |
|---|---|---|
| E1 | Golden path (ingest→capture→ETL→mine→replay→SPRT→recommend→apply) | `tests/e2e/test_golden_path.py` |
| E2 | Live-session tail, incremental, multi-session | `tests/integration/test_live_session_tail.py` + unit tests in `test_claude_transcripts.py` |
| E3 | Malformed transcript lines skipped, not crashed | `test_claude_transcripts.py` |
| E4 | Proxy lane golden path (streaming + tool_use) | `tests/integration/test_proxy.py` |
| E5 | Failure injection: capture writer killed mid-stream, SQLite locked, spill-dir also unwritable | `test_proxy.py`, `tests/unit/test_capture.py` |
| E6 | Redaction e2e: no secret leak across SQLite/DuckDB/logs, quarantine excluded from sampler | `tests/integration/test_redaction_e2e.py` |
| E7 | Replay budget cap + checkpoint/resume without double-spend | `tests/unit/test_runner.py` |
| E8 | SPRT correctness (exact boundary cases + 8k-trial empirical error-rate check) | `tests/unit/test_sprt.py` |
| E9 | Chain replay: candidate outputs carried forward, per-chain sub-cap | `replay/runner.py::run_chain`, `test_runner.py`, `test_replay_tracing.py` |
| E10 | Regression watch: spot-checks below floor → alert → byte-identical revert, history intact | `recommend/engine.py::watch_and_revert`, `tests/integration/test_regression_watch.py` |
| E11 | claude.ai export re-import idempotency | `tests/unit/test_claude_export.py` |
| E12 | Cold start: empty DB, dashboard renders empty states, no crash | `tests/integration/test_cold_start.py` |
| E13 | Egress containment (socket-blocking fixture) | `tests/integration/test_egress_containment.py` |
| E14 | Schema migration on a populated DB, data intact | `shadowtrace/db/migrations.py`, `tests/unit/test_migrations.py` |

**Why not literal docker-compose:** the value of E1–E14 is the assertions
— fail-open under failure injection, no secret leakage, idempotent
re-ingest, exact SPRT boundaries, byte-identical revert. A pytest harness
against fakes (`tests/fakes/anthropic_server.py`) and real local
mechanisms (real SQLite file locks, real sockets for egress containment,
a real local HTTP server for the perf test) verifies the same behavior
with less infrastructure and more reliable, faster CI than spinning up
containers would — so that's what's here instead. Nightly-cadence
concerns from §3.3 (full matrix nightly, live smoke before release) are
a CI-scheduling policy, not a code question, and aren't set up since
there's no live release process for this exercise.

## Scope and what's simplified

Being upfront about the gap between "the whole plan, implemented" and
what's actually verified:

- **Sentence-transformers:** not installed (pulls in torch; left
  uninstalled on size/time grounds). The fallback hashing embedder is
  what every test actually exercises for embeddings; HDBSCAN's real
  backend (lighter, no torch) *was* installed and verified — see above.
- **Judge grader:** `replay/graders/judge.py`'s default is a crude
  token-overlap heuristic, not a live cheap-model call — this is
  deliberate (CI must never need a live API key), and the module is
  built so a real `judge_fn` drops in without changing call sites.
- **Replay call wiring (`cli.py::_live_call`):** real HTTP wiring to
  Anthropic/Ollama, with its request/response handling verified against
  mocked endpoints (`tests/unit/test_cli_live_call.py`, via `respx` — no
  real network). Never exercised against the *actual* Anthropic/Ollama
  APIs in this session.
- **Transcript schema:** `ingest/claude_transcripts.py` parses a
  best-effort model of Claude Code's JSONL transcript format (based on
  general knowledge of its shape, not a verified spec) — defensively, so
  unknown fields/lines are skipped and counted rather than crashing.
  Validate against a real `~/.claude/projects/**/*.jsonl` file if the
  exact shape matters to you.
- **Request/response schema simplification:** each transcript-derived
  trace row's `request_json` is the single preceding user turn, not the
  full growing conversation history replayed into every row (see the
  docstring in `claude_transcripts.py` for why — full context is a
  downstream join on `session_id`, which is exactly what chain replay
  uses).
- **OTel tracing:** real spans (via `opentelemetry-sdk`, a core
  dependency) for the replay pipeline, transcript ingest, and proxy
  requests, exported locally as JSON lines by default. No distributed
  collector deployment or dashboarding on top of the trace data — you'd
  point `SHADOWTRACE_OTLP_ENDPOINT` at your own Grafana/Jaeger/etc. stack
  for that, which isn't set up or tested here since it needs external
  infrastructure.
- **Fixture corpus (§3.6):** the full 304 synthetic traces across 8
  archetypes the plan calls for (`tests/fixtures/generate_traces.py`,
  regeneratable, seeded).
- **CLI command surface:** `resume`, `sync`, `cluster`, and `metrics` are
  pragmatic additions beyond the five commands (`up`, `pause`, `run`,
  `report`, `apply`) the plan names in §1 — needed to actually drive the
  pipeline end to end.
- **Dashboard visuals:** built and wired to real API data (see above),
  but not eyeballed in an actual browser — no browser was available in
  this environment.

Everything else — redaction (regex + entropy quarantine), fail-open
capture with JSONL spill under real lock contention, byte-identical
streaming passthrough, idempotent transcript/export ingestion, idempotent
ETL, versioned schema migrations on populated databases, stable archetype
identity across re-clusters, budget-capped/checkpointed/resumable single-
call replay *and* chain replay with its own sub-cap, Wald SPRT (with an
empirical-error-rate test over 8,000 simulated trials), the
quota-headroom-vs-USD recommender currency split from D1, regression
watch with automatic byte-identical revert, egress containment, and a
proxy latency perf budget — is real and covered by tests that exercise
the actual failure modes described in `plan.md`, not just the happy path.

## CI

`.github/workflows/ci.yml` runs `ruff check`, `ruff format --check`,
`mypy shadowtrace`, and three `pytest` passes on every push and PR:
unit/integration (85% coverage gate), the `e2e` marker, and the `perf`
marker (proxy latency budget).
