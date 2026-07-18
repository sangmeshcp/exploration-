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
mining, shadow replay + SPRT, recommender, actuation writers, CLI,
dashboard) is real, working code with tests, not a stub. **Read
[Scope and what's simplified](#scope-and-whats-simplified) below** before
assuming any single piece matches the plan's every detail — a handful of
things are deliberately scoped down from the full HLD given this was
built in one sitting rather than across the plan's suggested
multi-day dogfooding arc.

```
130 tests passing · ruff clean · mypy --strict clean · ~90% coverage
```

## Quick start

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

pytest                      # unit + integration (fast)
pytest -m e2e                # golden-path end-to-end scenario

shadow up                    # start the proxy (API-key lane) + transcript
                              # watcher (primary lane) + periodic ETL
shadow pause / shadow resume # toggle the capture bypass flag
shadow sync                  # force an immediate SQLite -> DuckDB ETL run
shadow cluster                # embed + (re-)cluster prompt text into archetypes
shadow run --budget 3.00      # budgeted shadow replay against the candidate ladder
shadow report [--raw]         # summarize captured traffic
shadow apply <rec-id> --writer claude_code --output-path ~/.claude/settings.json
```

Local sentence-transformers / HDBSCAN are optional (`pip install '.[ml]'`)
— without them, `mining/embed.py` and `mining/cluster.py` fall back to a
dependency-free hashing embedder and a cosine-threshold clustering
implementation, so the base install, CI, and all tests never need to
download a model.

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
integration,e2e,fakes,fixtures}/`.

## Scope and what's simplified

Being upfront about the gap between "the whole plan, implemented" and
what one sitting can actually deliver and verify:

- **E1–E14 docker-compose matrix (§3.3):** only the golden path (E1) is
  implemented, as `tests/e2e/test_golden_path.py` — ingest → capture →
  ETL → mine → replay → SPRT → recommend → apply, all wired together and
  asserted end to end. The other 13 scenarios (failure injection under
  `docker-compose`, chain-replay cost sub-caps, egress containment via a
  socket-blocking fixture, schema migration, live-model smoke tests) are
  not built. The runner's failure-handling, redaction-quarantine, and
  checkpoint/resume logic *are* unit-tested directly (that's most of what
  E5/E6/E7 would exercise), just not through the full docker-compose
  harness the plan describes.
- **OTel:** `common/metrics.py` is a real in-process Prometheus-style
  registry (counters/gauges/histograms) that every module wires into, and
  `ingest/otel_sink.py` parses a simplified subset of the OTLP/HTTP-JSON
  metrics envelope. There's no actual OTLP exporter/collector integration
  or distributed tracing (spans) — `4.5`'s tracing section is not built.
- **Mining fidelity:** embeddings/clustering default to dependency-free
  fallbacks (hashing embedder, cosine-threshold union-find) so the base
  install and CI need no model download. Install `.[ml]` for the real
  sentence-transformers + HDBSCAN path described in the plan (same
  call sites, just a better backend).
- **Judge grader:** `replay/graders/judge.py`'s default is a crude
  token-overlap heuristic, not a live cheap-model call — this is
  deliberate (CI must never need a live API key), and the module is
  built so a real `judge_fn` drops in without changing call sites.
- **Replay call wiring (`cli.py::_live_call`):** real, minimal HTTP
  wiring to Anthropic/Ollama for actual `shadow run` usage, but
  untested against a live network in this session (the `ReplayRunner`
  itself is thoroughly tested against injected fake `call_fn`s).
- **Transcript schema:** `ingest/claude_transcripts.py` parses a
  best-effort model of Claude Code's JSONL transcript format (based on
  general knowledge of its shape, not a verified spec) — defensively,
  so unknown fields/lines are skipped and counted rather than crashing.
  Validate against a real `~/.claude/projects/**/*.jsonl` file if the
  exact shape matters to you.
- **Request/response schema simplification:** each transcript-derived
  trace row's `request_json` is the single preceding user turn, not the
  full growing conversation history replayed into every row (see the
  docstring in `claude_transcripts.py` for why — full context is a
  downstream join on `session_id`, not per-row duplication).
- **Fixture corpus (§3.6):** ~96 synthetic traces across 8 archetypes
  (`tests/fixtures/generate_traces.py`, regeneratable), not the full 300
  the plan calls for — the generator's `PER_ARCHETYPE` constant scales it
  up trivially.
- **Proxy latency budget (§3.4):** no dedicated locust/k6 perf CI job;
  `proxy_added_latency_ms` is measured and exposed, but there's no
  automated regression gate on it yet.
- **CLI command surface:** `resume`, `sync`, `cluster`, and `metrics` are
  pragmatic additions beyond the five commands (`up`, `pause`, `run`,
  `report`, `apply`) the plan names in §1 — needed to actually drive the
  pipeline end to end.

Everything else — redaction (regex + entropy quarantine), fail-open
capture with JSONL spill, byte-identical streaming passthrough, idempotent
transcript/export ingestion, idempotent ETL, stable archetype identity
across re-clusters, budget-capped/checkpointed/resumable replay, Wald
SPRT (with an empirical-error-rate test over thousands of simulated
trials), the quota-headroom-vs-USD recommender currency split from D1,
and the dry-run → confirm → write → revert actuation flow with
byte-identical revert — is real and covered by tests that exercise the
actual failure modes described in `plan.md`, not just the happy path.

## CI

`.github/workflows/ci.yml` runs `ruff check`, `ruff format --check`,
`mypy shadowtrace`, and `pytest` (unit/integration with an 85% coverage
gate, then the e2e marker separately) on every push and PR.
