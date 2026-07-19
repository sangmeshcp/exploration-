# plan.md — Personal Shadow Tracing & Model Recommender

> Implementation plan for Claude Code. Source design: `personal-shadow-tracing-hld.md`.
> Local-first, single-user app: capture own LLM traffic → cluster into task archetypes → budgeted shadow replay → objective-aware model recommendations → apply as router config.

---

## 0. Decisions (RESOLVED 2026-07-13)

| # | Decision | Resolution |
|---|---|---|
| D1 | Claude access mode | **Subscription (Max/Pro) via Claude Code** |
| D2 | v0 dashboard stack | **FastAPI + React from start** |
| D3 | Local model candidates | **Yes — Ollama/vLLM lane enabled by default (local GPU available)** |
| D4 | Proxy implementation | Custom FastAPI shim (~500 LoC) — proxy is now the *secondary* lane, see D1 consequences |
| D5 | Embeddings | Local sentence-transformers (privacy default; GPU makes this free) |

**D1 consequences (important — reshapes M1):**
1. **Primary capture = Claude Code local session transcripts.** Claude Code persists full session JSONL under `~/.claude/projects/<project>/*.jsonl` (complete message lists incl. tool_use). A file-watcher ingester on these transcripts captures the main lane with zero proxy involvement and zero risk of breaking OAuth-bound subscription auth. Supplement with Claude Code hooks (session metadata) + built-in OTel (token metrics).
2. **Proxy demoted to secondary lane:** API-key scripts, Codex/other CLIs, and anything with a `base_url` knob. Still built in M1, still fail-open, but no longer the critical path.
3. **Recommender currency = frontier quota headroom** (subscription is flat-rate): "route archetypes X,Y to Haiku/local → reclaim ~N% of your 5-hour/weekly limits." Dollar savings shown only for the API-key lane.
4. **Replay needs a separate metered API key** — subscription auth can't be used for programmatic replay calls. Budget caps in `shadow run` apply to this key. Local Ollama/vLLM candidates (D3) reduce replay spend to ~zero for archetypes where local models are viable.

---

## 1. Stack & Repo Layout

- Python 3.12, `uv` for env, `ruff` + `mypy --strict`, `pytest`.
- Hot-path capture store: **SQLite (WAL)** append-only. Analytics store: **DuckDB**, populated by ETL. (See review R2 for why not DuckDB on the hot path.)
- FastAPI (proxy + backend API), httpx (upstream), React + Vite + Tailwind (dashboard, from M2 per D2).

```
shadowtrace/
├── proxy/            # passthrough gateway + capture tee (secondary lane)
│   ├── server.py     # FastAPI app, /v1/messages, /v1/chat/completions
│   ├── passthrough.py# streaming tee, fail-open logic
│   ├── capture.py    # async writer → SQLite WAL
│   └── redact.py     # secret/PII scrub before persistence
├── ingest/
│   ├── claude_transcripts.py  # PRIMARY lane: watchdog on ~/.claude/projects/**/*.jsonl
│   ├── etl.py        # SQLite → DuckDB batch ETL + dedup
│   ├── claude_export.py  # claude.ai conversations.json importer
│   └── otel_sink.py  # OTLP receiver for Claude Code telemetry (token metrics)
├── mining/
│   ├── embed.py      # local embeddings
│   ├── cluster.py    # HDBSCAN + label
│   └── archetypes.py # persistence, rename/merge ops
├── replay/
│   ├── sampler.py    # stratified sampling per archetype
│   ├── ladder.py     # candidate registry (+ price table)
│   ├── runner.py     # budget-capped executor
│   ├── graders/      # deterministic checks + judge
│   └── sprt.py       # sequential test
├── recommend/
│   ├── engine.py     # objective-aware recommendations
│   └── apply/        # actuation writers (claude_code.py, litellm.py, nanoclaw.py)
├── dashboard/        # streamlit_app.py (v0) → web/ (M4)
├── cli.py            # `shadow` entrypoint: up, pause, run, report, apply
├── db/               # schemas + migrations (sqlite_schema.sql, duckdb_schema.sql)
└── tests/            # mirrors src tree + e2e/ + fixtures/
```

### Core schema (SQLite capture; DuckDB mirrors + derived tables)

```sql
CREATE TABLE traces (
  id TEXT PRIMARY KEY,            -- ulid
  ts INTEGER NOT NULL,
  source TEXT NOT NULL,           -- proxy|otel|claude_export
  tool TEXT,                      -- claude_code|script|chat|codex
  model TEXT NOT NULL,
  request_json BLOB NOT NULL,     -- full message list incl. tool_use blocks (redacted)
  response_json BLOB NOT NULL,    -- full content blocks (redacted)
  tokens_in INTEGER, tokens_out INTEGER,
  latency_ms INTEGER,
  cost_usd REAL,                  -- NULL for subscription lane
  session_id TEXT,                -- chain reconstruction
  parent_id TEXT,                 -- previous call in chain
  redaction_flags TEXT            -- what was scrubbed / quarantined
);
-- DuckDB adds: archetype_assignments, replay_results, recommendations, applied_policies
```

Store **full request/response JSON**, not flattened text — Anthropic responses contain `tool_use`/`tool_result` blocks and system prompts that graders and chain-replay need intact (review R1).

---

## 2. Milestones

### M0 — Skeleton & guardrails (½ day)
- Repo scaffold, CI (GitHub Actions: ruff, mypy, pytest, coverage ≥85% gate).
- `shadow up` starts proxy + collector; `shadow pause` sets capture bypass flag.
- **Accept:** CI green on empty skeleton; proxy boots and forwards a hardcoded request.

### M1 — Capture (transcript ingestion primary + proxy secondary + redaction + ETL)
Tasks:
1. **`claude_transcripts.py` (primary lane):** watchdog-based watcher on `~/.claude/projects/**/*.jsonl`; incremental tail-parse (files append during live sessions), map transcript entries → `traces` rows with `session_id` = transcript file, `parent_id` = message ordering, `source=transcript`, `cost_usd=NULL`. Idempotent re-ingest keyed on message uuid. Handle transcript format drift defensively (unknown fields preserved in raw JSON, parser version stamped).
2. `/v1/messages` (Anthropic) and `/v1/chat/completions` (OpenAI-compat) passthrough proxy with **streaming tee** for the API-key lane: forward SSE chunks unmodified in real time; assemble copy async for capture. Non-streaming path too.
3. **Fail-open invariants (both lanes):** ingest/capture exception ⇒ user's session unaffected; upstream error ⇒ passed through verbatim; SQLite unavailable ⇒ spill to JSONL, log, continue. Transcript watcher is read-only on Claude Code's files — never locks, never writes there.
4. Redaction (`redact.py`): regex pack (AWS/GCP keys, JWTs, PEM blocks, common token formats) + entropy scan. Entropy hits are **quarantined** (stored encrypted, flagged, excluded from replay) not deleted — code is high-entropy and false positives would gut coding traces (review R3).
5. Cost estimation from versioned price table for API lane; quota-weight estimation (tokens vs. published limit heuristics) for subscription lane.
6. OTel sink for Claude Code telemetry (token metrics, model events) — cross-check against transcript-derived counts.
7. `claude_export.py`: parse claude.ai conversations.json, idempotent upsert.
8. ETL job SQLite→DuckDB (cron + on-demand), idempotent.

**Accept:** a real Claude Code session (subscription auth, no proxy) appears fully in `shadow report --raw` within 60s of message completion; an API script through the proxy shows zero behavioral diff; kill -9 on any capture component disturbs nothing.

### M2 — Task mining + usage dashboard
1. Local embedding of prompt texts (extract user-visible text from request JSON; ignore system boilerplate).
2. HDBSCAN clustering; auto-label clusters via one cheap LLM call over k exemplars (feature-flag: fully-local labeling).
3. Archetype CRUD: rename/merge/pin; assignments persist across re-clustering (stable via medoid matching).
4. Dashboard v1 (FastAPI backend + React/Vite/Tailwind per D2): usage over time, model mix, tool split, archetype treemap, trace drill-down. Backend exposes `/api/*` over DuckDB; React app served from same process.

**Accept:** ≥80% of a week of real traffic lands in a labeled archetype; merge/rename survives re-cluster.

### M3 — Shadow replay
1. `sampler.py`: stratified by archetype; prioritize volume + recency; exclude quarantined traces.
2. Candidate ladder config (`ladder.yaml`): Haiku 4.5, Sonnet 4.6, OpenRouter open-weights, **local lane default-on (D3): Ollama for convenience, vLLM server for throughput replay runs** — local candidates cost $0, so the sampler exhausts local candidates before spending replay budget on cloud ones.
3. `runner.py`: `shadow run --budget 3.00 [--archetype X]` — cost preview, hard cap, resumable checkpoints, concurrency-limited, provider-rate-limit aware.
4. Graders, cheapest-first:
   - deterministic: JSON/schema validity, code parses (tree-sitter), diff applies, length/format constraints
   - pairwise judge: candidate answer vs. **stored frontier answer** (free baseline), rubric per archetype, judge prompt versioned + hash-stamped into results
5. `sprt.py`: per (archetype, candidate); params α=β=0.05, p0=floor (default 0.90, per-archetype override), p1=floor−0.10. Stop states: pass / fail / budget-exhausted(inconclusive).
6. Chain replay mode for agentic archetypes: replay top-2 archetypes as sub-chains (re-execute sequence with candidate, carry candidate outputs forward); single-call elsewhere, marked lower-confidence.

**Accept:** a $3 budgeted run over seeded fixtures produces SPRT-significant verdicts for ≥3 archetypes; interrupted run resumes without double-spend.

### M4 — Recommender + actuation + regression watch
1. Objective-aware engine (D1): **primary currency = frontier quota headroom** ("reclaim ~N% of 5-hour/weekly limits"); $/mo shown for API-key lane; local-viability flags per archetype.
2. Recommendation cards: cheapest passing model, pass-rate CI, latency delta, expiry date (stale after model-market change or 45 days).
3. Apply writers: Claude Code settings/`CLAUDE.md` model hints; LiteLLM router yaml; NanoClaw-style complexity-router rules. Each writer: dry-run diff → confirm → write → record in `applied_policies`.
4. Regression watch: online spot-check sampling (grade N/week per applied archetype against floor); breach ⇒ dashboard alert + one-click revert.
5. Human spot-review queue: 10 samples/archetype where judge decided alone (review R4).
6. Dashboard v2 (extend React app): recommendation cards, savings/quota tracker, replay leaderboards.

**Accept:** applying a recommendation changes real router config via dry-run→confirm; forced regression fixture triggers alert + revert path.

---

## 3. Testing Plan

### 3.1 Unit (pytest, per module)
- `redact`: corpus of true secrets (must scrub) + high-entropy code (must NOT scrub — quarantine only); property tests with hypothesis for format variants.
- `sprt`: known sequences → exact accept/reject boundaries; simulation test (10k trials) confirming empirical error ≤ α/β.
- `sampler`: stratification proportions, exclusion rules, determinism under seed.
- `ladder`: price-table math vs. hand-computed costs.
- `capture`: writer batching, JSONL spill on induced SQLite failure.
- `etl`: idempotency (run twice ⇒ identical DuckDB state), dedup on claude.ai re-import.
- `cluster`: stable archetype ids across re-cluster on fixture corpus.
- `recommend`: currency selection per access mode; expiry logic.

### 3.2 Integration
- **Transcript watcher suite (primary lane):** fixture `.jsonl` transcripts (incl. tool_use chains) appended incrementally by a writer process — asserts incremental tail-parse correctness, idempotent re-ingest, no file locks held on Claude Code's directory, graceful handling of malformed/unknown-schema lines (skip + flag, never crash).
- **Fake upstream** (`tests/fakes/anthropic_server.py`): FastAPI stub emitting realistic SSE streams incl. `tool_use` blocks, error statuses, mid-stream disconnects.
- Proxy suite against fake upstream:
  - byte-identical passthrough (streaming + non-streaming) — golden transcript diff
  - client receives full stream even when capture writer is killed mid-stream
  - upstream 429/500/timeout passed through verbatim; nothing captured as success
  - tool_use round-trip: multi-block conversation captured with structure intact
- Replay runner against fake upstream: budget cap enforcement (stops within one call of cap), checkpoint/resume without double-spend, rate-limit backoff.
- Grader integration: fixture set of (frontier answer, candidate answer, expected verdict) triples per grader; judge tests run against recorded cassettes (respx/vcr) — **no live API in CI**.

### 3.3 End-to-End Testing (comprehensive)

**Harness:** `tests/e2e/` runs the full stack via docker-compose (or process-manager fixture): SQLite/DuckDB in a temp `~/.shadowtrace-test/`, fake Anthropic/OpenRouter upstreams, a **synthetic Claude Code session simulator** that appends realistic JSONL transcripts incrementally (streaming-like cadence, tool_use chains, multi-session concurrency) into a temp `~/.claude/projects/` mirror, and a local Ollama stub. Every scenario asserts on final DuckDB state, emitted configs, and **observability signals** (§4.5) — a scenario fails if its expected metrics/log events are absent.

**Scenario matrix:**

| # | Scenario | Asserts |
|---|---|---|
| E1 | Golden path: simulator writes 3 sessions → watcher ingests → ETL → cluster → `shadow run` (local + fake cloud candidates) → recommendation → apply to temp Claude Code settings → tracker updates | Row counts, archetype assignments, SPRT verdicts, config diff matches expected, `capture_lag_seconds` < 60 |
| E2 | Live-session tail: simulator appends to an open transcript over 5 min | Incremental rows appear without waiting for file close; no partial/corrupt rows |
| E3 | Malformed transcript lines (truncated JSON, unknown schema version) | Skipped + flagged, ingest continues, `parse_errors_total` incremented, zero crash |
| E4 | Proxy lane golden path (API script → proxy → fake upstream, streaming + tool_use) | Byte-identical passthrough transcript, captured rows structurally complete |
| E5 | Failure injection — capture writer killed mid-session; SQLite locked; disk-full on spill dir | Live session unaffected (E2/E4 rerun green), spill JSONL created, recovery ETL backfills, `fail_open_events_total` incremented |
| E6 | Redaction end-to-end: seeded secrets in simulator prompts | No secret string anywhere in SQLite/DuckDB/logs/exports; quarantine rows flagged + excluded from replay sample |
| E7 | Replay budget: cap $2.00 with per-call cost fixtures priced to exceed it | Runner stops within one call of cap; checkpoint file valid; resume completes without double-spend; `replay_spend_usd` matches ledger |
| E8 | SPRT correctness in situ: candidate stubbed at known pass-rates (0.95 / 0.70) | Accept/reject verdicts match expectation; sample counts within theoretical bounds |
| E9 | Chain replay: agentic archetype replayed as sub-chain with candidate outputs fed forward | Downstream calls use candidate outputs (asserted via fake-upstream request log); per-chain sub-cap honored |
| E10 | Regression watch: applied policy, then spot-check grades forced below floor | Alert raised, revert restores prior config byte-identically, `applied_policies` history intact |
| E11 | claude.ai export re-import (same file twice, then extended file) | Idempotent; only-new rows on extended import |
| E12 | Fresh-install cold start: empty state → `shadow up` → dashboard renders empty-states → first ingest | No crashes on empty DB; onboarding hints shown |
| E13 | Egress containment: socket-blocking fixture allows only fake-upstream + candidate hosts | Any other outbound attempt fails the run |
| E14 | Upgrade/migration: run vN schema, apply vN+1 migration on populated DB | Data intact, checksums match, app boots |

**Cadence & gates:** E1–E6 on every PR; full matrix nightly; full matrix + **live smoke** (real Haiku key, $0.10 cap, real Ollama) before any release tag. A milestone is not "done" until its scenarios pass: M1 → E2–E6, E11, E13; M2 → E1 partial, E12; M3 → E7–E9; M4 → E10, E14.

### 3.4 Performance
- Proxy overhead budget: **p50 < 5ms, p99 < 15ms** added vs. direct call (locust/k6 against fake upstream, streaming). CI perf job fails on 2× regression.
- Capture writer sustains 50 req/s burst without backpressure on the live path.

### 3.5 Security/privacy tests
- Grep-style assertion: no unredacted secret from the test corpus ever appears in SQLite/DuckDB/JSONL/logs after a full e2e run.
- Quarantined rows excluded from replay sampler and dashboard exports.
- DB file encryption at rest verified (open without key fails).

### 3.6 Test data
- `fixtures/traces/` — 300 synthetic traces across 8 archetypes (generated once, committed), incl. tool-use chains, non-English, long-context, adversarial near-duplicates for cache-like confusions.
- Fixture generator script documented so the corpus can be regenerated/extended.

---

## 4. Non-Functional Requirements
- Fail-open is the prime invariant: **the app must never break or slow the user's real work**. Any violation is a P0.
- Single command up/down; state = one directory (`~/.shadowtrace/`), backup = copy it.
- No network calls except: live forwarding, replay candidates, optional cluster labeling. Enforced by an egress test using a socket-blocking fixture (E13).

## 4.5 Observability (the app instruments itself)

OTel-native throughout: every component emits metrics + structured JSON logs + traces to an embedded pipeline (OTLP → local file/SQLite exporter by default; optional OTLP export to the user's own collector/Grafana stack via config — never on by default).

### Metrics (Prometheus-convention names)

| Component | Metrics |
|---|---|
| Transcript watcher | `capture_lag_seconds` (now − last message ts ingested, gauge), `transcripts_watched`, `messages_ingested_total`, `parse_errors_total`, `unknown_schema_lines_total` |
| Proxy | `proxy_added_latency_ms` (histogram, p50/p99 — feeds the perf budget), `requests_total{status}`, `fail_open_events_total`, `capture_queue_depth`, `spill_rows_total` |
| Redaction | `scrubbed_total{pattern}`, `quarantined_total`, `redaction_latency_ms` |
| ETL | `etl_rows_total`, `etl_lag_seconds`, `dedup_dropped_total`, `etl_failures_total` |
| Mining | `archetype_count`, `unassigned_traces_ratio`, `embedding_latency_ms` |
| Replay | `replay_spend_usd` (counter, per key), `replay_calls_total{candidate,archetype}`, `sprt_verdicts_total{state}`, `provider_errors_total{provider,code}`, `checkpoint_age_seconds` |
| Recommender | `active_policies`, `recommendations_pending`, `regression_alerts_total`, `spot_checks_total{result}` |
| Self | `db_size_bytes`, `component_up{component}`, build info |

### Tracing
- Replay pipeline fully traced: `shadow_run` root span → sample → per-call candidate spans (model, tokens, cost attrs) → grader spans → SPRT decision span. One trace answers "why did this run cost $2.40 and stop early on archetype X."
- Ingest traced per transcript file; proxy traced per request with correlation id also stamped into the captured row (log↔trace↔data joinability).

### Logs
- Structured JSON, per-component files under `~/.shadowtrace/logs/`, size-based rotation. Redaction runs on log output too (same pack as capture) — logs must never leak what the DB wouldn't.

### Health & alerting (local, no cloud)
- `/healthz` (liveness) + `/readyz` (DB writable, watcher attached, disk headroom) on the backend.
- Dashboard "System" tab: component status, capture lag, queue depths, spend gauges, last 50 warnings.
- Local alert rules → desktop notification + dashboard banner: `capture_lag_seconds > 300`, any `fail_open_events_total` increase, `spill_rows_total > 0`, `etl_lag_seconds > 3600`, replay spend ≥ 80% of budget, regression alert raised, disk < 2GB.

### SLOs (self-imposed, tracked on System tab)
| SLO | Target |
|---|---|
| Capture completeness (transcript-derived token counts vs. OTel-reported, daily cross-check) | ≥ 99% |
| Capture lag p95 | < 60s |
| Proxy added latency p99 | < 15ms |
| ETL freshness | < 1h |
| Zero unredacted secrets (E6-style audit job, weekly) | 0 findings |

### Observability testing
- Unit: metric emission per code path (counter increments asserted).
- E2E: every scenario in §3.3 asserts its expected signals (e.g., E5 must show `fail_open_events_total` +1 and a WARN log with correlation id).
- Alert-rule tests: synthetic metric injection → notification fires.

### Milestone folding
Observability is built with each milestone, not after: M0 = OTel pipeline + `/healthz` + logging skeleton; M1 = watcher/proxy/redaction/ETL metrics + capture-lag alert; M2 = mining metrics + System tab in React app; M3 = replay tracing + spend gauges/alerts; M4 = recommender metrics + regression alerting + SLO panel.

---

## 5. Second-Pass Design Review (performed; changes applied above)

| # | Finding | Judgment | Resolution |
|---|---|---|---|
| R1 | HLD stored `prompt/completion` as text; Anthropic traffic is structured content blocks (tool_use, system). Flattening destroys grader + chain-replay fidelity | Design bug | Schema stores full request/response JSON (§1) |
| R2 | HLD wrote proxy captures straight to DuckDB. DuckDB is single-writer and weak for concurrent hot-path writes while dashboard reads | Design bug | SQLite WAL on hot path, DuckDB via ETL (§1, M1.7) |
| R3 | Entropy-based redaction will mass-flag code (hashes, minified JS) — dropping those rows guts the dataset | Real risk | Quarantine-not-delete + encrypted storage (M1.3) |
| R4 | Judge circularity: pairwise-vs-stored-answer helps but the judge still shares family bias with the baseline | Residual risk | Human spot-review queue (M4.5); optionally cross-family judge as feature flag |
| R5 | Subscription-auth Claude Code may not honor custom base URL (OAuth-bound) — proxy lane could silently capture nothing for the main tool | **Resolved by D1** | Primary capture switched to local transcript JSONL ingestion (M1.1) — sidesteps auth entirely; proxy retained for API-key lane |
| R6 | Chain replay cost can explode (candidate output changes downstream calls) | Real risk | Restrict to top-2 archetypes, per-chain budget sub-cap (M3.6) |
| R7 | Recommendation staleness: model prices/releases shift monthly | Gap in HLD | Expiry dates + re-run triggers (M4.2) |
| R8 | claude.ai export re-imports would duplicate traces | Minor | Idempotent upsert keyed on export ids (M1.6) |
| Overall | Architecture sound for single-user scope; fail-open passthrough is the highest-risk component and gets the deepest test coverage accordingly | — | Test plan weights proxy integration tests heaviest (§3.2) |

---

## 6. Suggested Claude Code Execution Order
M0 → M1 (proxy tests before capture features) → dogfood capture for ≥3 days to accumulate real traces → M2 → M3 → M4. Dogfooding gap is intentional: M2 clustering and M3 sampling need real personal traffic, not just fixtures.
