# Observability, evaluation and monitoring

shadowtrace already captured LLM traffic, replayed it against cheaper models
and recommended routing changes. This layer makes three things first-class:
**instrumentation** (every stage reports the same metrics, logs and traces),
**evaluation** (composable scorers, an LLM judge, and human review that
checks the judge) and **monitoring** (drift, data quality, performance over
time, SLOs, alerts and error budgets).

It was shaped by a typical ML/GenAI observability platform remit. The table
maps each of those responsibilities to what exists here.

| Platform responsibility | In shadowtrace |
|---|---|
| Metrics, logs and distributed traces across inference, batch scoring, pipelines and agent orchestration, instrumented consistently | `common/instrument.py`: one decorator/context manager emits a span, `operation_duration_ms`, `operations_total{status}` and an error log with the trace id. Model-call spans carry OpenTelemetry GenAI attributes (`gen_ai.*`). |
| Primitives to monitor model performance, data quality, drift and degradation | `monitor/performance.py` (week-over-week pass rate per archetype × candidate, two-proportion z-test), `monitor/quality.py` (six data-quality checks), `monitor/drift.py` (PSI and Jensen–Shannon divergence on task mix, model mix and prompt length) |
| Evaluation frameworks for LLM and agentic systems: response quality, grounding/hallucination, task success, tool-use and trajectory correctness, LLM-as-judge, human-in-the-loop | `replay/graders/scorers.py` (format, completion, trajectory, grounding, judge scorers plus a composable `EvalSuite`), `llm_judge.py` (Claude judge via the Anthropic SDK, cost counted in the replay budget), `human_review.py` (review queue, agreement, Cohen's kappa, false-pass rate) |
| Build-vs-buy; SDKs, connectors and APIs to vendor platforms | Vendor-neutral interfaces rather than a vendor SDK: OTLP span export (`SHADOWTRACE_OTLP_ENDPOINT`), Prometheus `/metrics`, GenAI semantic conventions. See [Build vs buy](#build-vs-buy). |
| Reusable libraries and templates for observability-by-default | `@instrument(...)`, the `Scorer` interface, `register_sli(...)`, and `monitor/slo_rules.yaml` as a copy-and-edit template |
| Dashboarding, alerting and SLO/SLI building blocks with templates for performance, latency and cost | SLI registry + declarative objectives + stored evaluations → attainment and error budgets (`monitor/slo.py`); transition-based alerting with optional desktop notifications; the dashboard's **Health** tab; `/api/slo`, `/api/monitor`, `/api/evals` |

## Instrumentation

```python
from shadowtrace.common.instrument import instrument, instrumented

@instrument("etl.run", component="etl")
def run_etl(...): ...

with instrumented("judge.call", component="replay", attributes=genai_attributes(...)) as span:
    ...
```

Because names and labels are fixed, one alert rule (`pipeline_errors`:
`operation_error_ratio <= 0.05`) covers every instrumented stage. The CLI
registers a process-wide tracer provider, so decorated library code exports
spans without a tracer being passed through every call.

Spans go to `~/.shadowtrace/logs/traces.jsonl` by default. Set
`SHADOWTRACE_OTLP_ENDPOINT` (and `pip install '.[otel]'`) to send them to any
OTLP backend. Replay and judge spans use the GenAI conventions
(`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, ...), so
LLM-aware backends show them as model calls.

`GET /metrics` on the dashboard port serves every metric in Prometheus text
format.

## Evaluation

`shadow run` grades each candidate answer with `default_suite()`:

| Scorer | Gates? | Checks |
|---|---|---|
| `format` | yes | If the frontier answer was JSON, a unified diff or a parseable Python block, the candidate's must be too |
| `completion` | yes | The candidate answered: not empty, and not a refusal where the frontier answered |
| `trajectory` | yes | Same tools, same order, same argument keys as the frontier's tool calls: 0.7 × LCS similarity of tool names + 0.3 × argument-key overlap |
| `grounding` | no (advisory) | Share of checkable specifics in the answer (paths, identifiers, numbers, URLs, inline code) found in the request or frontier answer. A cheap hallucination signal, recorded but not gating until it has been calibrated against human review |
| `judge` | yes | Pairwise LLM-as-judge against the stored frontier answer. Fails closed if the judge errors |

Every score is stored in `eval_scores`, so results break down by archetype,
candidate and scorer (`shadow evals`, the Health tab).

**Tool use is now replayable.** Candidates receive the stored conversation as
real messages, plus the system prompt and the tools the frontier had. Transcript
rows don't record tool definitions, so permissive schemas are inferred from
the frontier's actual calls. That lets a candidate make the same next call,
which is what the trajectory scorer compares.

**LLM judge.** If `SHADOWTRACE_JUDGE_ANTHROPIC_API_KEY` (or the replay key, or
`ANTHROPIC_API_KEY`) is set and `pip install '.[judge]'` is installed, the
judge is `claude-haiku-5-5` (override with `SHADOWTRACE_JUDGE_MODEL`).
Otherwise the local word-overlap heuristic is used, and `shadow run` says
which. Judge spend counts against `--budget`.

**Human review.** `shadow review` queues results where the judge was decisive
(every deterministic scorer passed), up to N per archetype. It shows request,
frontier answer and candidate answer, and takes pass/fail. It then reports:

- **agreement**: the share of judge verdicts the human agreed with
- **Cohen's kappa**: agreement corrected for chance
- **false-pass rate**: judge PASSes a human overturned. This is the costly
  error, since it routes traffic to a model that can't do the job.
- **false-fail rate**: judge FAILs a human overturned (missed savings)

The `judge_trustworthiness` SLO and `judge_unreliable` alert watch these.

## Monitoring

`shadow monitor` (or `/api/monitor`) reports three things:

- **Drift**: the last 7 days against the prior 28, for archetype mix (including
  an `(unassigned)` bucket, so a growing share signals work the archetypes
  don't cover), model mix and prompt length. PSI below 0.1 is stable,
  0.1–0.25 is a moderate shift, and above 0.25 is significant.
- **Data quality**: missing token counts, empty responses, unknown models,
  quarantine ratio, unassigned ratio and future timestamps.
- **Performance**: pass rate per archetype × candidate this week against last
  week, combining offline replay verdicts with online spot-checks. A pair is
  flagged as degraded when the drop is significant (one-sided z ≤ −1.645).

## SLOs and alerts

Objectives live in `monitor/slo_rules.yaml`. Copy it to
`~/.shadowtrace/slo_rules.yaml` to customise; the copy replaces the defaults.

```yaml
- name: capture_lag
  kind: slo              # tracked over time, with an error budget
  sli: capture_lag_seconds
  op: "<="               # the condition that means healthy
  threshold: 60
  target: 0.95           # share of evaluations that must be healthy...
  window_days: 28        # ...over this window
  severity: warning
```

`shadow up` evaluates every objective each minute (`--monitor-interval`) and
stores the results in `slo_evaluations`. Attainment and remaining error
budget come from those stored results. Alerts log once when they start
firing and once when they resolve, not on every cycle. With `--notify`,
critical alerts also send a desktop notification.

Some SLIs exist only inside the `shadow up` process: capture lag, proxy
latency, fail-open events and pipeline error ratio. `shadow up` therefore
also serves the dashboard and `/metrics` on `:8788`. A one-shot `shadow slo`
evaluates the SLIs backed by the analytics store and reports the in-process
ones as `no_data`. It exits with status 1 when anything is breached, so it
can be used in cron or CI.

New SLIs register with one decorator:

```python
from shadowtrace.monitor.slo import register_sli


@register_sli("replay_spend_usd_24h")
def _spend(ctx): ...
```

## Build vs buy

| Need | Buy option | What's here | Why |
|---|---|---|---|
| Trace storage and UI | Langfuse, Arize Phoenix, Datadog LLM Observability, Honeycomb | Local JSONL spans + OTLP export with GenAI attributes | Local-first is a hard requirement (prompts never leave the machine by default). OTLP plus the GenAI conventions means any of these can be plugged in later with one env var, no code change. |
| Metrics and alerting | Prometheus + Alertmanager, Grafana Cloud, Datadog | In-process registry, `/metrics`, local SLO engine | A single-user desktop tool can't assume a metrics stack. `/metrics` lets someone who has one scrape it anyway. |
| Eval frameworks | promptfoo, DeepEval, Ragas, OpenAI Evals, Braintrust | Small scorer interface + suite | Grading here is pairwise against a stored frontier answer for each captured trace, and results feed SPRT routing decisions. General eval frameworks assume curated datasets with gold labels. The scorer interface is plain async functions, so wrapping a DeepEval/Ragas metric as a scorer is a few lines if wanted. |
| Human review | Label Studio, Argilla | A terminal queue in `shadow review` | Volume is ~10 items per archetype. A labelling platform would cost more to run than the reviewing takes. |
| Drift detection | Evidently, WhyLabs, NannyML | PSI/JS over three categorical features | The features are low-cardinality categoricals with a few hundred rows a week; those tools are designed for wide tabular feature sets. |

The rule applied: build where the requirement is specific to this pipeline
(per-trace pairwise grading, quota-aware recommendations, local-only), and
keep a standard seam (OTLP, Prometheus, a plain scorer function) wherever a
vendor tool could reasonably replace the built piece later.

## Still open

- **OTel capture-completeness cross-check**: `ingest/otel_sink.py` exists but
  nothing starts it, so there's no `capture_completeness` SLI yet.
- **Online spot-checks for applied policies**: `record_spot_check` and
  `watch_and_revert` exist and feed the performance report and regression
  watch, but nothing schedules spot-check replays of live traffic yet.
- **Live calls never exercised for real**: provider calls and the judge are
  tested against mocked HTTP and a fake SDK client, not against real
  Anthropic, Ollama, vLLM or OpenRouter endpoints.
- **Grounding scorer is advisory** until human review shows it agrees with
  people often enough to gate.
