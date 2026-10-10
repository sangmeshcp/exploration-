import { useEffect, useState } from "react";
import { api, EvalsReport, MonitorReport, SloEvaluation, SloStatus } from "../api";

const pct = (v: number | null | undefined, digits = 0) =>
  v === null || v === undefined ? "–" : `${(v * 100).toFixed(digits)}%`;
const num = (v: number | null | undefined) =>
  v === null || v === undefined ? "–" : Number(v.toPrecision(3)).toString();

const STATUS_STYLE: Record<string, string> = {
  ok: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
  breach: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
  no_data: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400",
  error: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
  stable: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
  moderate: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
  significant: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
  insufficient_data: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400",
};

function Badge({ status }: { status: string }) {
  return (
    <span className={`rounded px-2 py-0.5 text-xs whitespace-nowrap ${STATUS_STYLE[status] ?? ""}`}>
      {status.split("_").join(" ")}
    </span>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h2 className="text-lg font-medium mb-2">{title}</h2>
      {children}
    </section>
  );
}

function BudgetCell({ budget }: { budget: SloStatus["compliance"][string] }) {
  if (budget.attainment === null) return <span className="text-slate-500">no history yet</span>;
  const left = budget.error_budget_remaining;
  const exhausted = left !== null && left <= 0;
  return (
    <span>
      {pct(budget.attainment, 1)} of {pct(budget.target)} target ·{" "}
      <span className={exhausted ? "text-red-700 dark:text-red-400 font-medium" : ""}>
        {exhausted ? "budget exhausted" : `${pct(left)} budget left`}
      </span>
    </span>
  );
}

function ObjectiveRow({ ev, budget }: { ev: SloEvaluation; budget?: SloStatus["compliance"][string] }) {
  return (
    <tr className="border-t border-slate-200 dark:border-slate-700 align-top">
      <td className="py-2 pr-3">
        <div className="font-medium">{ev.name}</div>
        <div className="text-xs text-slate-500">{ev.description}</div>
      </td>
      <td className="py-2 pr-3"><Badge status={ev.status} /></td>
      <td className="py-2 pr-3 text-sm tabular-nums whitespace-nowrap">
        {num(ev.value)} <span className="text-slate-500">({ev.op} {ev.threshold})</span>
      </td>
      <td className="py-2 text-sm tabular-nums whitespace-nowrap">
        {budget ? <BudgetCell budget={budget} /> : ev.severity}
      </td>
    </tr>
  );
}

export default function Health() {
  const [slo, setSlo] = useState<SloStatus | null>(null);
  const [monitor, setMonitor] = useState<MonitorReport | null>(null);
  const [evals, setEvals] = useState<EvalsReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.slo(), api.monitor(), api.evals()])
      .then(([s, m, e]) => {
        setSlo(s);
        setMonitor(m);
        setEvals(e);
      })
      .catch((e) => setError(String(e)));
  }, []);

  if (error) return <p className="text-red-600">Failed to load health data: {error}</p>;
  if (!slo || !monitor || !evals) return <p className="text-slate-500">Loading…</p>;

  const slos = slo.evaluations.filter((e) => e.kind === "slo");
  const alerts = slo.evaluations.filter((e) => e.kind === "alert");
  const agreement = evals.judge_agreement;

  return (
    <div className="space-y-8">
      {slo.firing.length > 0 && (
        <div className="rounded-lg border border-red-300 bg-red-50 dark:bg-red-950/40 dark:border-red-800 p-4">
          <div className="font-medium text-red-800 dark:text-red-300">
            {slo.firing.length} firing: {slo.firing.map((f) => f.name).join(", ")}
          </div>
        </div>
      )}

      <Section title="Service level objectives">
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <tbody>
              {slos.map((ev) => (
                <ObjectiveRow key={ev.name} ev={ev} budget={slo.compliance[ev.name]} />
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Alert rules">
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <tbody>
              {alerts.map((ev) => (
                <ObjectiveRow key={ev.name} ev={ev} />
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Drift (last 7 days vs prior 28)">
        <div className="grid gap-3 md:grid-cols-3">
          {monitor.drift.map((d) => (
            <div key={d.feature} className="rounded-lg border border-slate-200 dark:border-slate-700 p-3">
              <div className="flex justify-between items-center mb-1">
                <span className="font-medium">{d.feature.split("_").join(" ")}</span>
                <Badge status={d.status} />
              </div>
              <div className="text-sm text-slate-500 tabular-nums">
                PSI {num(d.psi)} · n {d.baseline_n}/{d.recent_n}
              </div>
              <ul className="mt-2 text-xs space-y-0.5">
                {d.top_shifts.slice(0, 3).map((s) => (
                  <li key={s.key} className="flex justify-between gap-2">
                    <span className="truncate">{s.key}</span>
                    <span className="tabular-nums">
                      {pct(s.baseline_share)} → {pct(s.recent_share)}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Data quality (last 7 days)">
        <ul className="space-y-1">
          {monitor.quality.map((c) => (
            <li key={c.name} className="flex justify-between gap-3 text-sm">
              <span title={c.description}>{c.name.split("_").join(" ")}</span>
              <span className="flex gap-2 items-center tabular-nums">
                {pct(c.value, 1)} <span className="text-slate-500">max {pct(c.threshold)}</span>
                <Badge status={c.ok === null ? "no_data" : c.ok ? "ok" : "breach"} />
              </span>
            </li>
          ))}
        </ul>
      </Section>

      <Section title="Candidate performance (pass rate, this week vs last)">
        {monitor.performance.length === 0 ? (
          <p className="text-sm text-slate-500">No replay results or spot-checks yet — run `shadow run`.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-slate-500">
                <tr>
                  <th className="py-1 pr-3 font-normal">Archetype</th>
                  <th className="py-1 pr-3 font-normal">Candidate</th>
                  <th className="py-1 pr-3 font-normal">This week</th>
                  <th className="py-1 pr-3 font-normal">Last week</th>
                  <th className="py-1 font-normal">Latency</th>
                </tr>
              </thead>
              <tbody>
                {monitor.performance.map((p) => (
                  <tr key={`${p.archetype_id}:${p.candidate}`} className="border-t border-slate-200 dark:border-slate-700">
                    <td className="py-1 pr-3">{p.archetype_label}</td>
                    <td className="py-1 pr-3">{p.candidate}</td>
                    <td className="py-1 pr-3 tabular-nums">
                      {pct(p.recent_pass_rate)} <span className="text-slate-500">n={p.recent_n}</span>
                      {p.degraded && <span className="ml-2"><Badge status="breach" /></span>}
                    </td>
                    <td className="py-1 pr-3 tabular-nums">
                      {pct(p.previous_pass_rate)} <span className="text-slate-500">n={p.previous_n}</span>
                    </td>
                    <td className="py-1 tabular-nums">{p.mean_latency_ms === null ? "–" : `${Math.round(p.mean_latency_ms)} ms`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      <Section title="Evaluation">
        <div className="text-sm mb-3">
          Judge vs human ({agreement.n} reviewed, {evals.pending_reviews} pending): agreement{" "}
          {pct(agreement.agreement)}, κ {num(agreement.cohen_kappa)}, false-pass{" "}
          {pct(agreement.false_pass_rate)}
        </div>
        {evals.scorers.length === 0 ? (
          <p className="text-sm text-slate-500">No scored replay results yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-slate-500">
                <tr>
                  <th className="py-1 pr-3 font-normal">Archetype</th>
                  <th className="py-1 pr-3 font-normal">Candidate</th>
                  <th className="py-1 pr-3 font-normal">Scorer</th>
                  <th className="py-1 pr-3 font-normal">Mean</th>
                  <th className="py-1 font-normal">Pass</th>
                </tr>
              </thead>
              <tbody>
                {evals.scorers.map((s) => (
                  <tr key={`${s.archetype_id}:${s.candidate}:${s.scorer}`} className="border-t border-slate-200 dark:border-slate-700">
                    <td className="py-1 pr-3">{s.archetype_label}</td>
                    <td className="py-1 pr-3">{s.candidate}</td>
                    <td className="py-1 pr-3">{s.scorer}</td>
                    <td className="py-1 pr-3 tabular-nums">{s.mean_value.toFixed(2)}</td>
                    <td className="py-1 tabular-nums">{pct(s.pass_rate)} <span className="text-slate-500">n={s.n}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>
    </div>
  );
}
