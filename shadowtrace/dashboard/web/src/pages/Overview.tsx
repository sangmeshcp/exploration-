import { useEffect, useState } from "react";
import { api, Summary } from "../api";

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg border border-slate-200 dark:border-slate-700 p-4">
      <div className="text-sm text-slate-500 dark:text-slate-400">{label}</div>
      <div className="text-2xl font-semibold">{value}</div>
    </div>
  );
}

export default function Overview() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.summary().then(setSummary).catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return <p className="text-red-600">Failed to load summary: {error}</p>;
  }
  if (!summary) {
    return <p className="text-slate-500">Loading…</p>;
  }

  const unassignedPct =
    summary.total_traces > 0
      ? Math.round((summary.unassigned_traces / summary.total_traces) * 100)
      : 0;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Total traces" value={summary.total_traces} />
        <StatCard label="Archetypes" value={summary.archetype_count} />
        <StatCard label="Unassigned" value={`${unassignedPct}%`} />
        <StatCard label="Quarantined" value={summary.quarantined_traces} />
      </div>

      <div>
        <h2 className="text-lg font-medium mb-2">Traffic by tool</h2>
        <ul className="space-y-1">
          {Object.entries(summary.traces_by_tool).map(([tool, count]) => (
            <li key={tool} className="flex justify-between text-sm">
              <span>{tool}</span>
              <span className="text-slate-500">{count}</span>
            </li>
          ))}
        </ul>
      </div>

      <div>
        <h2 className="text-lg font-medium mb-2">Model mix</h2>
        <ul className="space-y-1">
          {Object.entries(summary.traces_by_model).map(([model, count]) => (
            <li key={model} className="flex justify-between text-sm">
              <span>{model}</span>
              <span className="text-slate-500">{count}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
