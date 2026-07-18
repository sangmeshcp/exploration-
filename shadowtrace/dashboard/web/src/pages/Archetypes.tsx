import { useEffect, useState } from "react";
import { api, Archetype, Trace } from "../api";

export default function Archetypes() {
  const [archetypes, setArchetypes] = useState<Archetype[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [traces, setTraces] = useState<Trace[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.archetypes().then(setArchetypes).catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!selected) {
      setTraces([]);
      return;
    }
    api.traces(selected).then(setTraces).catch((e) => setError(String(e)));
  }, [selected]);

  const maxCount = Math.max(1, ...archetypes.map((a) => a.trace_count));

  if (error) {
    return <p className="text-red-600">Failed to load archetypes: {error}</p>;
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div>
        <h2 className="text-lg font-medium mb-2">Archetypes</h2>
        {archetypes.length === 0 && (
          <p className="text-slate-500 text-sm">
            No archetypes yet — run <code>shadow cluster</code> after capturing some traffic.
          </p>
        )}
        <ul className="space-y-2">
          {archetypes.map((a) => (
            <li key={a.id}>
              <button
                onClick={() => setSelected(a.id)}
                className={`w-full text-left rounded border px-3 py-2 hover:bg-slate-50 dark:hover:bg-slate-800 ${
                  selected === a.id
                    ? "border-blue-500"
                    : "border-slate-200 dark:border-slate-700"
                }`}
              >
                <div className="flex justify-between">
                  <span>
                    {a.label} {a.pinned && <span title="pinned">📌</span>}
                  </span>
                  <span className="text-slate-500">{a.trace_count}</span>
                </div>
                <div className="mt-1 h-2 rounded bg-slate-100 dark:bg-slate-800">
                  <div
                    className="h-2 rounded bg-blue-500"
                    style={{ width: `${(a.trace_count / maxCount) * 100}%` }}
                  />
                </div>
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div>
        <h2 className="text-lg font-medium mb-2">Trace drill-down</h2>
        {!selected && <p className="text-slate-500 text-sm">Select an archetype to see its traces.</p>}
        <ul className="divide-y divide-slate-200 dark:divide-slate-700">
          {traces.map((t) => (
            <li key={t.id} className="py-2 text-sm">
              <div className="flex justify-between">
                <span className="font-mono text-xs text-slate-500">{t.id}</span>
                <span>{t.model}</span>
              </div>
              <div className="text-slate-500">
                {t.tokens_in ?? "-"} in / {t.tokens_out ?? "-"} out
                {t.cost_usd != null ? ` · $${t.cost_usd.toFixed(4)}` : ""}
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
