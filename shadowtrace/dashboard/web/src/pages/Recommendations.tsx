import { useEffect, useState } from "react";
import { api, Recommendation } from "../api";

function formatValue(rec: Recommendation): string {
  if (rec.currency === "quota_headroom_pct") {
    return `reclaim ~${rec.value.toFixed(1)}% of quota`;
  }
  return `save ~$${rec.value.toFixed(2)}/mo`;
}

export default function Recommendations() {
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.recommendations().then(setRecs).catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return <p className="text-red-600">Failed to load recommendations: {error}</p>;
  }
  if (recs.length === 0) {
    return (
      <p className="text-slate-500 text-sm">
        No recommendations yet — run <code>shadow run --budget 3.00</code> to shadow-replay
        your archetypes.
      </p>
    );
  }

  return (
    <ul className="space-y-3">
      {recs.map((r) => (
        <li key={r.id} className="rounded-lg border border-slate-200 dark:border-slate-700 p-4">
          <div className="flex justify-between items-baseline">
            <span className="font-medium">{r.archetype_label}</span>
            <span className="text-sm text-slate-500">expires {new Date(r.expires_ts).toLocaleDateString()}</span>
          </div>
          <div className="mt-1 text-lg">
            Route to <span className="font-mono">{r.candidate}</span> — {formatValue(r)}
          </div>
          <div className="text-sm text-slate-500">
            pass rate {(r.pass_rate * 100).toFixed(1)}% (CI {(r.pass_rate_ci_low * 100).toFixed(0)}–
            {(r.pass_rate_ci_high * 100).toFixed(0)}%)
          </div>
        </li>
      ))}
    </ul>
  );
}
