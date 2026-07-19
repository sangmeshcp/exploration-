import { useEffect, useState } from "react";
import { api, SystemInfo } from "../api";

export default function System() {
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.system().then(setInfo).catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return <p className="text-red-600">Failed to load system info: {error}</p>;
  }
  if (!info) {
    return <p className="text-slate-500">Loading…</p>;
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-4 text-sm">
        <span
          className={`rounded px-2 py-1 ${info.paused ? "bg-amber-100 text-amber-800" : "bg-green-100 text-green-800"}`}
        >
          {info.paused ? "Capture paused" : "Capture active"}
        </span>
      </div>
      <div className="text-sm text-slate-500 space-y-1">
        <div>SQLite: {info.sqlite_path}</div>
        <div>DuckDB: {info.duckdb_path}</div>
      </div>
      <div>
        <h2 className="text-lg font-medium mb-2">Metrics snapshot</h2>
        <pre className="text-xs bg-slate-50 dark:bg-slate-800 rounded p-3 overflow-auto max-h-96">
          {JSON.stringify(info.metrics, null, 2)}
        </pre>
      </div>
    </div>
  );
}
