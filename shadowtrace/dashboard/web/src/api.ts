export interface Summary {
  total_traces: number;
  quarantined_traces: number;
  archetype_count: number;
  unassigned_traces: number;
  traces_by_tool: Record<string, number>;
  traces_by_model: Record<string, number>;
}

export interface Archetype {
  id: string;
  label: string;
  pinned: boolean;
  trace_count: number;
}

export interface Trace {
  id: string;
  ts: number;
  source: string;
  tool: string | null;
  model: string;
  tokens_in: number | null;
  tokens_out: number | null;
  cost_usd: number | null;
}

export interface Recommendation {
  id: string;
  archetype_id: string;
  archetype_label: string;
  candidate: string;
  currency: "quota_headroom_pct" | "usd_per_month";
  value: number;
  pass_rate: number;
  pass_rate_ci_low: number;
  pass_rate_ci_high: number;
  expires_ts: number;
}

export interface SystemInfo {
  metrics: Record<string, unknown>;
  duckdb_path: string;
  sqlite_path: string;
  paused: boolean;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`${path} -> HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

export const api = {
  summary: (): Promise<Summary> => getJSON("/api/summary"),
  archetypes: (): Promise<Archetype[]> => getJSON("/api/archetypes"),
  traces: (archetypeId?: string): Promise<Trace[]> =>
    getJSON(archetypeId ? `/api/traces?archetype_id=${encodeURIComponent(archetypeId)}` : "/api/traces"),
  recommendations: (): Promise<Recommendation[]> => getJSON("/api/recommendations"),
  system: (): Promise<SystemInfo> => getJSON("/api/system"),
};
