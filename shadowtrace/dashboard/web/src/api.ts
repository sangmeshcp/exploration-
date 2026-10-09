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

export interface SloEvaluation {
  name: string;
  kind: "slo" | "alert";
  severity: "info" | "warning" | "critical";
  sli: string;
  value: number | null;
  op: string;
  threshold: number;
  status: "ok" | "breach" | "no_data" | "error";
  description: string;
  ts: number;
}

export interface SloCompliance {
  name: string;
  target: number;
  window_days: number;
  n: number;
  attainment: number | null;
  error_budget_remaining: number | null;
}

export interface SloStatus {
  evaluations: SloEvaluation[];
  firing: SloEvaluation[];
  compliance: Record<string, SloCompliance>;
}

export interface DriftResult {
  feature: string;
  psi: number | null;
  js_divergence: number | null;
  status: "stable" | "moderate" | "significant" | "insufficient_data";
  baseline_n: number;
  recent_n: number;
  top_shifts: { key: string; baseline_share: number; recent_share: number; delta: number }[];
}

export interface QualityCheck {
  name: string;
  value: number | null;
  threshold: number;
  ok: boolean | null;
  description: string;
}

export interface PairPerformance {
  archetype_id: string;
  archetype_label: string;
  candidate: string;
  recent_n: number;
  recent_pass_rate: number | null;
  recent_ci: [number, number] | null;
  previous_n: number;
  previous_pass_rate: number | null;
  mean_latency_ms: number | null;
  z: number | null;
  degraded: boolean;
}

export interface MonitorReport {
  drift: DriftResult[];
  quality: QualityCheck[];
  performance: PairPerformance[];
}

export interface ScorerSummary {
  archetype_id: string;
  archetype_label: string;
  candidate: string;
  scorer: string;
  n: number;
  mean_value: number;
  pass_rate: number;
}

export interface EvalsReport {
  scorers: ScorerSummary[];
  judge_agreement: {
    n: number;
    agreement: number | null;
    cohen_kappa: number | null;
    false_pass_rate: number | null;
    false_fail_rate: number | null;
  };
  pending_reviews: number;
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
  slo: (): Promise<SloStatus> => getJSON("/api/slo"),
  monitor: (): Promise<MonitorReport> => getJSON("/api/monitor"),
  evals: (): Promise<EvalsReport> => getJSON("/api/evals"),
};
