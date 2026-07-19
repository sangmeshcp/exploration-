-- Analytics store (DuckDB). Populated by ETL from the SQLite hot path. See plan.md §1.
-- Schema version tracked in schema_meta, advanced by db/migrations.py.

CREATE TABLE IF NOT EXISTS schema_meta (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS traces (
    id VARCHAR PRIMARY KEY,
    ts BIGINT NOT NULL,
    source VARCHAR NOT NULL,
    tool VARCHAR,
    model VARCHAR NOT NULL,
    request_json VARCHAR NOT NULL,
    response_json VARCHAR NOT NULL,
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms INTEGER,
    cost_usd DOUBLE,
    session_id VARCHAR,
    parent_id VARCHAR,
    redaction_flags VARCHAR,
    quarantined BOOLEAN NOT NULL DEFAULT FALSE,
    prompt_text VARCHAR  -- extracted user-visible text, populated by mining.embed
);

CREATE TABLE IF NOT EXISTS archetype_assignments (
    trace_id VARCHAR NOT NULL,
    archetype_id VARCHAR NOT NULL,
    distance DOUBLE,
    assigned_ts BIGINT NOT NULL,
    PRIMARY KEY (trace_id)
);

CREATE TABLE IF NOT EXISTS archetypes (
    id VARCHAR PRIMARY KEY,        -- stable across re-cluster (medoid-matched)
    label VARCHAR NOT NULL,
    pinned BOOLEAN NOT NULL DEFAULT FALSE,
    merged_into VARCHAR,           -- non-null if this archetype was merged away
    created_ts BIGINT NOT NULL,
    updated_ts BIGINT NOT NULL,
    medoid_trace_id VARCHAR
);

CREATE TABLE IF NOT EXISTS replay_results (
    id VARCHAR PRIMARY KEY,
    archetype_id VARCHAR NOT NULL,
    candidate VARCHAR NOT NULL,
    source_trace_id VARCHAR NOT NULL,
    verdict VARCHAR NOT NULL,      -- pass|fail
    cost_usd DOUBLE NOT NULL,
    latency_ms INTEGER,
    grader VARCHAR NOT NULL,
    run_id VARCHAR NOT NULL,
    ts BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS sprt_verdicts (
    archetype_id VARCHAR NOT NULL,
    candidate VARCHAR NOT NULL,
    run_id VARCHAR NOT NULL,
    state VARCHAR NOT NULL,        -- pass|fail|inconclusive
    n_samples INTEGER NOT NULL,
    n_pass INTEGER NOT NULL,
    log_likelihood_ratio DOUBLE NOT NULL,
    ts BIGINT NOT NULL,
    PRIMARY KEY (archetype_id, candidate, run_id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id VARCHAR PRIMARY KEY,
    archetype_id VARCHAR NOT NULL,
    candidate VARCHAR NOT NULL,
    currency VARCHAR NOT NULL,     -- quota_headroom_pct | usd_per_month
    value DOUBLE NOT NULL,
    pass_rate DOUBLE NOT NULL,
    pass_rate_ci_low DOUBLE NOT NULL,
    pass_rate_ci_high DOUBLE NOT NULL,
    latency_delta_ms INTEGER,
    created_ts BIGINT NOT NULL,
    expires_ts BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS applied_policies (
    id VARCHAR PRIMARY KEY,
    archetype_id VARCHAR NOT NULL,
    candidate VARCHAR NOT NULL,
    writer VARCHAR NOT NULL,       -- claude_code|litellm|nanoclaw
    diff VARCHAR NOT NULL,
    previous_content VARCHAR NOT NULL,  -- full prior file content, for byte-identical revert
    applied_ts BIGINT NOT NULL,
    reverted_ts BIGINT,
    revert_reason VARCHAR
);

CREATE TABLE IF NOT EXISTS spot_checks (
    id VARCHAR PRIMARY KEY,
    archetype_id VARCHAR NOT NULL,
    candidate VARCHAR NOT NULL,
    trace_id VARCHAR NOT NULL,
    result VARCHAR NOT NULL,       -- pass|fail
    ts BIGINT NOT NULL
);
