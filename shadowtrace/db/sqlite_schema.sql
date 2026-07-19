-- Hot-path capture store (SQLite, WAL mode). Append-only. See plan.md §1, R2.
-- Schema version is tracked and advanced by db/migrations.py, not here —
-- setting PRAGMA user_version unconditionally in this file would stomp a
-- migrated version back down every time this idempotent script re-runs.

CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,              -- ulid
    ts INTEGER NOT NULL,              -- epoch ms
    source TEXT NOT NULL,             -- proxy|otel|claude_export|transcript
    tool TEXT,                        -- claude_code|script|chat|codex
    model TEXT NOT NULL,
    request_json TEXT NOT NULL,       -- full message list incl. tool_use blocks (redacted)
    response_json TEXT NOT NULL,      -- full content blocks (redacted)
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms INTEGER,
    cost_usd REAL,                    -- NULL for subscription lane
    session_id TEXT,
    parent_id TEXT,
    redaction_flags TEXT,             -- JSON array: what was scrubbed / quarantined
    quarantined INTEGER NOT NULL DEFAULT 0,
    ingest_key TEXT,                  -- dedup key (transcript msg uuid / export id)
    parser_version TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_traces_ingest_key
    ON traces (source, ingest_key) WHERE ingest_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_traces_ts ON traces (ts);
CREATE INDEX IF NOT EXISTS idx_traces_session ON traces (session_id);
CREATE INDEX IF NOT EXISTS idx_traces_source ON traces (source);

-- Spill ledger: rows written to JSONL spill files when SQLite is unavailable,
-- and their recovery/ETL status (M1.3 fail-open invariant).
CREATE TABLE IF NOT EXISTS spill_files (
    path TEXT PRIMARY KEY,
    created_ts INTEGER NOT NULL,
    recovered INTEGER NOT NULL DEFAULT 0,
    recovered_ts INTEGER
);
