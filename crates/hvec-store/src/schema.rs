//! Schema. Idempotent so `Store::open` can run it every time.
//!
//! Columns added after the first release are listed in [`MIGRATIONS`] and
//! applied when missing, so an older database file keeps working.

pub(crate) const DDL: &str = r#"
CREATE TABLE IF NOT EXISTS collections (
    name          TEXT PRIMARY KEY,
    embed_profile TEXT NOT NULL,
    embed_model   TEXT NOT NULL,
    dimension     INTEGER NOT NULL,
    codec         TEXT NOT NULL,
    metric        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    codec_params  BLOB
);

CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    collection TEXT NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
    source     TEXT NOT NULL,
    ordinal    INTEGER NOT NULL,
    text       TEXT NOT NULL,
    vector     BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_collection ON chunks(collection);

CREATE TABLE IF NOT EXISTS sessions (
    name         TEXT PRIMARY KEY,
    collection   TEXT,
    chat_profile TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session    TEXT NOT NULL REFERENCES sessions(name) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_session ON messages(session);

CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    kind          TEXT NOT NULL,
    collection    TEXT,
    chat_profile  TEXT,
    chat_model    TEXT,
    embed_profile TEXT,
    embed_model   TEXT,
    codec         TEXT,
    metric        TEXT,
    metadata      TEXT NOT NULL,
    metrics       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_by_created ON runs(created_at);
"#;

/// `(table, column, ALTER TABLE statement)` for columns that older databases lack.
pub(crate) const MIGRATIONS: &[(&str, &str, &str)] = &[(
    "collections",
    "codec_params",
    "ALTER TABLE collections ADD COLUMN codec_params BLOB",
)];
