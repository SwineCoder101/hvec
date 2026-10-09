//! SQLite-backed storage: vector collections, chat sessions and the run log.
//!
//! Everything lives in one file so a benchmark run, the context it retrieved
//! and the conversation it belonged to can be joined later. Vectors are
//! stored in their codec-encoded form and scored in the compressed domain
//! by brute-force scan. That is deliberate: the point of hvec is to measure
//! the codec, not an index.

mod runs;
mod schema;
mod sessions;
mod vectors;

use std::path::Path;

use rusqlite::Connection;

pub use runs::RunRow;
pub use sessions::{Session, StoredMessage};
pub use vectors::{Collection, Hit, NewChunk, NewCollection};

/// Convenience alias.
pub type Result<T, E = StoreError> = std::result::Result<T, E>;

/// Storage errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StoreError {
    #[error("could not create directory {path}")]
    CreateDir {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("sqlite error")]
    Sqlite(#[from] rusqlite::Error),
    #[error("collection `{0}` does not exist")]
    NoSuchCollection(String),
    #[error("collection `{0}` already exists")]
    CollectionExists(String),
    #[error(
        "collection `{collection}` was built with {existing_model} ({existing_dim} dims); refusing to mix in {new_model} ({new_dim} dims)"
    )]
    EmbedderMismatch {
        collection: String,
        existing_model: String,
        existing_dim: usize,
        new_model: String,
        new_dim: usize,
    },
    #[error("session `{0}` does not exist")]
    NoSuchSession(String),
    #[error("run `{0}` does not exist")]
    NoSuchRun(String),
    #[error("codec error")]
    Codec(#[from] hvec_core::CodecError),
    #[error("json error")]
    Json(#[from] serde_json::Error),
}

/// Handle to the database. Not `Sync`; open one per task.
#[derive(Debug)]
pub struct Store {
    conn: Connection,
}

impl Store {
    /// Open or create the database at `path`, creating parent directories.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| StoreError::CreateDir {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    /// An in-memory database, for tests.
    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON;")?;
        conn.execute_batch(schema::DDL)?;
        for (table, column, alter) in schema::MIGRATIONS {
            if !Self::has_column(&conn, table, column)? {
                conn.execute_batch(alter)?;
            }
        }
        Ok(Self { conn })
    }

    fn has_column(conn: &Connection, table: &str, column: &str) -> Result<bool> {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let names = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for name in names {
            if name? == column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) fn now() -> String {
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
    }
}
