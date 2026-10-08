//! The run log. Every query or benchmark cell appends one row.

use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::{Result, Store, StoreError};

/// One recorded run. `metadata` describes the setup, `metrics` the outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRow {
    pub id: String,
    pub created_at: String,
    /// `query`, `chat`, or a benchmark kind added later.
    pub kind: String,
    pub collection: Option<String>,
    pub chat_profile: Option<String>,
    pub chat_model: Option<String>,
    pub embed_profile: Option<String>,
    pub embed_model: Option<String>,
    pub codec: Option<String>,
    pub metric: Option<String>,
    pub metadata: serde_json::Value,
    pub metrics: serde_json::Value,
}

impl RunRow {
    /// A fresh row with a random id and the current time.
    #[must_use]
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            created_at: Store::now(),
            kind: kind.into(),
            collection: None,
            chat_profile: None,
            chat_model: None,
            embed_profile: None,
            embed_model: None,
            codec: None,
            metric: None,
            metadata: serde_json::Value::Object(Default::default()),
            metrics: serde_json::Value::Object(Default::default()),
        }
    }
}

impl Store {
    pub fn insert_run(&self, run: &RunRow) -> Result<()> {
        self.conn.execute(
            "INSERT INTO runs (id, created_at, kind, collection, chat_profile, chat_model, embed_profile, embed_model,
                               codec, metric, metadata, metrics)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                run.id,
                run.created_at,
                run.kind,
                run.collection,
                run.chat_profile,
                run.chat_model,
                run.embed_profile,
                run.embed_model,
                run.codec,
                run.metric,
                serde_json::to_string(&run.metadata)?,
                serde_json::to_string(&run.metrics)?,
            ],
        )?;
        Ok(())
    }

    /// Most recent runs first.
    pub fn list_runs(&self, limit: usize) -> Result<Vec<RunRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, created_at, kind, collection, chat_profile, chat_model, embed_profile, embed_model, codec, metric,
                    metadata, metrics
             FROM runs ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], row_to_run)?;
        rows.collect::<std::result::Result<_, _>>().map_err(Into::into)
    }

    /// Look up by full id or unique prefix.
    pub fn get_run(&self, id_or_prefix: &str) -> Result<RunRow> {
        let pattern = format!("{id_or_prefix}%");
        self.conn
            .query_row(
                "SELECT id, created_at, kind, collection, chat_profile, chat_model, embed_profile, embed_model, codec, metric,
                        metadata, metrics
                 FROM runs WHERE id LIKE ?1 ORDER BY created_at DESC LIMIT 1",
                params![pattern],
                row_to_run,
            )
            .optional()?
            .ok_or_else(|| StoreError::NoSuchRun(id_or_prefix.to_owned()))
    }
}

fn row_to_run(row: &rusqlite::Row<'_>) -> rusqlite::Result<RunRow> {
    let parse = |idx: usize| -> rusqlite::Result<serde_json::Value> {
        let s: String = row.get(idx)?;
        serde_json::from_str(&s)
            .map_err(|e| rusqlite::Error::FromSqlConversionFailure(idx, rusqlite::types::Type::Text, Box::new(e)))
    };
    Ok(RunRow {
        id: row.get(0)?,
        created_at: row.get(1)?,
        kind: row.get(2)?,
        collection: row.get(3)?,
        chat_profile: row.get(4)?,
        chat_model: row.get(5)?,
        embed_profile: row.get(6)?,
        embed_model: row.get(7)?,
        codec: row.get(8)?,
        metric: row.get(9)?,
        metadata: parse(10)?,
        metrics: parse(11)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_round_trips_with_json() {
        let store = Store::open_in_memory().unwrap();
        let mut run = RunRow::new("query");
        run.codec = Some("f32".into());
        run.metrics = serde_json::json!({ "retrieval_ms": 3, "k": 5 });
        store.insert_run(&run).unwrap();
        let back = store.get_run(&run.id[..8]).unwrap();
        assert_eq!(back, run);
        assert_eq!(store.list_runs(10).unwrap().len(), 1);
    }
}
