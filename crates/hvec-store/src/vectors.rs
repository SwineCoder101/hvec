//! Collections of encoded vectors and brute-force compressed-domain search.

use hvec_core::{Codec, Encoded, Metric};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::{Result, Store, StoreError};

/// Metadata for a collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Collection {
    pub name: String,
    pub embed_profile: String,
    pub embed_model: String,
    pub dimension: usize,
    pub codec: String,
    pub metric: Metric,
    pub created_at: String,
    pub chunk_count: u64,
}

/// What is needed to create a collection.
#[derive(Debug, Clone)]
pub struct NewCollection<'a> {
    pub name: &'a str,
    pub embed_profile: &'a str,
    pub embed_model: &'a str,
    pub dimension: usize,
    pub codec: &'a str,
    pub metric: Metric,
}

/// A chunk ready to insert.
#[derive(Debug, Clone)]
pub struct NewChunk {
    pub source: String,
    pub ordinal: u32,
    pub text: String,
    pub vector: Encoded,
}

/// One search result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hit {
    pub chunk_id: i64,
    pub source: String,
    pub ordinal: u32,
    pub text: String,
    pub score: f32,
}

impl Store {
    pub fn create_collection(&self, new: &NewCollection<'_>) -> Result<()> {
        let inserted = self.conn.execute(
            "INSERT OR IGNORE INTO collections (name, embed_profile, embed_model, dimension, codec, metric, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                new.name,
                new.embed_profile,
                new.embed_model,
                new.dimension as i64,
                new.codec,
                new.metric.name(),
                Self::now()
            ],
        )?;
        if inserted == 0 {
            return Err(StoreError::CollectionExists(new.name.to_owned()));
        }
        Ok(())
    }

    pub fn get_collection(&self, name: &str) -> Result<Option<Collection>> {
        self.conn
            .query_row(
                "SELECT c.name, c.embed_profile, c.embed_model, c.dimension, c.codec, c.metric, c.created_at,
                        (SELECT COUNT(*) FROM chunks WHERE collection = c.name)
                 FROM collections c WHERE c.name = ?1",
                params![name],
                row_to_collection,
            )
            .optional()
            .map_err(Into::into)
    }

    /// Fetch a collection, or fail if it does not exist.
    pub fn require_collection(&self, name: &str) -> Result<Collection> {
        self.get_collection(name)?
            .ok_or_else(|| StoreError::NoSuchCollection(name.to_owned()))
    }

    /// Fetch or create a collection, refusing to mix embedding models.
    pub fn ensure_collection(&self, new: &NewCollection<'_>) -> Result<Collection> {
        if let Some(existing) = self.get_collection(new.name)? {
            if existing.embed_model != new.embed_model || existing.dimension != new.dimension {
                return Err(StoreError::EmbedderMismatch {
                    collection: new.name.to_owned(),
                    existing_model: existing.embed_model,
                    existing_dim: existing.dimension,
                    new_model: new.embed_model.to_owned(),
                    new_dim: new.dimension,
                });
            }
            return Ok(existing);
        }
        self.create_collection(new)?;
        self.require_collection(new.name)
    }

    pub fn list_collections(&self) -> Result<Vec<Collection>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.name, c.embed_profile, c.embed_model, c.dimension, c.codec, c.metric, c.created_at,
                    (SELECT COUNT(*) FROM chunks WHERE collection = c.name)
             FROM collections c ORDER BY c.name",
        )?;
        let rows = stmt.query_map([], row_to_collection)?;
        rows.collect::<std::result::Result<_, _>>().map_err(Into::into)
    }

    pub fn delete_collection(&self, name: &str) -> Result<()> {
        let n = self
            .conn
            .execute("DELETE FROM collections WHERE name = ?1", params![name])?;
        if n == 0 {
            return Err(StoreError::NoSuchCollection(name.to_owned()));
        }
        Ok(())
    }

    /// Insert chunks in one transaction. Returns how many were written.
    pub fn insert_chunks(&mut self, collection: &str, chunks: &[NewChunk]) -> Result<usize> {
        self.require_collection(collection)?;
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO chunks (collection, source, ordinal, text, vector) VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            for c in chunks {
                stmt.execute(params![collection, c.source, c.ordinal, c.text, c.vector.0])?;
            }
        }
        tx.commit()?;
        Ok(chunks.len())
    }

    /// Brute-force top-k search scored through the codec.
    ///
    /// Scores every stored vector against `query` in the compressed domain,
    /// then keeps the best `k` according to the metric's direction.
    pub fn search(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
        codec: &dyn Codec,
        metric: Metric,
    ) -> Result<Vec<Hit>> {
        self.require_collection(collection)?;
        let mut stmt = self
            .conn
            .prepare_cached("SELECT id, source, ordinal, text, vector FROM chunks WHERE collection = ?1")?;
        let rows = stmt.query_map(params![collection], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, u32>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Vec<u8>>(4)?,
            ))
        })?;

        let mut hits = Vec::new();
        for row in rows {
            let (chunk_id, source, ordinal, text, blob) = row?;
            let score = codec.score(query, &Encoded(blob), metric)?;
            hits.push(Hit {
                chunk_id,
                source,
                ordinal,
                text,
                score,
            });
        }
        if metric.higher_is_better() {
            hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        } else {
            hits.sort_by(|a, b| a.score.total_cmp(&b.score));
        }
        hits.truncate(k);
        Ok(hits)
    }
}

fn row_to_collection(row: &rusqlite::Row<'_>) -> rusqlite::Result<Collection> {
    let metric_name: String = row.get(5)?;
    let metric = Metric::parse(&metric_name)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(5, rusqlite::types::Type::Text, Box::new(e)))?;
    Ok(Collection {
        name: row.get(0)?,
        embed_profile: row.get(1)?,
        embed_model: row.get(2)?,
        dimension: row.get::<_, i64>(3)? as usize,
        codec: row.get(4)?,
        metric,
        created_at: row.get(6)?,
        chunk_count: row.get::<_, i64>(7)? as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hvec_core::codecs::F32Codec;

    fn store_with_collection() -> (Store, F32Codec) {
        let store = Store::open_in_memory().unwrap();
        store
            .create_collection(&NewCollection {
                name: "docs",
                embed_profile: "local",
                embed_model: "test-model",
                dimension: 2,
                codec: "f32",
                metric: Metric::Cosine,
            })
            .unwrap();
        (store, F32Codec::new(2))
    }

    #[test]
    fn insert_and_search_returns_nearest_first() {
        let (mut store, codec) = store_with_collection();
        let chunks = vec![
            NewChunk {
                source: "a".into(),
                ordinal: 0,
                text: "east".into(),
                vector: codec.encode(&[1.0, 0.0]).unwrap(),
            },
            NewChunk {
                source: "a".into(),
                ordinal: 1,
                text: "north".into(),
                vector: codec.encode(&[0.0, 1.0]).unwrap(),
            },
            NewChunk {
                source: "a".into(),
                ordinal: 2,
                text: "ne".into(),
                vector: codec.encode(&[0.7, 0.7]).unwrap(),
            },
        ];
        assert_eq!(store.insert_chunks("docs", &chunks).unwrap(), 3);

        let hits = store.search("docs", &[1.0, 0.1], 2, &codec, Metric::Cosine).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].text, "east");
        assert_eq!(hits[1].text, "ne");

        let hits = store.search("docs", &[0.0, 1.0], 1, &codec, Metric::L2).unwrap();
        assert_eq!(hits[0].text, "north");
        assert_eq!(hits[0].score, 0.0);
    }

    #[test]
    fn ensure_collection_rejects_different_embedder() {
        let (store, _) = store_with_collection();
        let err = store
            .ensure_collection(&NewCollection {
                name: "docs",
                embed_profile: "x",
                embed_model: "other-model",
                dimension: 2,
                codec: "f32",
                metric: Metric::Cosine,
            })
            .unwrap_err();
        assert!(matches!(err, StoreError::EmbedderMismatch { .. }));
    }

    #[test]
    fn delete_cascades_to_chunks() {
        let (mut store, codec) = store_with_collection();
        store
            .insert_chunks(
                "docs",
                &[NewChunk {
                    source: "a".into(),
                    ordinal: 0,
                    text: "x".into(),
                    vector: codec.encode(&[1.0, 0.0]).unwrap(),
                }],
            )
            .unwrap();
        store.delete_collection("docs").unwrap();
        assert!(store.get_collection("docs").unwrap().is_none());
        let n: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }
}
