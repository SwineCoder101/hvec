//! Aggregate `bench` rows from the run log into matrix cells.

use std::collections::BTreeMap;

use hvec_store::RunRow;
use serde::{Deserialize, Serialize};

/// One cell of the matrix: everything that was held constant.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CellKey {
    pub question_set: String,
    pub collection: String,
    pub embed_model: String,
    pub codec: String,
    pub chat_model: String,
}

/// Aggregated outcome for one cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cell {
    pub key: CellKey,
    pub n: usize,
    pub exact: f64,
    pub contains: f64,
    /// Fraction of questions with a source hint whose retrieval hit it. `None` if no hints.
    pub source_hit: Option<f64>,
    pub mean_retrieval_ms: f64,
    pub mean_generation_ms: f64,
    pub mean_input_tokens: f64,
    pub mean_output_tokens: f64,
    pub errors: usize,
    /// Runs whose served model (`metadata.chat_model_served`) differed from the configured one.
    pub served_other: usize,
}

fn num(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

/// Group `bench` rows into cells. Rows of other kinds are ignored.
#[must_use]
pub fn aggregate(rows: &[RunRow]) -> Vec<Cell> {
    let mut groups: BTreeMap<CellKey, Vec<&RunRow>> = BTreeMap::new();
    for r in rows.iter().filter(|r| r.kind == "bench") {
        let key = CellKey {
            question_set: r.metadata["question_set"].as_str().unwrap_or("-").to_owned(),
            collection: r.collection.clone().unwrap_or_else(|| "-".into()),
            embed_model: r.embed_model.clone().unwrap_or_else(|| "-".into()),
            codec: r.codec.clone().unwrap_or_else(|| "-".into()),
            chat_model: r.chat_model.clone().unwrap_or_else(|| "-".into()),
        };
        groups.entry(key).or_default().push(r);
    }
    groups
        .into_iter()
        .map(|(key, rows)| {
            let n = rows.len();
            let ok: Vec<&&RunRow> = rows.iter().filter(|r| r.metadata["error"].is_null()).collect();
            let errors = n - ok.len();
            let mean = |f: &dyn Fn(&RunRow) -> f64| {
                if ok.is_empty() {
                    0.0
                } else {
                    ok.iter().map(|r| f(r)).sum::<f64>() / ok.len() as f64
                }
            };
            let hinted: Vec<&&&RunRow> = ok.iter().filter(|r| !r.metrics["source_hit"].is_null()).collect();
            let source_hit = if hinted.is_empty() {
                None
            } else {
                Some(
                    hinted
                        .iter()
                        .filter(|r| r.metrics["source_hit"].as_bool() == Some(true))
                        .count() as f64
                        / hinted.len() as f64,
                )
            };
            Cell {
                key,
                n,
                exact: mean(&|r| {
                    if r.metrics["exact"].as_bool() == Some(true) {
                        1.0
                    } else {
                        0.0
                    }
                }),
                contains: mean(&|r| {
                    if r.metrics["contains"].as_bool() == Some(true) {
                        1.0
                    } else {
                        0.0
                    }
                }),
                source_hit,
                mean_retrieval_ms: mean(&|r| num(&r.metrics["retrieval_ms"])),
                mean_generation_ms: mean(&|r| num(&r.metrics["generation_ms"])),
                mean_input_tokens: mean(&|r| num(&r.metrics["input_tokens"])),
                mean_output_tokens: mean(&|r| num(&r.metrics["output_tokens"])),
                errors,
                served_other: ok
                    .iter()
                    .filter(|r| {
                        r.metadata["chat_model_served"]
                            .as_str()
                            .is_some_and(|s| Some(s) != r.chat_model.as_deref())
                    })
                    .count(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(set: &str, codec: &str, chat: &str, exact: bool, contains: bool, hit: Option<bool>, gen_ms: u64) -> RunRow {
        let mut r = RunRow::new("bench");
        r.collection = Some(format!("c-{codec}"));
        r.embed_model = Some("bge".into());
        r.codec = Some(codec.into());
        r.chat_model = Some(chat.into());
        r.metadata = serde_json::json!({ "question_set": set });
        r.metrics = serde_json::json!({ "exact": exact, "contains": contains, "source_hit": hit, "retrieval_ms": 2, "generation_ms": gen_ms, "input_tokens": 100, "output_tokens": 10 });
        r
    }

    #[test]
    fn aggregates_per_cell_and_ignores_other_kinds() {
        let mut rows = vec![
            row("qa", "f32", "opus", true, true, Some(true), 100),
            row("qa", "f32", "opus", false, true, Some(false), 300),
            row("qa", "binary", "opus", false, false, None, 50),
            RunRow::new("query"),
        ];
        rows[2].metadata["error"] = serde_json::Value::Null;
        let cells = aggregate(&rows);
        assert_eq!(cells.len(), 2);
        let f32 = cells.iter().find(|c| c.key.codec == "f32").unwrap();
        assert_eq!(f32.n, 2);
        assert_eq!(f32.exact, 0.5);
        assert_eq!(f32.contains, 1.0);
        assert_eq!(f32.source_hit, Some(0.5));
        assert_eq!(f32.mean_generation_ms, 200.0);
        let bin = cells.iter().find(|c| c.key.codec == "binary").unwrap();
        assert_eq!(bin.source_hit, None);
        assert_eq!(bin.contains, 0.0);
        assert_eq!(f32.served_other, 0);
    }

    #[test]
    fn counts_runs_served_by_another_model() {
        let mut r = row("qa", "f32", "opus", true, true, None, 1);
        r.metadata["chat_model_served"] = serde_json::json!("opus-fallback");
        let same = row("qa", "f32", "opus", true, true, None, 1);
        let cells = aggregate(&[r, same]);
        assert_eq!(cells[0].served_other, 1);
    }

    #[test]
    fn errored_rows_count_but_do_not_score() {
        let mut bad = row("qa", "f32", "opus", false, false, None, 0);
        bad.metadata["error"] = serde_json::json!("boom");
        let good = row("qa", "f32", "opus", true, true, None, 10);
        let cells = aggregate(&[bad, good]);
        assert_eq!(cells[0].n, 2);
        assert_eq!(cells[0].errors, 1);
        assert_eq!(cells[0].exact, 1.0);
    }
}
