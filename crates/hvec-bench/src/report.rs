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
    /// Payload stage; `none` for runs recorded before the stage existed.
    pub payload: String,
}

/// Aggregated outcome for one cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cell {
    pub key: CellKey,
    pub n: usize,
    /// Questions with gold answers, over which `exact` and `contains` are averaged.
    pub scored: usize,
    pub exact: f64,
    pub contains: f64,
    /// Fraction of questions with a retrieval label whose retrieval hit it. `None` if none.
    pub source_hit: Option<f64>,
    /// Mean fraction of judged-relevant documents retrieved, over questions with judgments.
    pub doc_recall: Option<f64>,
    /// Mean nDCG@k against the judgments, over questions with judgments.
    pub ndcg: Option<f64>,
    pub mean_retrieval_ms: f64,
    pub mean_generation_ms: f64,
    pub mean_input_tokens: f64,
    pub mean_output_tokens: f64,
    pub errors: usize,
    /// Runs whose served model (`metadata.chat_model_served`) differed from the configured one.
    pub served_other: usize,
    /// Runs with a judge verdict (see `judge.rs`).
    pub judged: usize,
    /// Fraction judged correct, over judged runs with gold answers.
    pub judge_correct: Option<f64>,
    /// Fraction judged faithful to the retrieved passages, over judged runs.
    pub judge_faithful: Option<f64>,
}

fn num(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

/// Group `bench` rows into cells. Rows of other kinds are ignored, except
/// `judge` rows, which attach to the bench run named in `metadata.run_id`.
#[must_use]
pub fn aggregate(rows: &[RunRow]) -> Vec<Cell> {
    let verdicts: BTreeMap<&str, &RunRow> = rows
        .iter()
        .filter(|r| r.kind == "judge")
        .filter_map(|r| r.metadata["run_id"].as_str().map(|id| (id, r)))
        .collect();
    let mut groups: BTreeMap<CellKey, Vec<&RunRow>> = BTreeMap::new();
    for r in rows.iter().filter(|r| r.kind == "bench") {
        let key = CellKey {
            question_set: r.metadata["question_set"].as_str().unwrap_or("-").to_owned(),
            collection: r.collection.clone().unwrap_or_else(|| "-".into()),
            embed_model: r.embed_model.clone().unwrap_or_else(|| "-".into()),
            codec: r.codec.clone().unwrap_or_else(|| "-".into()),
            chat_model: r.chat_model.clone().unwrap_or_else(|| "-".into()),
            payload: r.metadata["payload"].as_str().unwrap_or("none").to_owned(),
        };
        groups.entry(key).or_default().push(r);
    }
    groups
        .into_iter()
        .map(|(key, rows)| {
            let n = rows.len();
            let ok: Vec<&RunRow> = rows.iter().copied().filter(|r| r.metadata["error"].is_null()).collect();
            let errors = n - ok.len();
            let has_gold = |r: &RunRow| r.metadata["answers"].as_array().is_some_and(|a| !a.is_empty());
            let gold: Vec<&RunRow> = ok.iter().copied().filter(|r| has_gold(r)).collect();
            let is_true = |v: &serde_json::Value| v.as_bool() == Some(true);
            let mean_of = |items: &[&RunRow], f: &dyn Fn(&RunRow) -> f64| {
                if items.is_empty() {
                    None
                } else {
                    Some(items.iter().map(|r| f(r)).sum::<f64>() / items.len() as f64)
                }
            };
            let frac_of = |items: &[&RunRow], field: &str| {
                mean_of(items, &|r| if is_true(&r.metrics[field]) { 1.0 } else { 0.0 })
            };

            let hinted: Vec<&RunRow> = ok
                .iter()
                .copied()
                .filter(|r| !r.metrics["source_hit"].is_null())
                .collect();
            let labelled: Vec<&RunRow> = ok.iter().copied().filter(|r| !r.metrics["ndcg"].is_null()).collect();

            let judged: Vec<(&RunRow, &RunRow)> = ok
                .iter()
                .filter_map(|r| verdicts.get(r.id.as_str()).map(|v| (*r, *v)))
                .collect();
            let verdict_rows: Vec<&RunRow> = judged.iter().map(|(_, v)| *v).collect();
            let verdict_rows_gold: Vec<&RunRow> = judged.iter().filter(|(r, _)| has_gold(r)).map(|(_, v)| *v).collect();

            Cell {
                key,
                n,
                scored: gold.len(),
                exact: frac_of(&gold, "exact").unwrap_or(0.0),
                contains: frac_of(&gold, "contains").unwrap_or(0.0),
                source_hit: frac_of(&hinted, "source_hit"),
                doc_recall: mean_of(&labelled, &|r| num(&r.metrics["doc_recall"])),
                ndcg: mean_of(&labelled, &|r| num(&r.metrics["ndcg"])),
                mean_retrieval_ms: mean_of(&ok, &|r| num(&r.metrics["retrieval_ms"])).unwrap_or(0.0),
                mean_generation_ms: mean_of(&ok, &|r| num(&r.metrics["generation_ms"])).unwrap_or(0.0),
                mean_input_tokens: mean_of(&ok, &|r| num(&r.metrics["input_tokens"])).unwrap_or(0.0),
                mean_output_tokens: mean_of(&ok, &|r| num(&r.metrics["output_tokens"])).unwrap_or(0.0),
                errors,
                served_other: ok
                    .iter()
                    .filter(|r| {
                        r.metadata["chat_model_served"]
                            .as_str()
                            .is_some_and(|s| Some(s) != r.chat_model.as_deref())
                    })
                    .count(),
                judged: judged.len(),
                judge_correct: frac_of(&verdict_rows_gold, "correct"),
                judge_faithful: frac_of(&verdict_rows, "faithful"),
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
        r.metadata = serde_json::json!({ "question_set": set, "answers": ["gold"] });
        r.metrics = serde_json::json!({ "exact": exact, "contains": contains, "source_hit": hit, "retrieval_ms": 2, "generation_ms": gen_ms, "input_tokens": 100, "output_tokens": 10 });
        r
    }

    fn verdict(run: &RunRow, correct: bool, faithful: bool) -> RunRow {
        let mut v = RunRow::new("judge");
        v.metadata = serde_json::json!({ "run_id": run.id });
        v.metrics = serde_json::json!({ "correct": correct, "faithful": faithful });
        v
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
        assert_eq!(f32.ndcg, None);
        assert_eq!(f32.scored, 2);
        assert_eq!(f32.judged, 0);
        assert_eq!(f32.judge_correct, None);
        assert_eq!(f32.key.payload, "none");
    }

    #[test]
    fn payload_splits_cells() {
        let a = row("qa", "f32", "opus", true, true, None, 1);
        let mut b = row("qa", "f32", "opus", true, true, None, 1);
        b.metadata["payload"] = serde_json::json!("truncate:40");
        let cells = aggregate(&[a, b]);
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].key.payload, "none");
        assert_eq!(cells[1].key.payload, "truncate:40");
    }

    #[test]
    fn accuracy_is_averaged_over_questions_with_gold_answers() {
        let mut no_gold = row("qa", "f32", "opus", false, false, Some(true), 1);
        no_gold.metadata["answers"] = serde_json::json!([]);
        let gold = row("qa", "f32", "opus", true, true, None, 1);
        let cells = aggregate(&[no_gold, gold]);
        assert_eq!(cells[0].n, 2);
        assert_eq!(cells[0].scored, 1);
        assert_eq!(cells[0].exact, 1.0);
        assert_eq!(cells[0].source_hit, Some(1.0));
    }

    #[test]
    fn judge_rows_attach_to_their_bench_run() {
        let a = row("qa", "f32", "opus", true, true, None, 1);
        let b = row("qa", "f32", "opus", false, false, None, 1);
        let mut c = row("qa", "f32", "opus", false, false, None, 1);
        c.metadata["answers"] = serde_json::json!([]);
        let va = verdict(&a, true, true);
        let vb = verdict(&b, false, true);
        let vc = verdict(&c, true, false);
        let cells = aggregate(&[a, b, c, va, vb, vc]);
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].judged, 3);
        // Correctness over the two judged runs with gold; faithfulness over all three.
        assert_eq!(cells[0].judge_correct, Some(0.5));
        assert!((cells[0].judge_faithful.unwrap() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn judged_rows_average_ndcg_and_doc_recall() {
        let mut a = row("qa", "f32", "opus", true, true, Some(true), 1);
        a.metrics["ndcg"] = serde_json::json!(1.0);
        a.metrics["doc_recall"] = serde_json::json!(0.5);
        let mut b = row("qa", "f32", "opus", true, true, Some(false), 1);
        b.metrics["ndcg"] = serde_json::json!(0.0);
        b.metrics["doc_recall"] = serde_json::json!(0.0);
        let unjudged = row("qa", "f32", "opus", true, true, None, 1);
        let cells = aggregate(&[a, b, unjudged]);
        assert_eq!(cells[0].ndcg, Some(0.5));
        assert_eq!(cells[0].doc_recall, Some(0.25));
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
