//! LLM-as-judge scoring of recorded answers.
//!
//! A cheaper model grades each answer twice: *correctness* against the gold
//! answers, when the question has any, and *faithfulness* to the passages the
//! answer was generated from. The judge prompt is versioned like the RAG
//! prompt; bump [`JUDGE_PROMPT_VERSION`] when it changes.
//!
//! [`agreement`] compares the judge's correctness verdict with the cheap
//! `exact` and `contains` rules, which is the check to run before trusting
//! the judge on a new question set.

use std::collections::BTreeMap;

use hvec_store::RunRow;
use serde::{Deserialize, Serialize};

/// Recorded with every verdict so prompt changes are visible in the log.
pub const JUDGE_PROMPT_VERSION: &str = "v1";

/// Operator instructions for the judge.
pub const JUDGE_SYSTEM_PROMPT: &str = "You are a strict grader of answers produced by a retrieval-augmented system. \
You will see a question, the numbered passages the system retrieved, the gold answers when there are any, \
and the system's answer. Judge two things. CORRECT: does the answer convey the same fact as a gold answer? \
Wording may differ; hedging, contradiction or a refusal to answer is not correct. When there are no gold \
answers, output null for correct. FAITHFUL: is every factual claim in the answer supported by the passages? \
An answer that says the passages do not contain the answer is faithful if that is true. \
Reply with one JSON object only, no prose: {\"correct\": true|false|null, \"faithful\": true|false, \"reason\": \"<one sentence>\"}";

/// Build the user turn for the judge.
#[must_use]
pub fn build_judge_prompt(question: &str, gold: &[String], passages: &[String], answer: &str) -> String {
    let mut s = String::new();
    s.push_str("Question: ");
    s.push_str(question.trim());
    s.push_str("\n\nRetrieved passages:\n");
    if passages.is_empty() {
        s.push_str("(none)\n");
    }
    for (i, p) in passages.iter().enumerate() {
        s.push_str(&format!("[{}] {}\n", i + 1, p.trim()));
    }
    s.push_str("\nGold answers: ");
    if gold.is_empty() {
        s.push_str("(none; output null for correct)");
    } else {
        s.push_str(&gold.iter().map(|g| format!("\"{g}\"")).collect::<Vec<_>>().join(", "));
    }
    s.push_str("\n\nSystem answer: ");
    s.push_str(answer.trim());
    s.push_str("\n\nReply with the JSON object only.");
    s
}

/// What the judge decided.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verdict {
    /// `None` when the question had no gold answers.
    pub correct: Option<bool>,
    pub faithful: bool,
    #[serde(default)]
    pub reason: String,
}

/// Why a judge reply could not be used.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VerdictError {
    #[error("judge reply contains no JSON object: {0:?}")]
    NoJson(String),
    #[error("judge reply is not a verdict: {0}")]
    Shape(String),
}

/// Parse the judge's reply, tolerating prose or code fences around the JSON.
pub fn parse_verdict(text: &str) -> Result<Verdict, VerdictError> {
    let start = text.find('{').ok_or_else(|| VerdictError::NoJson(short(text)))?;
    let end = text.rfind('}').ok_or_else(|| VerdictError::NoJson(short(text)))?;
    if end < start {
        return Err(VerdictError::NoJson(short(text)));
    }
    let v: serde_json::Value =
        serde_json::from_str(&text[start..=end]).map_err(|e| VerdictError::Shape(e.to_string()))?;
    let faithful = v
        .get("faithful")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| VerdictError::Shape("`faithful` must be true or false".into()))?;
    let correct = match v.get("correct") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Bool(b)) => Some(*b),
        Some(other) => {
            return Err(VerdictError::Shape(format!(
                "`correct` must be a bool or null, got {other}"
            )));
        }
    };
    Ok(Verdict {
        correct,
        faithful,
        reason: v.get("reason").and_then(|r| r.as_str()).unwrap_or("").to_owned(),
    })
}

fn short(s: &str) -> String {
    s.chars().take(120).collect()
}

/// How the judge's correctness verdict lines up with the cheap rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Agreement {
    /// Judged bench runs whose question had gold answers.
    pub n: usize,
    pub judge_true: usize,
    pub exact_true: usize,
    pub contains_true: usize,
    /// Fraction of runs where the judge and `exact` give the same verdict.
    pub agree_exact: f64,
    /// Fraction of runs where the judge and `contains` give the same verdict.
    pub agree_contains: f64,
    /// Judge says correct, `contains` says wrong: the rule missed a paraphrase.
    pub judge_only: usize,
    /// `contains` says correct, judge says wrong: the rule matched a substring the judge rejects.
    pub contains_only: usize,
}

/// Compare judge verdicts with `exact` and `contains` over the bench runs
/// they attach to. Runs without gold answers are left out.
#[must_use]
pub fn agreement(rows: &[RunRow]) -> Agreement {
    let verdicts: BTreeMap<&str, &RunRow> = rows
        .iter()
        .filter(|r| r.kind == "judge")
        .filter_map(|r| r.metadata["run_id"].as_str().map(|id| (id, r)))
        .collect();
    let mut a = Agreement {
        n: 0,
        judge_true: 0,
        exact_true: 0,
        contains_true: 0,
        agree_exact: 0.0,
        agree_contains: 0.0,
        judge_only: 0,
        contains_only: 0,
    };
    let (mut same_exact, mut same_contains) = (0usize, 0usize);
    for r in rows.iter().filter(|r| r.kind == "bench") {
        let Some(v) = verdicts.get(r.id.as_str()) else { continue };
        let Some(judge) = v.metrics["correct"].as_bool() else {
            continue;
        };
        let exact = r.metrics["exact"].as_bool() == Some(true);
        let contains = r.metrics["contains"].as_bool() == Some(true);
        a.n += 1;
        a.judge_true += usize::from(judge);
        a.exact_true += usize::from(exact);
        a.contains_true += usize::from(contains);
        same_exact += usize::from(judge == exact);
        same_contains += usize::from(judge == contains);
        a.judge_only += usize::from(judge && !contains);
        a.contains_only += usize::from(contains && !judge);
    }
    if a.n > 0 {
        a.agree_exact = same_exact as f64 / a.n as f64;
        a.agree_contains = same_contains as f64 / a.n as f64;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_lists_passages_gold_and_answer() {
        let p = build_judge_prompt(
            "refund?",
            &["30 days".into()],
            &["Refunds within 30 days.".into(), "Shipping.".into()],
            "Thirty days [1].",
        );
        assert!(p.starts_with("Question: refund?"));
        assert!(p.contains("[1] Refunds within 30 days.\n[2] Shipping."));
        assert!(p.contains("Gold answers: \"30 days\""));
        assert!(p.contains("System answer: Thirty days [1]."));
        let none = build_judge_prompt("q", &[], &[], "a");
        assert!(none.contains("(none; output null for correct)") && none.contains("(none)\n"));
    }

    #[test]
    fn parses_verdicts_with_and_without_decoration() {
        let v = parse_verdict("{\"correct\": true, \"faithful\": false, \"reason\": \"x\"}").unwrap();
        assert_eq!(
            v,
            Verdict {
                correct: Some(true),
                faithful: false,
                reason: "x".into()
            }
        );
        let v = parse_verdict("Sure:\n```json\n{\"correct\": null, \"faithful\": true}\n```\n").unwrap();
        assert_eq!(v.correct, None);
        assert!(v.faithful && v.reason.is_empty());
        assert!(matches!(parse_verdict("no json here"), Err(VerdictError::NoJson(_))));
        assert!(matches!(
            parse_verdict("{\"correct\": true}"),
            Err(VerdictError::Shape(_))
        ));
        assert!(matches!(
            parse_verdict("{\"correct\": \"yes\", \"faithful\": true}"),
            Err(VerdictError::Shape(_))
        ));
        assert!(matches!(parse_verdict("} {"), Err(VerdictError::NoJson(_))));
    }

    fn bench(exact: bool, contains: bool) -> RunRow {
        let mut r = RunRow::new("bench");
        r.metadata = serde_json::json!({ "answers": ["g"] });
        r.metrics = serde_json::json!({ "exact": exact, "contains": contains });
        r
    }

    fn verdict(run: &RunRow, correct: Option<bool>) -> RunRow {
        let mut v = RunRow::new("judge");
        v.metadata = serde_json::json!({ "run_id": run.id });
        v.metrics = serde_json::json!({ "correct": correct, "faithful": true });
        v
    }

    #[test]
    fn agreement_counts_the_disagreements_both_ways() {
        let a = bench(true, true); // judge agrees
        let b = bench(false, true); // contains matched, judge rejects
        let c = bench(false, false); // judge accepts a paraphrase
        let d = bench(false, false); // no gold: skipped
        let rows = vec![
            verdict(&a, Some(true)),
            verdict(&b, Some(false)),
            verdict(&c, Some(true)),
            verdict(&d, None),
            a,
            b,
            c,
            d,
        ];
        let g = agreement(&rows);
        assert_eq!(g.n, 3);
        assert_eq!(g.judge_true, 2);
        assert_eq!(g.exact_true, 1);
        assert_eq!(g.contains_true, 2);
        assert!((g.agree_contains - 1.0 / 3.0).abs() < 1e-9);
        assert!((g.agree_exact - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(g.judge_only, 1);
        assert_eq!(g.contains_only, 1);
        assert_eq!(agreement(&[]).n, 0);
    }
}
