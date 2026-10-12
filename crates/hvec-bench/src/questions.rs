//! Question sets with gold answers and relevance judgments, and the cheap
//! scoring rules.
//!
//! Format: JSON Lines, one object per question.
//!
//! ```json
//! {"id": "q1", "question": "What is the refund window?", "answers": ["30 days", "thirty days"], "source": "refunds.md"}
//! {"id": "3", "question": "Does the evidence support or refute: ...", "query": "...", "answers": ["supports"], "relevant": ["14717"]}
//! ```
//!
//! `answers` lists every acceptable surface form. `source` is an optional
//! substring that a retrieved chunk's source path should contain. `relevant`
//! lists document ids judged relevant (BEIR qrels); a chunk belongs to a
//! document when the file stem of its source path equals the id. `query`
//! optionally replaces `question` as the text that is embedded for retrieval,
//! so a question can be phrased for the chat model without changing the
//! retrieval query.

use std::collections::HashSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// One question with its acceptable answers and relevance judgments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Question {
    pub id: String,
    pub question: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    #[serde(default)]
    pub answers: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relevant: Vec<String>,
}

impl Question {
    /// The text embedded for retrieval.
    #[must_use]
    pub fn retrieval_text(&self) -> &str {
        self.query.as_deref().unwrap_or(&self.question)
    }

    /// Whether the question carries any retrieval label at all.
    #[must_use]
    pub fn has_retrieval_label(&self) -> bool {
        self.source.is_some() || !self.relevant.is_empty()
    }
}

/// Errors loading a question set.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum QuestionError {
    #[error("could not read {path}")]
    Read {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}:{line}: invalid question")]
    Parse {
        path: std::path::PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("{path}: duplicate question id `{id}`")]
    DuplicateId { path: std::path::PathBuf, id: String },
    #[error("{path}: no questions")]
    Empty { path: std::path::PathBuf },
}

/// Load a JSONL question set. Blank lines and `#` comments are skipped.
pub fn load(path: &Path) -> Result<Vec<Question>, QuestionError> {
    let text = std::fs::read_to_string(path).map_err(|source| QuestionError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    parse(&text, path)
}

/// Parse JSONL text. `path` is only used in error messages.
pub fn parse(text: &str, path: &Path) -> Result<Vec<Question>, QuestionError> {
    let mut out: Vec<Question> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let q: Question = serde_json::from_str(line).map_err(|source| QuestionError::Parse {
            path: path.to_path_buf(),
            line: i + 1,
            source,
        })?;
        if !seen.insert(q.id.clone()) {
            return Err(QuestionError::DuplicateId {
                path: path.to_path_buf(),
                id: q.id,
            });
        }
        out.push(q);
    }
    if out.is_empty() {
        return Err(QuestionError::Empty {
            path: path.to_path_buf(),
        });
    }
    Ok(out)
}

/// Serialise questions as JSONL, one per line, with a trailing newline.
#[must_use]
pub fn to_jsonl(questions: &[Question]) -> String {
    let mut s = String::new();
    for q in questions {
        s.push_str(&serde_json::to_string(q).expect("Question serialises to JSON"));
        s.push('\n');
    }
    s
}

/// Lowercase, collapse whitespace, strip punctuation and leading articles.
#[must_use]
pub fn normalize(s: &str) -> String {
    let lowered = s.to_lowercase();
    let cleaned: String = lowered
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    let words: Vec<&str> = cleaned
        .split_whitespace()
        .filter(|w| !matches!(*w, "a" | "an" | "the"))
        .collect();
    words.join(" ")
}

/// How an answer compares with the gold answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerScore {
    /// Normalised answer equals a normalised gold answer.
    pub exact: bool,
    /// Normalised answer contains a normalised gold answer.
    pub contains: bool,
}

/// Score an answer against the gold answers. Empty gold lists score false.
#[must_use]
pub fn score_answer(answer: &str, gold: &[String]) -> AnswerScore {
    let a = normalize(answer);
    let golds: Vec<String> = gold.iter().map(|g| normalize(g)).filter(|g| !g.is_empty()).collect();
    AnswerScore {
        exact: golds.contains(&a),
        contains: !a.is_empty() && golds.iter().any(|g| a.contains(g.as_str())),
    }
}

/// The document id of a chunk: the file stem of its source path.
///
/// `/data/corpus/14717.txt` and `14717.txt` both give `14717`; a source with
/// no extension is its own id.
#[must_use]
pub fn doc_id(source: &str) -> &str {
    let name = source.rsplit(['/', '\\']).next().unwrap_or(source);
    match name.rsplit_once('.') {
        Some((stem, _)) if !stem.is_empty() => stem,
        _ => name,
    }
}

/// How a ranked list of retrieved chunks compares with the relevance labels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RelevanceScore {
    /// At least one retrieved chunk comes from a relevant document.
    pub hit: bool,
    /// Fraction of the relevant documents that appear among the retrieved chunks.
    pub doc_recall: f64,
    /// Normalised discounted cumulative gain over documents, binary relevance,
    /// with each document counted once at the rank of its first chunk.
    pub ndcg: f64,
}

/// Score retrieved chunk sources (in rank order) against the question's
/// `relevant` document ids. `None` when the question has no judgments.
#[must_use]
pub fn relevance(relevant: &[String], retrieved_sources: &[String]) -> Option<RelevanceScore> {
    if relevant.is_empty() {
        return None;
    }
    let truth: HashSet<&str> = relevant.iter().map(String::as_str).collect();
    // Documents in the order their first chunk appears.
    let mut docs: Vec<&str> = Vec::new();
    for s in retrieved_sources {
        let d = doc_id(s);
        if !docs.contains(&d) {
            docs.push(d);
        }
    }
    let found = docs.iter().filter(|d| truth.contains(*d)).count();
    let dcg: f64 = docs
        .iter()
        .enumerate()
        .filter(|(_, d)| truth.contains(*d))
        .map(|(i, _)| 1.0 / ((i + 2) as f64).log2())
        .sum();
    let ideal_n = truth.len().min(retrieved_sources.len().max(1));
    let idcg: f64 = (0..ideal_n).map(|i| 1.0 / ((i + 2) as f64).log2()).sum();
    Some(RelevanceScore {
        hit: found > 0,
        doc_recall: found as f64 / truth.len() as f64,
        ndcg: if idcg == 0.0 { 0.0 } else { dcg / idcg },
    })
}

/// Whether retrieval hit the question's label: a relevant document when the
/// question has judgments, otherwise a source path containing the `source`
/// hint. `None` when the question has neither.
#[must_use]
pub fn source_hit(expected: Option<&str>, relevant: &[String], retrieved_sources: &[String]) -> Option<bool> {
    if let Some(r) = relevance(relevant, retrieved_sources) {
        return Some(r.hit);
    }
    expected.map(|e| retrieved_sources.iter().any(|s| s.contains(e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_case_punctuation_and_articles() {
        assert_eq!(normalize("The refund window is 30 days."), "refund window is 30 days");
        assert_eq!(normalize("  A   B  "), "b");
        assert_eq!(normalize("!!!"), "");
    }

    #[test]
    fn scoring_exact_and_contains() {
        let gold = vec!["30 days".to_owned(), "thirty days".to_owned()];
        assert_eq!(
            score_answer("Thirty days.", &gold),
            AnswerScore {
                exact: true,
                contains: true
            }
        );
        assert_eq!(
            score_answer("Refunds within 30 days [1].", &gold),
            AnswerScore {
                exact: false,
                contains: true
            }
        );
        assert_eq!(
            score_answer("Two weeks.", &gold),
            AnswerScore {
                exact: false,
                contains: false
            }
        );
        assert_eq!(
            score_answer("", &gold),
            AnswerScore {
                exact: false,
                contains: false
            }
        );
        assert_eq!(
            score_answer("anything", &[]),
            AnswerScore {
                exact: false,
                contains: false
            }
        );
    }

    #[test]
    fn source_hit_is_none_without_hint() {
        let srcs = vec!["/tmp/docs/refunds.md".to_owned()];
        assert_eq!(source_hit(None, &[], &srcs), None);
        assert_eq!(source_hit(Some("refunds.md"), &[], &srcs), Some(true));
        assert_eq!(source_hit(Some("shipping"), &[], &srcs), Some(false));
        // Judgments take precedence over the substring hint.
        assert_eq!(source_hit(Some("refunds.md"), &["other".into()], &srcs), Some(false));
        assert_eq!(source_hit(None, &["refunds".into()], &srcs), Some(true));
    }

    #[test]
    fn doc_id_is_the_file_stem() {
        assert_eq!(doc_id("/data/corpus/14717.txt"), "14717");
        assert_eq!(doc_id("14717.txt"), "14717");
        assert_eq!(doc_id("C:\\x\\4983.md"), "4983");
        assert_eq!(doc_id("noext"), "noext");
        assert_eq!(doc_id("a/.hidden"), ".hidden");
        assert_eq!(doc_id("a.b.c.txt"), "a.b.c");
    }

    #[test]
    fn relevance_counts_documents_once_and_ranks_by_first_chunk() {
        let rel = vec!["a".to_owned(), "b".to_owned()];
        // Three chunks of `a`, then `c`, then `b`: docs ranked a, c, b.
        let srcs: Vec<String> = ["x/a.txt", "x/a.txt", "x/a.txt", "x/c.txt", "x/b.txt"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let r = relevance(&rel, &srcs).unwrap();
        assert!(r.hit);
        assert_eq!(r.doc_recall, 1.0);
        // DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5; IDCG = 1 + 1/log2(3)
        let expected = 1.5 / (1.0 + 1.0 / 3f64.log2());
        assert!((r.ndcg - expected).abs() < 1e-9, "{}", r.ndcg);

        let miss = relevance(&rel, &["x/c.txt".to_owned()]).unwrap();
        assert!(!miss.hit);
        assert_eq!(miss.doc_recall, 0.0);
        assert_eq!(miss.ndcg, 0.0);

        let perfect = relevance(&rel, &["b.txt".to_owned(), "a.txt".to_owned()]).unwrap();
        assert!((perfect.ndcg - 1.0).abs() < 1e-9);

        assert_eq!(relevance(&[], &srcs), None);
        let empty = relevance(&rel, &[]).unwrap();
        assert!(!empty.hit && empty.ndcg == 0.0);
    }

    #[test]
    fn ideal_dcg_is_capped_by_k() {
        // Five relevant documents, one retrieved and relevant: ndcg@1 = 1.
        let rel: Vec<String> = (0..5).map(|i| format!("d{i}")).collect();
        let r = relevance(&rel, &["d3.txt".to_owned()]).unwrap();
        assert!((r.ndcg - 1.0).abs() < 1e-9);
        assert_eq!(r.doc_recall, 0.2);
    }

    #[test]
    fn parse_jsonl_with_comments_and_errors() {
        let p = Path::new("qs.jsonl");
        let text = "# comment\n{\"id\":\"a\",\"question\":\"x?\",\"answers\":[\"y\"]}\n\n{\"id\":\"b\",\"question\":\"z?\",\"query\":\"zz\",\"relevant\":[\"1\",\"2\"]}\n";
        let qs = parse(text, p).unwrap();
        assert_eq!(qs.len(), 2);
        assert!(qs[1].answers.is_empty());
        assert_eq!(qs[0].retrieval_text(), "x?");
        assert_eq!(qs[1].retrieval_text(), "zz");
        assert_eq!(qs[1].relevant, ["1", "2"]);
        assert!(!qs[0].has_retrieval_label() && qs[1].has_retrieval_label());
        assert!(matches!(parse("", p), Err(QuestionError::Empty { .. })));
        assert!(matches!(parse("{bad", p), Err(QuestionError::Parse { line: 1, .. })));
        let dup = "{\"id\":\"a\",\"question\":\"x\"}\n{\"id\":\"a\",\"question\":\"y\"}";
        assert!(matches!(parse(dup, p), Err(QuestionError::DuplicateId { .. })));
        // Round trip keeps optional fields out when unset.
        let back = to_jsonl(&qs);
        assert!(!back.lines().next().unwrap().contains("relevant"));
        assert_eq!(parse(&back, p).unwrap(), qs);
    }
}
