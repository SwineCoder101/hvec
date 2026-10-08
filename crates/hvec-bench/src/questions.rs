//! Question sets with gold answers, and the cheap scoring rules.
//!
//! Format: JSON Lines, one object per question.
//!
//! ```json
//! {"id": "q1", "question": "What is the refund window?", "answers": ["30 days", "thirty days"], "source": "refunds.md"}
//! ```
//!
//! `answers` lists every acceptable surface form. `source` is an optional
//! substring that a retrieved chunk's source path should contain; it lets the
//! retrieval step be scored without a chat model.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// One question with its acceptable answers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Question {
    pub id: String,
    pub question: String,
    #[serde(default)]
    pub answers: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
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
        if out.iter().any(|x| x.id == q.id) {
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

/// Whether any retrieved source path contains the expected `source` hint.
/// `None` when the question has no hint.
#[must_use]
pub fn source_hit(expected: Option<&str>, retrieved_sources: &[String]) -> Option<bool> {
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
        assert_eq!(source_hit(None, &srcs), None);
        assert_eq!(source_hit(Some("refunds.md"), &srcs), Some(true));
        assert_eq!(source_hit(Some("shipping"), &srcs), Some(false));
    }

    #[test]
    fn parse_jsonl_with_comments_and_errors() {
        let p = Path::new("qs.jsonl");
        let text =
            "# comment\n{\"id\":\"a\",\"question\":\"x?\",\"answers\":[\"y\"]}\n\n{\"id\":\"b\",\"question\":\"z?\"}\n";
        let qs = parse(text, p).unwrap();
        assert_eq!(qs.len(), 2);
        assert!(qs[1].answers.is_empty());
        assert!(matches!(parse("", p), Err(QuestionError::Empty { .. })));
        assert!(matches!(parse("{bad", p), Err(QuestionError::Parse { line: 1, .. })));
        let dup = "{\"id\":\"a\",\"question\":\"x\"}\n{\"id\":\"a\",\"question\":\"y\"}";
        assert!(matches!(parse(dup, p), Err(QuestionError::DuplicateId { .. })));
    }
}
