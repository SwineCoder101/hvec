//! Importer for BEIR-format retrieval datasets.
//!
//! A BEIR dataset directory holds `corpus.jsonl` (`_id`, `title`, `text`),
//! `queries.jsonl` (`_id`, `text`) and `qrels/<split>.tsv` with the columns
//! `query-id`, `corpus-id`, `score`. The importer turns every judged query of
//! one split into a [`Question`] whose `relevant` list holds the corpus ids
//! with a positive score, and can write the corpus out as one `<id>.txt` per
//! document so that `hvec ingest` produces chunks whose source stem is the
//! document id.
//!
//! BEIR carries no gold answers. For SciFact the original release's claim
//! files add a label per cited document (`SUPPORT` or `CONTRADICT`), which
//! [`apply_scifact_labels`] turns into a yes/no style question with gold
//! answers, keeping the bare claim as the retrieval query.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::questions::Question;

/// Errors importing a dataset.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BeirError {
    #[error("could not read {path}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not write {path}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}:{line}: invalid record")]
    Parse {
        path: PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("{path}:{line}: expected `query-id<TAB>corpus-id<TAB>score`")]
    Qrels { path: PathBuf, line: usize },
    #[error("{path}: no judged queries in split")]
    NoQueries { path: PathBuf },
}

#[derive(Debug, Deserialize)]
struct CorpusDoc {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
struct QueryRec {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

/// What an import produced, for the summary line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportSummary {
    pub questions: usize,
    pub judgments: usize,
    pub corpus_docs: usize,
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>, BeirError> {
    let text = std::fs::read_to_string(path).map_err(|source| BeirError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            serde_json::from_str(l).map_err(|source| BeirError::Parse {
                path: path.to_path_buf(),
                line: i + 1,
                source,
            })
        })
        .collect()
}

/// Parse a qrels TSV into `query id -> relevant corpus ids` (positive scores only).
pub fn parse_qrels(text: &str, path: &Path) -> Result<BTreeMap<String, BTreeSet<String>>, BeirError> {
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 3 {
            return Err(BeirError::Qrels {
                path: path.to_path_buf(),
                line: i + 1,
            });
        }
        // Header row.
        if i == 0 && cols[2].parse::<f64>().is_err() {
            continue;
        }
        let score: f64 = cols[2].trim().parse().map_err(|_| BeirError::Qrels {
            path: path.to_path_buf(),
            line: i + 1,
        })?;
        if score > 0.0 {
            out.entry(cols[0].trim().to_owned())
                .or_default()
                .insert(cols[1].trim().to_owned());
        }
    }
    Ok(out)
}

/// Build questions for every query judged in `qrels/<split>.tsv`.
///
/// Questions come out in the order of `queries.jsonl`. The question text is
/// the query verbatim and `answers` is empty.
pub fn import(dir: &Path, split: &str) -> Result<Vec<Question>, BeirError> {
    let qrels_path = dir.join("qrels").join(format!("{split}.tsv"));
    let qrels_text = std::fs::read_to_string(&qrels_path).map_err(|source| BeirError::Read {
        path: qrels_path.clone(),
        source,
    })?;
    let qrels = parse_qrels(&qrels_text, &qrels_path)?;
    let queries: Vec<QueryRec> = read_jsonl(&dir.join("queries.jsonl"))?;
    let out: Vec<Question> = queries
        .into_iter()
        .filter_map(|q| {
            qrels.get(&q.id).map(|rel| Question {
                id: q.id,
                question: q.text.split_whitespace().collect::<Vec<_>>().join(" "),
                query: None,
                answers: Vec::new(),
                source: None,
                relevant: rel.iter().cloned().collect(),
            })
        })
        .collect();
    if out.is_empty() {
        return Err(BeirError::NoQueries { path: qrels_path });
    }
    Ok(out)
}

/// Write `corpus.jsonl` as `<out>/<id>.txt`, title then a blank line then the
/// text. Returns the number of documents written.
pub fn write_corpus(dir: &Path, out: &Path) -> Result<usize, BeirError> {
    let docs: Vec<CorpusDoc> = read_jsonl(&dir.join("corpus.jsonl"))?;
    std::fs::create_dir_all(out).map_err(|source| BeirError::Write {
        path: out.to_path_buf(),
        source,
    })?;
    for d in &docs {
        let path = out.join(format!("{}.txt", d.id));
        let body = match (d.title.trim(), d.text.trim()) {
            ("", t) => t.to_owned(),
            (title, "") => title.to_owned(),
            (title, t) => format!("{title}\n\n{t}"),
        };
        std::fs::write(&path, body + "\n").map_err(|source| BeirError::Write { path, source })?;
    }
    Ok(docs.len())
}

#[derive(Debug, Deserialize)]
struct ScifactClaim {
    id: serde_json::Value,
    claim: String,
    #[serde(default)]
    evidence: BTreeMap<String, Vec<ScifactEvidence>>,
}

#[derive(Debug, Deserialize)]
struct ScifactEvidence {
    label: String,
}

/// Gold answers for a SciFact verdict, by label.
pub const SUPPORT_ANSWERS: &[&str] = &["supports"];
/// Gold answers for a SciFact verdict, by label.
pub const CONTRADICT_ANSWERS: &[&str] = &["refutes", "contradicts"];

/// Phrase a SciFact claim as the question the chat model sees.
#[must_use]
pub fn scifact_question(claim: &str) -> String {
    format!("Does the evidence support or refute this claim? Answer with one word, supports or refutes, then cite. Claim: {claim}")
}

/// Enrich BEIR SciFact questions with verdicts from the original release's
/// `claims_*.jsonl`. A claim whose evidence is all `SUPPORT` gets the answer
/// `supports`; all `CONTRADICT` gets `refutes`; mixed or absent evidence
/// leaves the question unlabelled. The bare claim becomes `query`. Returns
/// how many questions were labelled.
pub fn apply_scifact_labels(questions: &mut [Question], claims_path: &Path) -> Result<usize, BeirError> {
    let claims: Vec<ScifactClaim> = read_jsonl(claims_path)?;
    let by_id: BTreeMap<String, &ScifactClaim> = claims
        .iter()
        .map(|c| {
            let id = match &c.id {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            (id, c)
        })
        .collect();
    let mut labelled = 0;
    for q in questions.iter_mut() {
        let Some(c) = by_id.get(&q.id) else { continue };
        let labels: BTreeSet<&str> = c
            .evidence
            .values()
            .flatten()
            .map(|e| e.label.as_str())
            .collect();
        let answers: &[&str] = match labels.iter().copied().collect::<Vec<_>>().as_slice() {
            ["SUPPORT"] => SUPPORT_ANSWERS,
            ["CONTRADICT"] => CONTRADICT_ANSWERS,
            _ => continue,
        };
        q.query = Some(c.claim.split_whitespace().collect::<Vec<_>>().join(" "));
        q.question = scifact_question(q.query.as_deref().unwrap_or(&c.claim));
        q.answers = answers.iter().map(|s| (*s).to_owned()).collect();
        labelled += 1;
    }
    Ok(labelled)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("corpus.jsonl"),
            "{\"_id\":\"10\",\"title\":\"Alpha\",\"text\":\"alpha body\"}\n{\"_id\":\"20\",\"title\":\"\",\"text\":\"beta body\"}\n{\"_id\":\"30\",\"title\":\"Gamma\",\"text\":\"\"}\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("queries.jsonl"),
            "{\"_id\":\"1\",\"text\":\"first   query\"}\n{\"_id\":\"2\",\"text\":\"unjudged\"}\n{\"_id\":\"3\",\"text\":\"third\"}\n",
        )
        .unwrap();
        std::fs::create_dir_all(dir.path().join("qrels")).unwrap();
        std::fs::write(
            dir.path().join("qrels/test.tsv"),
            "query-id\tcorpus-id\tscore\n1\t10\t1\n1\t20\t2\n3\t30\t1\n3\t10\t0\n",
        )
        .unwrap();
        dir
    }

    #[test]
    fn imports_judged_queries_with_positive_judgments() {
        let d = fixture();
        let qs = import(d.path(), "test").unwrap();
        assert_eq!(qs.len(), 2);
        assert_eq!(qs[0].id, "1");
        assert_eq!(qs[0].question, "first query");
        assert_eq!(qs[0].relevant, ["10", "20"]);
        assert!(qs[0].answers.is_empty());
        assert_eq!(qs[1].id, "3");
        assert_eq!(qs[1].relevant, ["30"], "zero-score judgment is dropped");
        assert!(matches!(import(d.path(), "dev"), Err(BeirError::Read { .. })));
    }

    #[test]
    fn qrels_without_header_and_malformed_rows() {
        let p = Path::new("q.tsv");
        let m = parse_qrels("1\t10\t1\n2\t20\t1\n", p).unwrap();
        assert_eq!(m.len(), 2);
        assert!(matches!(parse_qrels("1\t10\n", p), Err(BeirError::Qrels { line: 1, .. })));
        assert!(matches!(
            parse_qrels("a\tb\tc\n1\t10\tx\n", p),
            Err(BeirError::Qrels { line: 2, .. })
        ));
    }

    #[test]
    fn writes_corpus_as_one_file_per_document() {
        let d = fixture();
        let out = d.path().join("txt");
        assert_eq!(write_corpus(d.path(), &out).unwrap(), 3);
        assert_eq!(std::fs::read_to_string(out.join("10.txt")).unwrap(), "Alpha\n\nalpha body\n");
        assert_eq!(std::fs::read_to_string(out.join("20.txt")).unwrap(), "beta body\n");
        assert_eq!(std::fs::read_to_string(out.join("30.txt")).unwrap(), "Gamma\n");
        assert_eq!(crate::questions::doc_id(out.join("10.txt").to_str().unwrap()), "10");
    }

    #[test]
    fn scifact_labels_turn_claims_into_verdict_questions() {
        let d = fixture();
        let mut qs = import(d.path(), "test").unwrap();
        let claims = d.path().join("claims_dev.jsonl");
        std::fs::write(
            &claims,
            concat!(
                "{\"id\":1,\"claim\":\"Alpha  is good\",\"evidence\":{\"10\":[{\"sentences\":[0],\"label\":\"SUPPORT\"}],\"20\":[{\"sentences\":[1],\"label\":\"SUPPORT\"}]}}\n",
                "{\"id\":3,\"claim\":\"Gamma is bad\",\"evidence\":{\"30\":[{\"sentences\":[0],\"label\":\"CONTRADICT\"}]}}\n",
                "{\"id\":4,\"claim\":\"no such query\",\"evidence\":{}}\n",
            ),
        )
        .unwrap();
        assert_eq!(apply_scifact_labels(&mut qs, &claims).unwrap(), 2);
        assert_eq!(qs[0].query.as_deref(), Some("Alpha is good"));
        assert!(qs[0].question.starts_with("Does the evidence support or refute"));
        assert!(qs[0].question.ends_with("Claim: Alpha is good"));
        assert_eq!(qs[0].answers, ["supports"]);
        assert_eq!(qs[1].answers, ["refutes", "contradicts"]);
        assert_eq!(qs[0].retrieval_text(), "Alpha is good");

        // Mixed evidence leaves the question unlabelled.
        let mut qs = import(d.path(), "test").unwrap();
        std::fs::write(
            &claims,
            "{\"id\":\"1\",\"claim\":\"x\",\"evidence\":{\"10\":[{\"sentences\":[0],\"label\":\"SUPPORT\"}],\"20\":[{\"sentences\":[0],\"label\":\"CONTRADICT\"}]}}\n",
        )
        .unwrap();
        assert_eq!(apply_scifact_labels(&mut qs, &claims).unwrap(), 0);
        assert!(qs[0].answers.is_empty() && qs[0].query.is_none());
    }
}
