//! The agent memory loop: a script of writes and questions.
//!
//! An agent that keeps tool results and turns in a vector store writes far
//! more often than it reads, and reads back things it wrote minutes ago. A
//! memory script replays that pattern: `write` steps append text to the
//! store, `ask` steps retrieve from it and answer. The same script runs once
//! per codec, so the per-step bytes, latencies and answer scores line up.
//!
//! ```json
//! {"id": "w1", "write": "The deploy to eu-west failed with exit code 137.", "source": "tool:deploy"}
//! {"id": "q1", "ask": "What exit code did the eu-west deploy fail with?", "answers": ["137"], "source": "tool:deploy"}
//! ```

use std::path::Path;

use hvec_store::RunRow;
use serde::{Deserialize, Serialize};

/// One step of a memory script.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    pub id: String,
    /// Text to store, for a write step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub write: Option<String>,
    /// Question to answer from memory, for an ask step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ask: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub answers: Vec<String>,
    /// For a write: the source label stored with the chunk. For an ask: the
    /// substring a retrieved chunk's source should contain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl Step {
    #[must_use]
    pub fn is_write(&self) -> bool {
        self.write.is_some()
    }
}

/// Errors loading a script.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ScriptError {
    #[error("could not read {path}")]
    Read {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}:{line}: invalid step")]
    Parse {
        path: std::path::PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("{path}:{line}: step `{id}` must have exactly one of `write` or `ask`")]
    Shape {
        path: std::path::PathBuf,
        line: usize,
        id: String,
    },
    #[error("{path}: duplicate step id `{id}`")]
    DuplicateId { path: std::path::PathBuf, id: String },
    #[error("{path}: no steps")]
    Empty { path: std::path::PathBuf },
}

/// Load a JSONL memory script. Blank lines and `#` comments are skipped.
pub fn load(path: &Path) -> Result<Vec<Step>, ScriptError> {
    let text = std::fs::read_to_string(path).map_err(|source| ScriptError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    parse(&text, path)
}

/// Parse JSONL text. `path` is only used in error messages.
pub fn parse(text: &str, path: &Path) -> Result<Vec<Step>, ScriptError> {
    let mut out: Vec<Step> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let step: Step = serde_json::from_str(line).map_err(|source| ScriptError::Parse {
            path: path.to_path_buf(),
            line: i + 1,
            source,
        })?;
        if step.write.is_some() == step.ask.is_some() {
            return Err(ScriptError::Shape {
                path: path.to_path_buf(),
                line: i + 1,
                id: step.id,
            });
        }
        if out.iter().any(|s| s.id == step.id) {
            return Err(ScriptError::DuplicateId {
                path: path.to_path_buf(),
                id: step.id,
            });
        }
        out.push(step);
    }
    if out.is_empty() {
        return Err(ScriptError::Empty {
            path: path.to_path_buf(),
        });
    }
    Ok(out)
}

/// What one codec did over a whole script, aggregated from `memory` rows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemorySummary {
    pub codec: String,
    pub writes: usize,
    pub asks: usize,
    pub errors: usize,
    /// Encoded vector bytes in the store at the end of the script.
    pub total_bytes: u64,
    pub total_chunks: u64,
    pub mean_write_ms: f64,
    pub mean_retrieval_ms: f64,
    pub mean_generation_ms: f64,
    pub mean_input_tokens: f64,
    /// Over asks with gold answers.
    pub scored: usize,
    pub exact: f64,
    pub contains: f64,
    /// Over asks with a source hint.
    pub source_hit: Option<f64>,
}

fn num(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

/// Summarise `memory` rows per codec. Rows of other kinds are ignored.
#[must_use]
pub fn summarize(rows: &[RunRow]) -> Vec<MemorySummary> {
    let mut codecs: Vec<String> = Vec::new();
    for r in rows.iter().filter(|r| r.kind == "memory") {
        let c = r.codec.clone().unwrap_or_else(|| "-".into());
        if !codecs.contains(&c) {
            codecs.push(c);
        }
    }
    codecs
        .into_iter()
        .map(|codec| {
            let mine: Vec<&RunRow> = rows
                .iter()
                .filter(|r| r.kind == "memory" && r.codec.as_deref() == Some(codec.as_str()))
                .collect();
            let ok: Vec<&RunRow> = mine.iter().copied().filter(|r| r.metadata["error"].is_null()).collect();
            let writes: Vec<&RunRow> = ok.iter().copied().filter(|r| r.metadata["step"] == "write").collect();
            let asks: Vec<&RunRow> = ok.iter().copied().filter(|r| r.metadata["step"] == "ask").collect();
            let mean = |items: &[&RunRow], field: &str| {
                if items.is_empty() {
                    0.0
                } else {
                    items.iter().map(|r| num(&r.metrics[field])).sum::<f64>() / items.len() as f64
                }
            };
            let frac = |items: &[&RunRow], field: &str| {
                if items.is_empty() {
                    None
                } else {
                    Some(
                        items
                            .iter()
                            .filter(|r| r.metrics[field].as_bool() == Some(true))
                            .count() as f64
                            / items.len() as f64,
                    )
                }
            };
            let gold: Vec<&RunRow> = asks
                .iter()
                .copied()
                .filter(|r| r.metadata["answers"].as_array().is_some_and(|a| !a.is_empty()))
                .collect();
            let hinted: Vec<&RunRow> = asks
                .iter()
                .copied()
                .filter(|r| !r.metrics["source_hit"].is_null())
                .collect();
            let last = ok.last();
            MemorySummary {
                codec,
                writes: writes.len(),
                asks: asks.len(),
                errors: mine.len() - ok.len(),
                total_bytes: last.map_or(0, |r| r.metrics["cumulative_bytes"].as_u64().unwrap_or(0)),
                total_chunks: last.map_or(0, |r| r.metrics["cumulative_chunks"].as_u64().unwrap_or(0)),
                mean_write_ms: mean(&writes, "write_ms"),
                mean_retrieval_ms: mean(&asks, "retrieval_ms"),
                mean_generation_ms: mean(&asks, "generation_ms"),
                mean_input_tokens: mean(&asks, "input_tokens"),
                scored: gold.len(),
                exact: frac(&gold, "exact").unwrap_or(0.0),
                contains: frac(&gold, "contains").unwrap_or(0.0),
                source_hit: frac(&hinted, "source_hit"),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_writes_and_asks_and_rejects_bad_shapes() {
        let p = Path::new("m.jsonl");
        let text = "# memory\n{\"id\":\"w1\",\"write\":\"fact\",\"source\":\"tool:x\"}\n\n{\"id\":\"q1\",\"ask\":\"fact?\",\"answers\":[\"fact\"]}\n";
        let steps = parse(text, p).unwrap();
        assert_eq!(steps.len(), 2);
        assert!(steps[0].is_write() && !steps[1].is_write());
        assert_eq!(steps[1].answers, ["fact"]);
        assert!(matches!(parse("", p), Err(ScriptError::Empty { .. })));
        assert!(matches!(
            parse("{\"id\":\"a\"}", p),
            Err(ScriptError::Shape { line: 1, .. })
        ));
        assert!(matches!(
            parse("{\"id\":\"a\",\"write\":\"x\",\"ask\":\"y\"}", p),
            Err(ScriptError::Shape { .. })
        ));
        assert!(matches!(parse("{nope", p), Err(ScriptError::Parse { .. })));
        let dup = "{\"id\":\"a\",\"write\":\"x\"}\n{\"id\":\"a\",\"ask\":\"y\"}";
        assert!(matches!(parse(dup, p), Err(ScriptError::DuplicateId { .. })));
    }

    fn row(codec: &str, step: &str, metrics: serde_json::Value, answers: &[&str]) -> RunRow {
        let mut r = RunRow::new("memory");
        r.codec = Some(codec.into());
        r.metadata = serde_json::json!({ "step": step, "answers": answers });
        r.metrics = metrics;
        r
    }

    #[test]
    fn summary_takes_cumulative_bytes_from_the_last_step() {
        let rows = vec![
            row(
                "int8",
                "write",
                serde_json::json!({"write_ms": 4, "cumulative_bytes": 100, "cumulative_chunks": 1}),
                &[],
            ),
            row(
                "int8",
                "ask",
                serde_json::json!({"retrieval_ms": 2, "generation_ms": 50, "input_tokens": 300, "exact": false, "contains": true, "source_hit": true, "cumulative_bytes": 100, "cumulative_chunks": 1}),
                &["x"],
            ),
            row(
                "int8",
                "write",
                serde_json::json!({"write_ms": 6, "cumulative_bytes": 250, "cumulative_chunks": 3}),
                &[],
            ),
            row(
                "binary",
                "write",
                serde_json::json!({"write_ms": 1, "cumulative_bytes": 10, "cumulative_chunks": 1}),
                &[],
            ),
            RunRow::new("bench"),
        ];
        let s = summarize(&rows);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].codec, "int8");
        assert_eq!(s[0].writes, 2);
        assert_eq!(s[0].asks, 1);
        assert_eq!(s[0].total_bytes, 250);
        assert_eq!(s[0].total_chunks, 3);
        assert_eq!(s[0].mean_write_ms, 5.0);
        assert_eq!(s[0].contains, 1.0);
        assert_eq!(s[0].exact, 0.0);
        assert_eq!(s[0].source_hit, Some(1.0));
        assert_eq!(s[0].scored, 1);
        assert_eq!(s[1].codec, "binary");
        assert_eq!(s[1].source_hit, None);
    }
}
