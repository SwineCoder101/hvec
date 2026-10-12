//! The payload stage: what happens to the retrieved passages between
//! retrieval and the prompt.
//!
//! Key compression (the codec) changes *which* passages are retrieved;
//! payload compression changes *what is sent* for each of them. The two are
//! independent axes of the matrix, so the stage is named and recorded on
//! every run exactly like the codec.
//!
//! Stages: `none`; `truncate:<words>`, a naive baseline that keeps the first
//! N words of each passage; and `http:<url>`, an adapter that POSTs the
//! passages to an external compressor running as a local service, for
//! example a shim around Headroom or LLMLingua. The wire format is in
//! [`HttpRequest`] and [`HttpResponse`].

use hvec_store::Hit;
use serde::{Deserialize, Serialize};

/// A payload stage, parsed from `--payload`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Payload {
    None,
    Truncate { words: usize },
    Http { url: String },
}

/// Why a `--payload` value could not be parsed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PayloadError {
    #[error("unknown payload stage `{0}`; use none, truncate:<words> or http:<url>")]
    Unknown(String),
    #[error("truncate needs a positive word count, got `{0}`")]
    BadWords(String),
}

impl Payload {
    /// Parse `none`, `truncate:<words>` or `http:<url>`.
    pub fn parse(spec: &str) -> Result<Self, PayloadError> {
        let spec = spec.trim();
        if spec.is_empty() || spec == "none" {
            return Ok(Payload::None);
        }
        if let Some(n) = spec.strip_prefix("truncate:") {
            return n
                .parse::<usize>()
                .ok()
                .filter(|n| *n > 0)
                .map(|words| Payload::Truncate { words })
                .ok_or_else(|| PayloadError::BadWords(n.to_owned()));
        }
        if let Some(url) = spec.strip_prefix("http:") {
            let url = url.trim();
            if url.is_empty() {
                return Err(PayloadError::Unknown(spec.to_owned()));
            }
            return Ok(Payload::Http { url: url.to_owned() });
        }
        Err(PayloadError::Unknown(spec.to_owned()))
    }

    /// The name recorded with every run.
    #[must_use]
    pub fn name(&self) -> String {
        match self {
            Payload::None => "none".to_owned(),
            Payload::Truncate { words } => format!("truncate:{words}"),
            Payload::Http { url } => format!("http:{url}"),
        }
    }

    /// Whether the stage needs an HTTP round trip.
    #[must_use]
    pub fn is_remote(&self) -> bool {
        matches!(self, Payload::Http { .. })
    }
}

/// Keep the first `words` whitespace-delimited words of a passage.
#[must_use]
pub fn truncate_words(text: &str, words: usize) -> String {
    text.split_whitespace().take(words).collect::<Vec<_>>().join(" ")
}

/// Apply a local stage. `Http` is handled by the caller; here it is a no-op.
#[must_use]
pub fn apply_local(payload: &Payload, hits: &[Hit]) -> Vec<Hit> {
    match payload {
        Payload::Truncate { words } => hits
            .iter()
            .map(|h| Hit {
                text: truncate_words(&h.text, *words),
                ..h.clone()
            })
            .collect(),
        Payload::None | Payload::Http { .. } => hits.to_vec(),
    }
}

/// Body POSTed to an `http:<url>` compressor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpRequest {
    pub question: String,
    pub passages: Vec<HttpPassage>,
}

/// One passage on the wire. `source` and `ordinal` let a compressor use
/// document structure; it must return the same number of passages, in order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpPassage {
    pub source: String,
    pub ordinal: u32,
    pub text: String,
}

/// Body returned by the compressor: the passages with their `text` replaced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpResponse {
    pub passages: Vec<HttpPassage>,
}

impl HttpRequest {
    #[must_use]
    pub fn new(question: &str, hits: &[Hit]) -> Self {
        Self {
            question: question.to_owned(),
            passages: hits
                .iter()
                .map(|h| HttpPassage {
                    source: h.source.clone(),
                    ordinal: h.ordinal,
                    text: h.text.clone(),
                })
                .collect(),
        }
    }
}

/// Merge a compressor's response back onto the hits. Errors when the count differs.
pub fn apply_response(hits: &[Hit], response: HttpResponse) -> Result<Vec<Hit>, String> {
    if response.passages.len() != hits.len() {
        return Err(format!(
            "compressor returned {} passages for {} sent",
            response.passages.len(),
            hits.len()
        ));
    }
    Ok(hits
        .iter()
        .zip(response.passages)
        .map(|(h, p)| Hit {
            text: p.text,
            ..h.clone()
        })
        .collect())
}

/// What the stage did to the context, recorded in the run metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayloadStats {
    pub context_chars_before: usize,
    pub context_chars_after: usize,
    pub context_words_before: usize,
    pub context_words_after: usize,
}

impl PayloadStats {
    #[must_use]
    pub fn measure(before: &[Hit], after: &[Hit]) -> Self {
        let chars = |hs: &[Hit]| hs.iter().map(|h| h.text.chars().count()).sum();
        let words = |hs: &[Hit]| hs.iter().map(|h| h.text.split_whitespace().count()).sum();
        Self {
            context_chars_before: chars(before),
            context_chars_after: chars(after),
            context_words_before: words(before),
            context_words_after: words(after),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(text: &str) -> Hit {
        Hit {
            chunk_id: 1,
            source: "a.md".into(),
            ordinal: 0,
            text: text.into(),
            score: 0.5,
        }
    }

    #[test]
    fn parses_specs_and_names_them_back() {
        assert_eq!(Payload::parse("none").unwrap(), Payload::None);
        assert_eq!(Payload::parse("").unwrap(), Payload::None);
        assert_eq!(Payload::parse("truncate:40").unwrap(), Payload::Truncate { words: 40 });
        assert_eq!(
            Payload::parse("http:http://localhost:8080/compress").unwrap(),
            Payload::Http {
                url: "http://localhost:8080/compress".into()
            }
        );
        assert_eq!(Payload::parse("truncate:40").unwrap().name(), "truncate:40");
        assert!(matches!(Payload::parse("truncate:0"), Err(PayloadError::BadWords(_))));
        assert!(matches!(Payload::parse("truncate:x"), Err(PayloadError::BadWords(_))));
        assert!(matches!(Payload::parse("http:"), Err(PayloadError::Unknown(_))));
        assert!(matches!(Payload::parse("gzip"), Err(PayloadError::Unknown(_))));
        assert!(Payload::parse("http:x").unwrap().is_remote());
    }

    #[test]
    fn truncation_keeps_leading_words_and_measures() {
        let hits = vec![hit("one two three four five"), hit("six")];
        let out = apply_local(&Payload::Truncate { words: 2 }, &hits);
        assert_eq!(out[0].text, "one two");
        assert_eq!(out[1].text, "six");
        assert_eq!(out[0].chunk_id, 1);
        let s = PayloadStats::measure(&hits, &out);
        assert_eq!(s.context_words_before, 6);
        assert_eq!(s.context_words_after, 3);
        assert_eq!(s.context_chars_after, 10);
        assert_eq!(apply_local(&Payload::None, &hits), hits);
    }

    #[test]
    fn http_response_must_match_passage_count() {
        let hits = vec![hit("alpha beta"), hit("gamma")];
        let req = HttpRequest::new("q?", &hits);
        assert_eq!(req.passages.len(), 2);
        assert_eq!(req.passages[1].text, "gamma");
        let ok = apply_response(
            &hits,
            HttpResponse {
                passages: vec![
                    HttpPassage {
                        source: "a.md".into(),
                        ordinal: 0,
                        text: "α".into(),
                    },
                    HttpPassage {
                        source: "a.md".into(),
                        ordinal: 0,
                        text: "γ".into(),
                    },
                ],
            },
        )
        .unwrap();
        assert_eq!(ok[0].text, "α");
        assert_eq!(ok[1].chunk_id, 1);
        assert!(apply_response(&hits, HttpResponse { passages: vec![] }).is_err());
    }
}
