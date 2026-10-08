//! The fixed RAG prompt. Changing this changes every benchmark, so bump the
//! `PROMPT_VERSION` when you do and record it in the run.

use hvec_store::Hit;

/// Recorded with every run so prompt changes are visible in the log.
pub const PROMPT_VERSION: &str = "v1";

/// Operator instructions sent as the system prompt.
pub const RAG_SYSTEM_PROMPT: &str = "You answer questions using only the numbered context passages provided. \
If the passages do not contain the answer, say so plainly instead of guessing. \
Cite the passage numbers you relied on in square brackets, for example [2].";

/// Format retrieved passages and the question as the user turn.
#[must_use]
pub fn build_user_prompt(hits: &[Hit], question: &str) -> String {
    let mut s = String::with_capacity(hits.iter().map(|h| h.text.len() + 32).sum::<usize>() + question.len() + 64);
    if hits.is_empty() {
        s.push_str("No context passages were retrieved.\n\n");
    } else {
        s.push_str("Context passages:\n\n");
        for (i, hit) in hits.iter().enumerate() {
            s.push_str(&format!(
                "[{}] (source: {}, chunk {})\n{}\n\n",
                i + 1,
                hit.source,
                hit.ordinal,
                hit.text.trim()
            ));
        }
    }
    s.push_str("Question: ");
    s.push_str(question.trim());
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_passages_from_one() {
        let hits = vec![
            Hit {
                chunk_id: 9,
                source: "a.md".into(),
                ordinal: 0,
                text: "alpha".into(),
                score: 0.9,
            },
            Hit {
                chunk_id: 3,
                source: "b.md".into(),
                ordinal: 2,
                text: "beta".into(),
                score: 0.8,
            },
        ];
        let p = build_user_prompt(&hits, "what?");
        assert!(p.contains("[1] (source: a.md, chunk 0)\nalpha"));
        assert!(p.contains("[2] (source: b.md, chunk 2)\nbeta"));
        assert!(p.ends_with("Question: what?"));
    }
}
