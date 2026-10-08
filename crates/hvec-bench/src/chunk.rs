//! Fixed-size word chunking with overlap.
//!
//! Deliberately simple. The chunker is part of the pipeline under test, so it
//! must be deterministic and easy to describe in a write-up.

/// Chunking parameters, in whitespace-delimited words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkOptions {
    pub words: usize,
    pub overlap: usize,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            words: 200,
            overlap: 40,
        }
    }
}

/// Split `text` into overlapping word windows. Empty input yields no chunks.
#[must_use]
pub fn chunk_text(text: &str, opts: ChunkOptions) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }
    let size = opts.words.max(1);
    let step = size.saturating_sub(opts.overlap).max(1);

    let mut out = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + size).min(words.len());
        out.push(words[start..end].join(" "));
        if end == words.len() {
            break;
        }
        start += step;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_gives_no_chunks() {
        assert!(chunk_text("   \n ", ChunkOptions::default()).is_empty());
    }

    #[test]
    fn short_text_is_one_chunk() {
        let chunks = chunk_text("one two three", ChunkOptions { words: 10, overlap: 2 });
        assert_eq!(chunks, vec!["one two three"]);
    }

    #[test]
    fn overlap_repeats_trailing_words() {
        let text = (1..=10).map(|n| n.to_string()).collect::<Vec<_>>().join(" ");
        let chunks = chunk_text(&text, ChunkOptions { words: 4, overlap: 1 });
        assert_eq!(chunks, vec!["1 2 3 4", "4 5 6 7", "7 8 9 10"]);
    }

    #[test]
    fn overlap_at_least_size_still_terminates() {
        let chunks = chunk_text("a b c d e", ChunkOptions { words: 2, overlap: 5 });
        assert_eq!(chunks, vec!["a b", "b c", "c d", "d e"]);
    }
}
