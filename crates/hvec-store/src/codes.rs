//! A collection's codes in one contiguous buffer, scanned in place.
//!
//! The SQLite table is the system of record, one row per chunk. Scanning it
//! row by row on every query copies every blob (and, for search, every
//! chunk's text) out of the database. [`CodeBuffer`] is the columnar form:
//! the codes back to back in a single allocation, with ids and sources as side
//! arrays. Scoring borrows each code as a `&[u8]` slice of the buffer, so
//! nothing is copied or allocated per vector, and the text is read only for
//! the hits.
//!
//! A buffer can be written to a flat file and read back with one read. A
//! memory-mapped file would avoid even that copy, but needs `unsafe` (the OS
//! may change a mapped file underneath safe code), which this workspace
//! forbids; SQLite's own memory map is available instead through
//! [`Store::set_mmap_size`](crate::Store::set_mmap_size).

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::Path;

use hvec_core::{Codec, CodecError, Metric};

const MAGIC: &[u8; 8] = b"HVECODE1";

/// Errors reading or writing a code file.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CodeFileError {
    #[error("could not read or write {path}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path} is not an hvec code file or is truncated")]
    Format { path: std::path::PathBuf },
}

/// One collection's codes, back to back, with their chunk ids and sources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeBuffer {
    code_size: usize,
    ids: Vec<i64>,
    sources: Vec<String>,
    codes: Vec<u8>,
}

/// A scored position in a [`CodeBuffer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scored {
    /// Index into the buffer.
    pub index: usize,
    pub score: f32,
}

/// Heap entry ordered so that the *worst* kept hit is on top.
#[derive(Debug, Clone, Copy)]
struct Entry {
    key: f32,
    index: usize,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    // `key` is "higher is better" after orienting by the metric. The heap's
    // top must be the worst: lowest key, and among equal keys the highest
    // index, so earlier vectors win ties as they did in a stable sort.
    fn cmp(&self, other: &Self) -> Ordering {
        other.key.total_cmp(&self.key).then(self.index.cmp(&other.index))
    }
}

impl CodeBuffer {
    /// An empty buffer for codes of `code_size` bytes.
    #[must_use]
    pub fn new(code_size: usize) -> Self {
        Self {
            code_size,
            ids: Vec::new(),
            sources: Vec::new(),
            codes: Vec::new(),
        }
    }

    /// Append one code. Panics if its length is not the buffer's code size.
    pub fn push(&mut self, id: i64, source: String, code: &[u8]) {
        assert_eq!(code.len(), self.code_size, "code size mismatch");
        self.ids.push(id);
        self.sources.push(source);
        self.codes.extend_from_slice(code);
    }

    /// Re-encode f32 vectors with `codec` into a new buffer.
    pub fn encode(
        codec: &dyn Codec,
        ids: &[i64],
        sources: &[String],
        vectors: &[Vec<f32>],
    ) -> Result<Self, CodecError> {
        let mut b = Self::new(codec.encoded_size());
        b.codes.reserve(vectors.len() * b.code_size);
        for ((id, source), v) in ids.iter().zip(sources).zip(vectors) {
            b.push(*id, source.clone(), &codec.encode(v)?);
        }
        Ok(b)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    #[must_use]
    pub fn code_size(&self) -> usize {
        self.code_size
    }

    #[must_use]
    pub fn ids(&self) -> &[i64] {
        &self.ids
    }

    #[must_use]
    pub fn sources(&self) -> &[String] {
        &self.sources
    }

    /// The whole code region, `len() × code_size()` bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.codes
    }

    /// The code at `index`, borrowed from the buffer.
    #[must_use]
    pub fn code(&self, index: usize) -> &[u8] {
        &self.codes[index * self.code_size..(index + 1) * self.code_size]
    }

    /// Score every code and keep the best `k`, best first. The query is
    /// prepared once; each code is scored in place.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        codec: &dyn Codec,
        metric: Metric,
    ) -> Result<Vec<Scored>, CodecError> {
        self.search_threads(query, k, codec, metric, 1)
    }

    /// As [`CodeBuffer::search`], with the scan split over `threads` scoped
    /// threads. Results are identical to the single-threaded scan.
    pub fn search_threads(
        &self,
        query: &[f32],
        k: usize,
        codec: &dyn Codec,
        metric: Metric,
        threads: usize,
    ) -> Result<Vec<Scored>, CodecError> {
        if codec.encoded_size() != self.code_size {
            return Err(CodecError::PayloadSize {
                expected: codec.encoded_size(),
                got: self.code_size,
            });
        }
        let prepared = codec.prepare_query(query)?;
        let higher = metric.higher_is_better();
        let n = self.len();
        let threads = threads.clamp(1, n.max(1));
        let scan = |lo: usize, hi: usize| -> Result<BinaryHeap<Entry>, CodecError> {
            let mut heap = BinaryHeap::with_capacity(k + 1);
            if k == 0 {
                return Ok(heap);
            }
            for (offset, code) in self.codes[lo * self.code_size..hi * self.code_size]
                .chunks_exact(self.code_size)
                .enumerate()
            {
                let s = codec.score_prepared(&prepared, code, metric)?;
                let e = Entry {
                    key: if higher { s } else { -s },
                    index: lo + offset,
                };
                if heap.len() < k {
                    heap.push(e);
                } else if heap.peek().is_some_and(|worst| e < *worst) {
                    heap.pop();
                    heap.push(e);
                }
            }
            Ok(heap)
        };
        let heaps: Vec<BinaryHeap<Entry>> = if threads == 1 {
            vec![scan(0, n)?]
        } else {
            let per = n.div_ceil(threads);
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let (lo, hi) = ((t * per).min(n), ((t + 1) * per).min(n));
                        s.spawn(move || scan(lo, hi))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("scan thread panicked"))
                    .collect::<Result<Vec<_>, _>>()
            })?
        };
        let mut all: Vec<Entry> = heaps.into_iter().flat_map(BinaryHeap::into_vec).collect();
        all.sort();
        all.truncate(k);
        Ok(all
            .into_iter()
            .map(|e| Scored {
                index: e.index,
                score: if higher { e.key } else { -e.key },
            })
            .collect())
    }

    /// Write the buffer as a flat file: a header, the ids, then the codes.
    /// Sources are not written; they stay in the database.
    pub fn write_file(&self, path: &Path) -> Result<(), CodeFileError> {
        let mut out = Vec::with_capacity(24 + self.len() * (8 + self.code_size));
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(self.code_size as u64).to_le_bytes());
        out.extend_from_slice(&(self.len() as u64).to_le_bytes());
        for id in &self.ids {
            out.extend_from_slice(&id.to_le_bytes());
        }
        out.extend_from_slice(&self.codes);
        std::fs::write(path, out).map_err(|source| CodeFileError::Io {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Read a file written by [`CodeBuffer::write_file`] with a single read.
    /// Sources come back empty.
    pub fn read_file(path: &Path) -> Result<Self, CodeFileError> {
        let data = std::fs::read(path).map_err(|source| CodeFileError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let bad = || CodeFileError::Format {
            path: path.to_path_buf(),
        };
        if data.len() < 24 || &data[..8] != MAGIC {
            return Err(bad());
        }
        let u64_at = |o: usize| u64::from_le_bytes(data[o..o + 8].try_into().unwrap_or([0; 8]));
        let code_size = usize::try_from(u64_at(8)).map_err(|_| bad())?;
        let n = usize::try_from(u64_at(16)).map_err(|_| bad())?;
        let ids_end = 24usize.checked_add(n.checked_mul(8).ok_or_else(bad)?).ok_or_else(bad)?;
        let end = ids_end
            .checked_add(n.checked_mul(code_size).ok_or_else(bad)?)
            .ok_or_else(bad)?;
        if data.len() != end {
            return Err(bad());
        }
        let ids = data[24..ids_end]
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
            .collect();
        let mut codes = data;
        codes.drain(..ids_end);
        Ok(Self {
            code_size,
            ids,
            sources: vec![String::new(); n],
            codes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hvec_core::codecs::{BinaryCodec, F32Codec, HammingCodec, Int8Codec};
    use hvec_core::distance;

    fn corpus(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut s = 12345u64;
        (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                    })
                    .collect();
                distance::normalize(&mut v);
                v
            })
            .collect()
    }

    fn reference(codec: &dyn Codec, q: &[f32], vs: &[Vec<f32>], k: usize, metric: Metric) -> Vec<(usize, f32)> {
        let mut all: Vec<(usize, f32)> = vs
            .iter()
            .enumerate()
            .map(|(i, v)| (i, codec.score(q, &codec.encode(v).unwrap(), metric).unwrap()))
            .collect();
        if metric.higher_is_better() {
            all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        } else {
            all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        }
        all.truncate(k);
        all
    }

    #[test]
    fn buffer_search_matches_a_full_sort_for_every_codec_and_thread_count() {
        let vs = corpus(500, 32);
        let ids: Vec<i64> = (0..500).map(|i| i * 3 + 1).collect();
        let sources: Vec<String> = (0..500).map(|i| format!("doc{i}.txt")).collect();
        let codecs: Vec<Box<dyn Codec>> = vec![
            Box::new(F32Codec::new(32)),
            Box::new(Int8Codec::new(32)),
            Box::new(BinaryCodec::new(32)),
            Box::new(HammingCodec::new(32)),
        ];
        for codec in &codecs {
            let buf = CodeBuffer::encode(codec.as_ref(), &ids, &sources, &vs).unwrap();
            assert_eq!(buf.len(), 500);
            assert_eq!(buf.bytes().len(), 500 * codec.encoded_size());
            for metric in [Metric::Cosine, Metric::L2] {
                let want = reference(codec.as_ref(), &vs[7], &vs, 10, metric);
                for threads in [1, 3, 8] {
                    let got = buf.search_threads(&vs[7], 10, codec.as_ref(), metric, threads).unwrap();
                    let got: Vec<(usize, f32)> = got.iter().map(|s| (s.index, s.score)).collect();
                    assert_eq!(got, want, "{} {metric:?} threads={threads}", codec.name());
                }
            }
        }
    }

    #[test]
    fn ties_keep_the_earlier_vector_and_k_is_respected() {
        let codec = BinaryCodec::new(8);
        let v = vec![vec![1.0f32; 8]; 5];
        let ids: Vec<i64> = (0..5).collect();
        let buf = CodeBuffer::encode(&codec, &ids, &vec![String::new(); 5], &v).unwrap();
        let got = buf.search(&v[0], 3, &codec, Metric::Cosine).unwrap();
        assert_eq!(got.iter().map(|s| s.index).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert!(buf.search(&v[0], 0, &codec, Metric::Cosine).unwrap().is_empty());
        assert_eq!(buf.search(&v[0], 99, &codec, Metric::Cosine).unwrap().len(), 5);
        assert!(buf.search(&v[0], 3, &Int8Codec::new(8), Metric::Cosine).is_err());
    }

    #[test]
    fn code_files_roundtrip_and_reject_damage() {
        let vs = corpus(40, 16);
        let ids: Vec<i64> = (100..140).collect();
        let codec = Int8Codec::new(16);
        let buf = CodeBuffer::encode(&codec, &ids, &vec![String::new(); 40], &vs).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.codes");
        buf.write_file(&path).unwrap();
        let back = CodeBuffer::read_file(&path).unwrap();
        assert_eq!(back, buf);
        let mut data = std::fs::read(&path).unwrap();
        data.pop();
        std::fs::write(&path, &data).unwrap();
        assert!(matches!(
            CodeBuffer::read_file(&path),
            Err(CodeFileError::Format { .. })
        ));
        std::fs::write(&path, b"not a code file at all.").unwrap();
        assert!(CodeBuffer::read_file(&path).is_err());
    }
}
