//! Symmetric binary scoring: sign bits on both sides, compared by popcount.
//!
//! `binary-hamming` stores exactly what `binary` stores, one sign bit per
//! dimension. The difference is the query. `binary` scores the f32 query
//! against the bits (asymmetric: the query keeps its magnitudes).
//! `binary-hamming` binarises the query too, once per search in
//! [`Codec::prepare_query`], and scores each stored vector with XOR and
//! popcount over 64-bit words. That is the form whose cost is set by the bytes
//! read, so it is the binary codec a speed claim can be made for.
//!
//! Error model: approximate, and coarser than `binary`, because both sides are
//! quantized. For sign random-projection codes the fraction of differing bits
//! estimates the angle between the vectors divided by π (Charikar 2002), so
//! the score is `cos(π · hamming / d)`, an estimate of cosine similarity.
//! `Dot` is treated as cosine (embeddings are unit length); `L2` is the
//! unit-vector distance `2 − 2·cos`. Unlike every other codec, `score` does
//! not equal the metric on the decoded vector, because the query is quantized
//! as well; tests check it against the Hamming distance instead.

use crate::codecs::{check_dim, check_payload};
use crate::{Codec, CodecError, Encoded, Metric, Vector};

/// Sign bits, scored symmetrically by Hamming distance.
#[derive(Debug, Clone)]
pub struct HammingCodec {
    dimension: usize,
}

impl HammingCodec {
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn pack(&self, v: &[f32]) -> Vec<u8> {
        let mut out = vec![0u8; self.encoded_size()];
        for (i, &x) in v.iter().enumerate() {
            if x > 0.0 {
                out[i / 8] |= 1 << (i % 8);
            }
        }
        out
    }

    /// Number of 64-bit words a code occupies, the last one zero-padded.
    fn words(&self) -> usize {
        self.encoded_size().div_ceil(8)
    }

    fn word(bytes: &[u8], j: usize) -> u64 {
        let start = j * 8;
        let end = (start + 8).min(bytes.len());
        let mut w = [0u8; 8];
        w[..end - start].copy_from_slice(&bytes[start..end]);
        u64::from_le_bytes(w)
    }

    /// Hamming distance between a prepared query and a stored code.
    fn hamming(&self, prepared: &[f32], encoded: &[u8]) -> u32 {
        let mut h = 0u32;
        if encoded.len() % 8 == 0 {
            for (j, chunk) in encoded.chunks_exact(8).enumerate() {
                let x = u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8]));
                h += (x ^ Self::query_word(prepared, j)).count_ones();
            }
        } else {
            for j in 0..self.words() {
                h += (Self::word(encoded, j) ^ Self::query_word(prepared, j)).count_ones();
            }
        }
        h
    }

    /// The prepared query holds its packed sign bits, 32 per f32 slot (the
    /// slot's bit pattern, not a number), two slots per 64-bit word.
    fn query_word(prepared: &[f32], j: usize) -> u64 {
        u64::from(prepared[2 * j].to_bits()) | (u64::from(prepared[2 * j + 1].to_bits()) << 32)
    }
}

impl Codec for HammingCodec {
    fn name(&self) -> &str {
        "binary-hamming"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn encoded_size(&self) -> usize {
        self.dimension.div_ceil(8)
    }

    fn supported_metrics(&self) -> &'static [Metric] {
        &[Metric::Cosine, Metric::Dot, Metric::L2]
    }

    fn encode(&self, vector: &[f32]) -> Result<Encoded, CodecError> {
        check_dim(self.dimension, vector.len())?;
        Ok(Encoded(self.pack(vector)))
    }

    /// The same reconstruction as `binary`: ±1/√d per dimension.
    fn decode(&self, encoded: &[u8]) -> Result<Vector, CodecError> {
        check_payload(self.encoded_size(), encoded.len())?;
        let mag = (1.0 / (self.dimension as f64).sqrt()) as f32;
        Ok((0..self.dimension)
            .map(|i| if encoded[i / 8] >> (i % 8) & 1 == 1 { mag } else { -mag })
            .collect())
    }

    fn score(&self, query: &[f32], encoded: &[u8], metric: Metric) -> Result<f32, CodecError> {
        let prepared = self.prepare_query(query)?;
        self.score_prepared(&prepared, encoded, metric)
    }

    /// Binarise and pack the query once per search.
    fn prepare_query(&self, query: &[f32]) -> Result<Vector, CodecError> {
        check_dim(self.dimension, query.len())?;
        let mut bytes = self.pack(query);
        bytes.resize(self.words() * 8, 0);
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect())
    }

    fn score_prepared(&self, prepared: &[f32], encoded: &[u8], metric: Metric) -> Result<f32, CodecError> {
        check_payload(self.encoded_size(), encoded.len())?;
        if prepared.len() != self.words() * 2 {
            return Err(CodecError::DimensionMismatch {
                expected: self.words() * 2,
                got: prepared.len(),
            });
        }
        let h = self.hamming(prepared, encoded);
        let cos = (std::f64::consts::PI * f64::from(h) / self.dimension as f64).cos();
        Ok(match metric {
            Metric::Cosine | Metric::Dot => cos as f32,
            Metric::L2 => (2.0 - 2.0 * cos) as f32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec_by_name;

    fn naive_hamming(a: &[f32], b: &[f32]) -> u32 {
        a.iter().zip(b).filter(|(x, y)| (**x > 0.0) != (**y > 0.0)).count() as u32
    }

    #[test]
    fn popcount_equals_the_naive_hamming_distance() {
        for dim in [384, 100, 7, 64, 65] {
            let c = HammingCodec::new(dim);
            let a: Vec<f32> = (0..dim).map(|i| ((i * 37 % 11) as f32) - 5.0).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i * 13 % 7) as f32) - 3.0).collect();
            let p = c.prepare_query(&a).unwrap();
            let e = c.encode(&b).unwrap();
            assert_eq!(c.hamming(&p, &e), naive_hamming(&a, &b), "dim {dim}");
            let want = (std::f64::consts::PI * f64::from(naive_hamming(&a, &b)) / dim as f64).cos() as f32;
            assert_eq!(c.score(&a, &e, Metric::Cosine).unwrap(), want);
            assert_eq!(c.score_prepared(&p, &e, Metric::Cosine).unwrap(), want);
        }
    }

    #[test]
    fn identical_signs_score_one_and_opposite_minus_one() {
        let c = HammingCodec::new(16);
        let v: Vec<f32> = (0..16).map(|i| if i % 3 == 0 { 1.0 } else { -1.0 }).collect();
        let neg: Vec<f32> = v.iter().map(|x| -x).collect();
        assert_eq!(c.score(&v, &c.encode(&v).unwrap(), Metric::Cosine).unwrap(), 1.0);
        assert!((c.score(&v, &c.encode(&neg).unwrap(), Metric::Cosine).unwrap() + 1.0).abs() < 1e-6);
        assert!(c.score(&v, &c.encode(&v).unwrap(), Metric::L2).unwrap().abs() < 1e-6);
    }

    #[test]
    fn stores_the_same_bits_as_binary_and_is_built_by_name() {
        let c = codec_by_name("binary-hamming", 24).unwrap();
        let b = codec_by_name("binary", 24).unwrap();
        let v: Vec<f32> = (0..24).map(|i| (i as f32 * 0.7).sin()).collect();
        assert_eq!(c.encode(&v).unwrap(), b.encode(&v).unwrap());
        assert_eq!(c.encoded_size(), 3);
        assert!(
            c.score_prepared(&[0.0; 3], &c.encode(&v).unwrap(), Metric::Cosine)
                .is_err()
        );
    }
}
