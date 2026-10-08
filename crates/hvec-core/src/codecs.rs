//! Codec implementations. Milestone 1 ships the uncompressed baseline only.

use crate::{Codec, CodecError, Encoded, Metric, Vector, distance};

/// The uncompressed baseline. Stores raw little-endian f32 values.
///
/// Error model: exact for every metric. Every other codec is measured
/// against this one.
#[derive(Debug, Clone)]
pub struct F32Codec {
    dimension: usize,
}

impl F32Codec {
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn check_dim(&self, len: usize) -> Result<(), CodecError> {
        if len == self.dimension {
            Ok(())
        } else {
            Err(CodecError::DimensionMismatch {
                expected: self.dimension,
                got: len,
            })
        }
    }
}

impl Codec for F32Codec {
    fn name(&self) -> &'static str {
        "f32"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn encoded_size(&self) -> usize {
        self.dimension * 4
    }

    fn supported_metrics(&self) -> &'static [Metric] {
        &[Metric::Dot, Metric::Cosine, Metric::L2]
    }

    fn encode(&self, vector: &[f32]) -> Result<Encoded, CodecError> {
        self.check_dim(vector.len())?;
        Ok(Encoded(vector.iter().flat_map(|v| v.to_le_bytes()).collect()))
    }

    fn decode(&self, encoded: &Encoded) -> Result<Vector, CodecError> {
        if encoded.0.len() != self.encoded_size() {
            return Err(CodecError::PayloadSize {
                expected: self.encoded_size(),
                got: encoded.0.len(),
            });
        }
        Ok(encoded
            .0
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn score(&self, query: &[f32], encoded: &Encoded, metric: Metric) -> Result<f32, CodecError> {
        self.check_dim(query.len())?;
        // Baseline: decoding is the documented behaviour here.
        let stored = self.decode(encoded)?;
        Ok(distance::score(query, &stored, metric))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_roundtrip_is_exact() {
        let codec = F32Codec::new(3);
        let v = vec![0.1, -2.5, 1e-7];
        let enc = codec.encode(&v).unwrap();
        assert_eq!(enc.0.len(), 12);
        assert_eq!(codec.decode(&enc).unwrap(), v);
    }

    #[test]
    fn f32_score_matches_reference() {
        let codec = F32Codec::new(2);
        let q = [1.0, 0.0];
        let enc = codec.encode(&[0.0, 1.0]).unwrap();
        assert_eq!(codec.score(&q, &enc, Metric::Dot).unwrap(), 0.0);
        assert_eq!(codec.score(&q, &enc, Metric::L2).unwrap(), 2.0);
    }

    #[test]
    fn f32_rejects_wrong_dimension() {
        let codec = F32Codec::new(2);
        assert!(matches!(
            codec.encode(&[1.0]),
            Err(CodecError::DimensionMismatch { expected: 2, got: 1 })
        ));
    }

    #[test]
    fn compression_ratio_of_baseline_is_one() {
        assert_eq!(F32Codec::new(128).compression_ratio(), 1.0);
    }
}
