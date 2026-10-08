//! Codec implementations.
//!
//! Every codec's `score` is tested two ways: it must equal the reference
//! distance on its own decoded vector (internal consistency), and it must
//! approximate the reference distance on the original f32 vector within the
//! codec's documented error model.

use crate::{Codec, CodecError, Encoded, Metric, Vector, distance};

fn check_dim(expected: usize, got: usize) -> Result<(), CodecError> {
    if got == expected {
        Ok(())
    } else {
        Err(CodecError::DimensionMismatch { expected, got })
    }
}

fn check_payload(expected: usize, got: usize) -> Result<(), CodecError> {
    if got == expected {
        Ok(())
    } else {
        Err(CodecError::PayloadSize { expected, got })
    }
}

// ---------------------------------------------------------------------------
// f32 baseline
// ---------------------------------------------------------------------------

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
        check_dim(self.dimension, vector.len())?;
        Ok(Encoded(vector.iter().flat_map(|v| v.to_le_bytes()).collect()))
    }

    fn decode(&self, encoded: &Encoded) -> Result<Vector, CodecError> {
        check_payload(self.encoded_size(), encoded.0.len())?;
        Ok(encoded
            .0
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn score(&self, query: &[f32], encoded: &Encoded, metric: Metric) -> Result<f32, CodecError> {
        check_dim(self.dimension, query.len())?;
        // Baseline: decoding is the documented behaviour here.
        let stored = self.decode(encoded)?;
        Ok(distance::score(query, &stored, metric))
    }
}

// ---------------------------------------------------------------------------
// int8 scalar quantization
// ---------------------------------------------------------------------------

/// Per-vector affine scalar quantization to 8 bits.
///
/// Layout: `[min: f32 LE][scale: f32 LE][q_0 .. q_{d-1}: u8]` where the stored
/// value is `min + scale * q_i`. About 4× smaller than f32 for typical
/// dimensions (`4d / (d + 8)`).
///
/// Error model: each component is reconstructed within `scale / 2`, where
/// `scale = (max - min) / 255` of that vector. Scores are computed in one pass
/// over the bytes using the identities below; nothing is decoded.
#[derive(Debug, Clone)]
pub struct Int8Codec {
    dimension: usize,
}

const INT8_HEADER: usize = 8;

impl Int8Codec {
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn split(encoded: &[u8]) -> (f32, f32, &[u8]) {
        let min = f32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        let scale = f32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]);
        (min, scale, &encoded[INT8_HEADER..])
    }
}

impl Codec for Int8Codec {
    fn name(&self) -> &'static str {
        "int8"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn encoded_size(&self) -> usize {
        INT8_HEADER + self.dimension
    }

    fn supported_metrics(&self) -> &'static [Metric] {
        &[Metric::Dot, Metric::Cosine, Metric::L2]
    }

    fn encode(&self, vector: &[f32]) -> Result<Encoded, CodecError> {
        check_dim(self.dimension, vector.len())?;
        let (min, max) = vector.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        });
        let (min, max) = if min.is_finite() { (min, max) } else { (0.0, 0.0) };
        let range = max - min;
        let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
        let mut out = Vec::with_capacity(self.encoded_size());
        out.extend_from_slice(&min.to_le_bytes());
        out.extend_from_slice(&scale.to_le_bytes());
        out.extend(
            vector
                .iter()
                .map(|&x| ((x - min) / scale).round().clamp(0.0, 255.0) as u8),
        );
        Ok(Encoded(out))
    }

    fn decode(&self, encoded: &Encoded) -> Result<Vector, CodecError> {
        check_payload(self.encoded_size(), encoded.0.len())?;
        let (min, scale, q) = Self::split(&encoded.0);
        Ok(q.iter().map(|&b| min + scale * f32::from(b)).collect())
    }

    fn score(&self, query: &[f32], encoded: &Encoded, metric: Metric) -> Result<f32, CodecError> {
        check_dim(self.dimension, query.len())?;
        check_payload(self.encoded_size(), encoded.0.len())?;
        let (min, scale, q) = Self::split(&encoded.0);
        let (min, scale) = (f64::from(min), f64::from(scale));

        // One pass: everything below is derived from these five sums.
        let (mut sum_q, mut sum_qq, mut dot_qx, mut sum_x, mut sum_xx) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (&b, &x) in q.iter().zip(query) {
            let qi = f64::from(b);
            let x = f64::from(x);
            sum_q += qi;
            sum_qq += qi * qi;
            dot_qx += qi * x;
            sum_x += x;
            sum_xx += x * x;
        }
        let d = self.dimension as f64;
        // stored_i = min + scale * q_i
        let dot = min * sum_x + scale * dot_qx;
        let stored_norm_sq = d * min * min + 2.0 * min * scale * sum_q + scale * scale * sum_qq;

        let v = match metric {
            Metric::Dot => dot,
            Metric::Cosine => {
                let denom = sum_xx.sqrt() * stored_norm_sq.max(0.0).sqrt();
                if denom == 0.0 { 0.0 } else { dot / denom }
            }
            Metric::L2 => (sum_xx - 2.0 * dot + stored_norm_sq).max(0.0),
        };
        Ok(v as f32)
    }
}

// ---------------------------------------------------------------------------
// binary (sign) quantization
// ---------------------------------------------------------------------------

/// One bit per component: the sign. 32× smaller than f32.
///
/// Layout: `ceil(d / 8)` bytes, component `i` is bit `i % 8` of byte `i / 8`,
/// set when the value is positive. The decoded vector is `±1/√d`, so it has
/// unit length.
///
/// Error model: approximate. For unit-length inputs the asymmetric score
/// `Σ q_i · sign_i / √d` is a rank-preserving proxy for the inner product
/// (the SimHash argument). Magnitudes are lost entirely, so `L2` and `Dot`
/// on unnormalised data are only meaningful for ranking, not as distances.
#[derive(Debug, Clone)]
pub struct BinaryCodec {
    dimension: usize,
}

impl BinaryCodec {
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn inv_sqrt_d(&self) -> f64 {
        1.0 / (self.dimension as f64).sqrt()
    }
}

impl Codec for BinaryCodec {
    fn name(&self) -> &'static str {
        "binary"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn encoded_size(&self) -> usize {
        self.dimension.div_ceil(8)
    }

    fn supported_metrics(&self) -> &'static [Metric] {
        &[Metric::Dot, Metric::Cosine, Metric::L2]
    }

    fn encode(&self, vector: &[f32]) -> Result<Encoded, CodecError> {
        check_dim(self.dimension, vector.len())?;
        let mut out = vec![0u8; self.encoded_size()];
        for (i, &x) in vector.iter().enumerate() {
            if x > 0.0 {
                out[i / 8] |= 1 << (i % 8);
            }
        }
        Ok(Encoded(out))
    }

    fn decode(&self, encoded: &Encoded) -> Result<Vector, CodecError> {
        check_payload(self.encoded_size(), encoded.0.len())?;
        let mag = self.inv_sqrt_d() as f32;
        Ok((0..self.dimension)
            .map(|i| {
                if encoded.0[i / 8] >> (i % 8) & 1 == 1 {
                    mag
                } else {
                    -mag
                }
            })
            .collect())
    }

    fn score(&self, query: &[f32], encoded: &Encoded, metric: Metric) -> Result<f32, CodecError> {
        check_dim(self.dimension, query.len())?;
        check_payload(self.encoded_size(), encoded.0.len())?;
        let (mut signed, mut sum_xx) = (0f64, 0f64);
        for (i, &x) in query.iter().enumerate() {
            let x = f64::from(x);
            let positive = encoded.0[i / 8] >> (i % 8) & 1 == 1;
            signed += if positive { x } else { -x };
            sum_xx += x * x;
        }
        // dot(query, decoded) where decoded = ±1/√d and |decoded| = 1
        let dot = signed * self.inv_sqrt_d();
        let v = match metric {
            Metric::Dot => dot,
            Metric::Cosine => {
                let qn = sum_xx.sqrt();
                if qn == 0.0 { 0.0 } else { dot / qn }
            }
            Metric::L2 => (sum_xx - 2.0 * dot + 1.0).max(0.0),
        };
        Ok(v as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic generator so tests need no external crate.
    struct XorShift(u64);
    impl XorShift {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            // uniform in [-1, 1)
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
        fn vector(&mut self, dim: usize) -> Vec<f32> {
            (0..dim).map(|_| self.next_f32()).collect()
        }
    }

    fn all_codecs(dim: usize) -> Vec<Box<dyn Codec>> {
        vec![
            Box::new(F32Codec::new(dim)),
            Box::new(Int8Codec::new(dim)),
            Box::new(BinaryCodec::new(dim)),
        ]
    }

    #[test]
    fn f32_roundtrip_is_exact() {
        let codec = F32Codec::new(3);
        let v = vec![0.1, -2.5, 1e-7];
        let enc = codec.encode(&v).unwrap();
        assert_eq!(enc.0.len(), 12);
        assert_eq!(codec.decode(&enc).unwrap(), v);
    }

    #[test]
    fn every_codec_rejects_wrong_dimension_and_payload() {
        for codec in all_codecs(16) {
            assert!(matches!(
                codec.encode(&[1.0; 15]),
                Err(CodecError::DimensionMismatch { expected: 16, got: 15 })
            ));
            let bad = Encoded(vec![0u8; codec.encoded_size() + 1]);
            assert!(matches!(codec.decode(&bad), Err(CodecError::PayloadSize { .. })));
            assert!(codec.score(&[0.0; 16], &bad, Metric::Dot).is_err());
        }
    }

    #[test]
    fn encoded_sizes_and_ratios() {
        assert_eq!(F32Codec::new(384).encoded_size(), 1536);
        assert_eq!(Int8Codec::new(384).encoded_size(), 392);
        assert_eq!(BinaryCodec::new(384).encoded_size(), 48);
        assert_eq!(BinaryCodec::new(13).encoded_size(), 2);
        assert_eq!(F32Codec::new(128).compression_ratio(), 1.0);
        assert!((Int8Codec::new(384).compression_ratio() - 3.918).abs() < 0.01);
        assert_eq!(BinaryCodec::new(384).compression_ratio(), 32.0);
    }

    #[test]
    fn score_is_consistent_with_own_decode() {
        // For every codec and metric, scoring in the compressed domain must
        // equal the reference distance computed on the decoded vector.
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        for dim in [8, 33, 384] {
            for codec in all_codecs(dim) {
                for _ in 0..20 {
                    let stored = rng.vector(dim);
                    let query = rng.vector(dim);
                    let enc = codec.encode(&stored).unwrap();
                    let dec = codec.decode(&enc).unwrap();
                    for metric in [Metric::Dot, Metric::Cosine, Metric::L2] {
                        let fast = codec.score(&query, &enc, metric).unwrap();
                        let reference = distance::score(&query, &dec, metric);
                        let tol = 1e-4 * (1.0 + reference.abs());
                        assert!(
                            (fast - reference).abs() <= tol,
                            "{} {metric} dim {dim}: {fast} vs {reference}",
                            codec.name()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn int8_reconstruction_error_is_within_half_a_step() {
        let mut rng = XorShift(42);
        let codec = Int8Codec::new(256);
        for _ in 0..50 {
            let v = rng.vector(256);
            let enc = codec.encode(&v).unwrap();
            let (min, scale, _) = Int8Codec::split(&enc.0);
            let dec = codec.decode(&enc).unwrap();
            for (a, b) in v.iter().zip(&dec) {
                assert!(
                    (a - b).abs() <= scale / 2.0 + 1e-6,
                    "min {min} scale {scale}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn int8_scores_approximate_f32_closely() {
        let mut rng = XorShift(7);
        let dim = 384;
        let codec = Int8Codec::new(dim);
        let mut worst = 0f32;
        for _ in 0..100 {
            let mut stored = rng.vector(dim);
            let mut query = rng.vector(dim);
            distance::normalize(&mut stored);
            distance::normalize(&mut query);
            let enc = codec.encode(&stored).unwrap();
            let exact = distance::cosine(&query, &stored);
            let approx = codec.score(&query, &enc, Metric::Cosine).unwrap();
            worst = worst.max((exact - approx).abs());
        }
        assert!(worst < 0.01, "worst cosine error {worst}");
    }

    #[test]
    fn int8_handles_constant_vectors() {
        let codec = Int8Codec::new(4);
        let enc = codec.encode(&[0.5; 4]).unwrap();
        assert_eq!(codec.decode(&enc).unwrap(), vec![0.5; 4]);
        assert!((codec.score(&[1.0; 4], &enc, Metric::Dot).unwrap() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn binary_keeps_signs_and_is_unit_length() {
        let codec = BinaryCodec::new(10);
        let v = [0.3, -0.1, 0.0, 2.0, -5.0, 0.01, -0.01, 1.0, 1.0, -1.0];
        let enc = codec.encode(&v).unwrap();
        assert_eq!(enc.0.len(), 2);
        let dec = codec.decode(&enc).unwrap();
        for (a, b) in v.iter().zip(&dec) {
            assert_eq!(*a > 0.0, *b > 0.0, "sign mismatch for {a}");
        }
        assert!((distance::norm(&dec) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn binary_ranks_like_f32_on_well_separated_data() {
        // Two clusters; binary cosine must put same-cluster items first.
        let mut rng = XorShift(99);
        let dim = 128;
        let codec = BinaryCodec::new(dim);
        let centre_a: Vec<f32> = rng.vector(dim);
        let centre_b: Vec<f32> = rng.vector(dim);
        let jitter = |rng: &mut XorShift, c: &[f32]| -> Vec<f32> {
            let mut v: Vec<f32> = c.iter().map(|x| x + 0.2 * rng.next_f32()).collect();
            distance::normalize(&mut v);
            v
        };
        let query = jitter(&mut rng, &centre_a);
        let same = jitter(&mut rng, &centre_a);
        let other = jitter(&mut rng, &centre_b);
        let s_same = codec
            .score(&query, &codec.encode(&same).unwrap(), Metric::Cosine)
            .unwrap();
        let s_other = codec
            .score(&query, &codec.encode(&other).unwrap(), Metric::Cosine)
            .unwrap();
        assert!(s_same > s_other, "{s_same} vs {s_other}");
        assert!(s_same > 0.5, "same-cluster binary cosine should be high: {s_same}");
    }

    #[test]
    fn codec_by_name_knows_all_three() {
        for name in crate::codec_names() {
            let c = crate::codec_by_name(name, 32).unwrap();
            assert_eq!(c.name(), *name);
        }
        assert!(matches!(
            crate::codec_by_name("pq", 32),
            Err(CodecError::UnknownCodec(_))
        ));
    }
}
