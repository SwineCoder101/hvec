//! Product quantization with trained codebooks.

use crate::{Codec, CodecError, Encoded, Metric, Vector, distance};

use super::{check_dim, check_payload};

/// Centroids per subspace. Codes are one byte, so this is the maximum.
const K: usize = 256;
/// Lloyd iterations per subspace during [`Codec::fit`].
const ITERATIONS: usize = 20;
const HEADER: usize = 12;

/// Product quantization: the vector is cut into `m` equal subspaces and each
/// piece is replaced by the index of its nearest centroid in a codebook of
/// up to 256 learned by k-means on the fitting sample. One byte per subspace.
///
/// Layout: `m` bytes, byte `j` is the code of subspace `j`. The codebooks,
/// `m × k × (d / m)` little-endian f32 after a `[m: u32][k: u32][sub: u32]`
/// header, are stored once per collection through [`Codec::params`].
///
/// Error model: approximate. `score` sums, over subspaces, the metric's
/// partial terms between the query piece and the chosen centroid, so the
/// result equals the reference distance on the decoded vector. It walks the
/// centroids directly, `d` multiply-adds per pair like f32; a per-query
/// lookup table (asymmetric distance computation, `m` adds per pair) is the
/// production form and is deliberately not implemented here.
#[derive(Debug, Clone)]
pub struct PqCodec {
    dimension: usize,
    m: usize,
    sub: usize,
    name: String,
    fitted: Option<Codebooks>,
}

#[derive(Debug, Clone)]
struct Codebooks {
    k: usize,
    /// `centroids[(j * k + c) * sub .. +sub]` is centroid `c` of subspace `j`.
    centroids: Vec<f32>,
    /// `‖centroid‖²` in the same order, for the decoded vector's norm.
    norms_sq: Vec<f64>,
}

impl Codebooks {
    fn new(k: usize, sub: usize, centroids: Vec<f32>) -> Self {
        let norms_sq = centroids
            .chunks_exact(sub)
            .map(|c| c.iter().map(|&x| f64::from(x) * f64::from(x)).sum())
            .collect();
        Self { k, centroids, norms_sq }
    }

    fn centroid(&self, sub: usize, j: usize, c: usize) -> &[f32] {
        let start = (j * self.k + c) * sub;
        &self.centroids[start..start + sub]
    }
}

impl PqCodec {
    /// `m` subspaces; `dimension` must be a positive multiple of `m`.
    pub fn new(dimension: usize, m: usize) -> Result<Self, CodecError> {
        if m == 0 || dimension == 0 || dimension % m != 0 {
            return Err(CodecError::BadParams {
                codec: format!("pq{m}"),
                reason: format!("dimension {dimension} is not a positive multiple of {m} subspaces"),
            });
        }
        Ok(Self {
            dimension,
            m,
            sub: dimension / m,
            name: format!("pq{m}"),
            fitted: None,
        })
    }

    /// Number of subspaces, one byte each.
    #[must_use]
    pub fn subspaces(&self) -> usize {
        self.m
    }

    /// Centroids per subspace once fitted.
    #[must_use]
    pub fn centroids_per_subspace(&self) -> Option<usize> {
        self.fitted.as_ref().map(|c| c.k)
    }

    fn books(&self) -> Result<&Codebooks, CodecError> {
        self.fitted
            .as_ref()
            .ok_or_else(|| CodecError::NotFitted(self.name.clone()))
    }

    fn nearest(&self, books: &Codebooks, j: usize, piece: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..books.k {
            let d = distance::l2_squared(piece, books.centroid(self.sub, j, c));
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        best
    }
}

/// k-means on `points` (each `sub` long), `k` centroids, deterministic
/// initialisation from evenly spaced sample points.
fn kmeans(points: &[&[f32]], sub: usize, k: usize) -> Vec<f32> {
    let n = points.len();
    let mut centroids: Vec<f32> = Vec::with_capacity(k * sub);
    for c in 0..k {
        let idx = (c * n) / k;
        centroids.extend_from_slice(points[idx]);
    }
    let mut assign = vec![usize::MAX; n];
    for _ in 0..ITERATIONS {
        let mut changed = false;
        for (i, p) in points.iter().enumerate() {
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let d = distance::l2_squared(p, &centroids[c * sub..(c + 1) * sub]);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            if assign[i] != best {
                assign[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        let mut sums = vec![0f64; k * sub];
        let mut counts = vec![0usize; k];
        for (i, p) in points.iter().enumerate() {
            let c = assign[i];
            counts[c] += 1;
            for (s, &x) in sums[c * sub..(c + 1) * sub].iter_mut().zip(*p) {
                *s += f64::from(x);
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Empty cluster: keep the old centroid.
                continue;
            }
            for d in 0..sub {
                centroids[c * sub + d] = (sums[c * sub + d] / counts[c] as f64) as f32;
            }
        }
    }
    centroids
}

impl Codec for PqCodec {
    fn name(&self) -> &str {
        &self.name
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn encoded_size(&self) -> usize {
        self.m
    }

    fn supported_metrics(&self) -> &'static [Metric] {
        &[Metric::Dot, Metric::Cosine, Metric::L2]
    }

    fn needs_fit(&self) -> bool {
        self.fitted.is_none()
    }

    fn fit(&mut self, sample: &[Vector]) -> Result<(), CodecError> {
        if sample.is_empty() {
            return Err(CodecError::EmptySample {
                codec: self.name.clone(),
            });
        }
        for v in sample {
            check_dim(self.dimension, v.len())?;
        }
        let k = K.min(sample.len());
        let mut centroids = Vec::with_capacity(self.m * k * self.sub);
        for j in 0..self.m {
            let pieces: Vec<&[f32]> = sample.iter().map(|v| &v[j * self.sub..(j + 1) * self.sub]).collect();
            centroids.extend(kmeans(&pieces, self.sub, k));
        }
        self.fitted = Some(Codebooks::new(k, self.sub, centroids));
        Ok(())
    }

    fn params(&self) -> Vec<u8> {
        let Some(b) = &self.fitted else { return Vec::new() };
        let mut out = Vec::with_capacity(HEADER + b.centroids.len() * 4);
        out.extend_from_slice(&(self.m as u32).to_le_bytes());
        out.extend_from_slice(&(b.k as u32).to_le_bytes());
        out.extend_from_slice(&(self.sub as u32).to_le_bytes());
        out.extend(b.centroids.iter().flat_map(|v| v.to_le_bytes()));
        out
    }

    fn restore(&mut self, params: &[u8]) -> Result<(), CodecError> {
        let bad = |reason: String| CodecError::BadParams {
            codec: self.name.clone(),
            reason,
        };
        if params.len() < HEADER {
            return Err(bad(format!("{} bytes is too short for a header", params.len())));
        }
        let u = |i: usize| u32::from_le_bytes([params[i], params[i + 1], params[i + 2], params[i + 3]]) as usize;
        let (m, k, sub) = (u(0), u(4), u(8));
        if m != self.m || sub != self.sub {
            return Err(bad(format!("parameters are for pq{m} over {} dims", m * sub)));
        }
        if k == 0 || k > K {
            return Err(bad(format!("{k} centroids per subspace")));
        }
        let expected = HEADER + m * k * sub * 4;
        if params.len() != expected {
            return Err(bad(format!("expected {expected} bytes, got {}", params.len())));
        }
        let centroids: Vec<f32> = params[HEADER..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if centroids.iter().any(|v| !v.is_finite()) {
            return Err(bad("non-finite centroid".to_owned()));
        }
        self.fitted = Some(Codebooks::new(k, sub, centroids));
        Ok(())
    }

    fn encode(&self, vector: &[f32]) -> Result<Encoded, CodecError> {
        check_dim(self.dimension, vector.len())?;
        let books = self.books()?;
        let codes = (0..self.m)
            .map(|j| self.nearest(books, j, &vector[j * self.sub..(j + 1) * self.sub]) as u8)
            .collect();
        Ok(Encoded(codes))
    }

    fn decode(&self, encoded: &Encoded) -> Result<Vector, CodecError> {
        check_payload(self.encoded_size(), encoded.0.len())?;
        let books = self.books()?;
        let mut out = Vec::with_capacity(self.dimension);
        for (j, &code) in encoded.0.iter().enumerate() {
            let c = usize::from(code);
            if c >= books.k {
                return Err(CodecError::BadParams {
                    codec: self.name.clone(),
                    reason: format!("code {c} in subspace {j} exceeds {} centroids", books.k),
                });
            }
            out.extend_from_slice(books.centroid(self.sub, j, c));
        }
        Ok(out)
    }

    fn score(&self, query: &[f32], encoded: &Encoded, metric: Metric) -> Result<f32, CodecError> {
        check_dim(self.dimension, query.len())?;
        check_payload(self.encoded_size(), encoded.0.len())?;
        let books = self.books()?;
        let (mut dot, mut stored_norm_sq, mut sum_qq) = (0f64, 0f64, 0f64);
        for (j, &code) in encoded.0.iter().enumerate() {
            let c = usize::from(code);
            if c >= books.k {
                return Err(CodecError::BadParams {
                    codec: self.name.clone(),
                    reason: format!("code {c} in subspace {j} exceeds {} centroids", books.k),
                });
            }
            let q = &query[j * self.sub..(j + 1) * self.sub];
            let cen = books.centroid(self.sub, j, c);
            for (&x, &y) in q.iter().zip(cen) {
                dot += f64::from(x) * f64::from(y);
                sum_qq += f64::from(x) * f64::from(x);
            }
            stored_norm_sq += books.norms_sq[j * books.k + c];
        }
        let v = match metric {
            Metric::Dot => dot,
            Metric::Cosine => {
                let denom = (sum_qq * stored_norm_sq).sqrt();
                if denom == 0.0 { 0.0 } else { dot / denom }
            }
            Metric::L2 => (sum_qq - 2.0 * dot + stored_norm_sq).max(0.0),
        };
        Ok(v as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct XorShift(u64);
    impl XorShift {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
        fn vector(&mut self, dim: usize) -> Vec<f32> {
            (0..dim).map(|_| self.next_f32()).collect()
        }
    }

    fn clustered(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = XorShift(77);
        let centres: Vec<Vec<f32>> = (0..6).map(|_| rng.vector(dim)).collect();
        (0..n)
            .map(|i| {
                let mut v: Vec<f32> = centres[i % 6].iter().map(|x| x + 0.1 * rng.next_f32()).collect();
                distance::normalize(&mut v);
                v
            })
            .collect()
    }

    #[test]
    fn rejects_bad_subspace_counts() {
        assert!(matches!(PqCodec::new(32, 0), Err(CodecError::BadParams { .. })));
        assert!(matches!(PqCodec::new(32, 5), Err(CodecError::BadParams { .. })));
        assert!(matches!(PqCodec::new(0, 1), Err(CodecError::BadParams { .. })));
        let c = PqCodec::new(32, 8).unwrap();
        assert_eq!(c.name(), "pq8");
        assert_eq!(c.encoded_size(), 8);
        assert_eq!(c.compression_ratio(), 16.0);
        assert!(c.needs_fit());
        assert!(matches!(c.encode(&[0.0; 32]), Err(CodecError::NotFitted(_))));
    }

    #[test]
    fn small_samples_use_fewer_centroids_and_reconstruct_exactly() {
        let sample = clustered(6, 16);
        let mut c = PqCodec::new(16, 4).unwrap();
        c.fit(&sample).unwrap();
        assert_eq!(c.centroids_per_subspace(), Some(6));
        for v in &sample {
            let dec = c.decode(&c.encode(v).unwrap()).unwrap();
            assert!(distance::l2_squared(v, &dec) < 1e-10);
        }
    }

    #[test]
    fn fitted_pq_scores_close_to_exact_on_clusters() {
        let sample = clustered(600, 32);
        let mut c = PqCodec::new(32, 8).unwrap();
        c.fit(&sample).unwrap();
        assert_eq!(c.centroids_per_subspace(), Some(256));
        let mut rng = XorShift(5);
        let mut worst = 0f32;
        for _ in 0..50 {
            let q = {
                let mut q = rng.vector(32);
                distance::normalize(&mut q);
                q
            };
            let v = &sample[rng.next_f32().abs().mul_add(599.0, 0.0) as usize];
            let enc = c.encode(v).unwrap();
            let approx = c.score(&q, &enc, Metric::Cosine).unwrap();
            let exact = distance::cosine(&q, v);
            worst = worst.max((approx - exact).abs());
            // Internal consistency with the decoded vector, every metric.
            let dec = c.decode(&enc).unwrap();
            for metric in [Metric::Dot, Metric::Cosine, Metric::L2] {
                let r = distance::score(&q, &dec, metric);
                assert!((c.score(&q, &enc, metric).unwrap() - r).abs() < 1e-4, "{metric}");
            }
        }
        assert!(worst < 0.1, "worst cosine error {worst}");
    }

    #[test]
    fn params_roundtrip_and_are_validated() {
        let sample = clustered(300, 24);
        let mut c = PqCodec::new(24, 6).unwrap();
        c.fit(&sample).unwrap();
        let params = c.params();
        assert_eq!(params.len(), HEADER + 6 * 256 * 4 * 4);
        let restored = crate::codec_with_params("pq6", 24, &params).unwrap();
        assert!(!restored.needs_fit());
        assert_eq!(restored.params(), params);
        let v = &sample[7];
        assert_eq!(restored.encode(v).unwrap(), c.encode(v).unwrap());

        assert!(matches!(
            crate::codec_with_params("pq6", 24, &[]),
            Err(CodecError::BadParams { .. })
        ));
        assert!(matches!(
            crate::codec_with_params("pq4", 24, &params),
            Err(CodecError::BadParams { .. })
        ));
        assert!(matches!(
            crate::codec_with_params("pq6", 24, &params[..HEADER + 7]),
            Err(CodecError::BadParams { .. })
        ));
        // A code beyond the fitted centroid count is rejected at decode and score time.
        let few = {
            let mut f = PqCodec::new(24, 6).unwrap();
            f.fit(&sample[..3]).unwrap();
            f
        };
        let bad = Encoded(vec![9u8; 6]);
        assert!(matches!(few.decode(&bad), Err(CodecError::BadParams { .. })));
        assert!(matches!(
            few.score(v, &bad, Metric::Dot),
            Err(CodecError::BadParams { .. })
        ));
    }
}
