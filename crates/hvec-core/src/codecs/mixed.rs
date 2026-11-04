//! Mixed-precision scalar quantization: a fixed byte budget per vector, with
//! bits allocated per dimension by variance.
//!
//! Uniform codecs spend the same bits on every dimension: one for `binary`,
//! eight for `int8`. `mixed<B>` spends `8·B` bits in all, giving each
//! dimension between 0 and 8 by greedy reverse water-filling on its variance:
//! each extra bit roughly quarters a dimension's quantization error, so the
//! next bit always goes where it removes the most error. This is the classic
//! transform-coding allocation (Huang and Schultheiss, 1963), and the idea
//! mixed-precision model quantization applies to weights, where structures
//! that matter more get more bits.
//!
//! `mixed<B>-pca` first rotates into the corpus's principal components, which
//! concentrates the variance in few dimensions and gives the allocation
//! something to work with; raw embedding dimensions have fairly even variance.
//! The rotation is orthogonal, so dot products and norms are preserved, and
//! the query is rotated once per search through [`Codec::prepare_query`].
//!
//! `-w` changes what the allocation minimises from reconstruction error to
//! expected score error (see `fit`).
//!
//! Each dimension is quantized uniformly around its mean with the step that
//! minimises mean squared error for a Gaussian of its standard deviation
//! (Max, 1960). Error model: approximate for every metric; the error falls as
//! the budget grows and is zero only in the limit.

use crate::codecs::{check_dim, check_payload};
use crate::{Codec, CodecError, Encoded, Metric, Vector};

/// Optimal uniform quantizer step for a unit Gaussian, by bit depth 1..=8
/// (Max 1960, Table II). Index 0 is unused: a 0-bit dimension decodes to its mean.
const GAUSSIAN_STEP: [f64; 9] = [0.0, 1.596, 0.9957, 0.5860, 0.3352, 0.1881, 0.1041, 0.0569, 0.0308];
const MAX_BITS: u8 = 8;
/// Vectors used to fit; larger samples are thinned evenly.
const FIT_SAMPLE: usize = 20_000;

/// Fixed byte budget, bits allocated per dimension by variance.
#[derive(Debug, Clone)]
pub struct MixedCodec {
    name: String,
    dimension: usize,
    bytes: usize,
    pca: bool,
    weighted: bool,
    fitted: Option<Fitted>,
}

#[derive(Debug, Clone)]
struct Fitted {
    /// Row-major `d × d`; row `j` is component `j`. `None` without PCA.
    rotation: Option<Vec<f32>>,
    /// Per (rotated) dimension.
    mean: Vec<f32>,
    step: Vec<f32>,
    bits: Vec<u8>,
}

impl MixedCodec {
    /// `bytes` per vector, optionally after a PCA rotation, with bits
    /// allocated by variance or (`weighted`) by variance × query energy.
    pub fn new(dimension: usize, bytes: usize, pca: bool, weighted: bool) -> Result<Self, CodecError> {
        let name = format!(
            "mixed{bytes}{}{}",
            if pca { "-pca" } else { "" },
            if weighted { "-w" } else { "" }
        );
        if dimension == 0 || bytes == 0 || bytes * 8 > dimension * usize::from(MAX_BITS) {
            return Err(CodecError::BadParams {
                codec: name,
                reason: format!("{bytes} bytes cannot be spread over {dimension} dimensions at 0 to 8 bits each"),
            });
        }
        Ok(Self {
            name,
            dimension,
            bytes,
            pca,
            weighted,
            fitted: None,
        })
    }

    /// Bits given to each (rotated) dimension, once fitted.
    #[must_use]
    pub fn bits(&self) -> Option<&[u8]> {
        self.fitted.as_ref().map(|f| f.bits.as_slice())
    }

    fn state(&self) -> Result<&Fitted, CodecError> {
        self.fitted
            .as_ref()
            .ok_or_else(|| CodecError::NotFitted(self.name.clone()))
    }

    fn rotate(f: &Fitted, x: &[f32]) -> Vector {
        match &f.rotation {
            None => x.to_vec(),
            Some(r) => {
                let d = x.len();
                (0..d)
                    .map(|j| {
                        let row = &r[j * d..(j + 1) * d];
                        row.iter()
                            .zip(x)
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .sum::<f64>() as f32
                    })
                    .collect()
            }
        }
    }

    /// Calls `f(dimension, reconstructed value)` for every dimension.
    fn for_each_value(fit: &Fitted, encoded: &Encoded, mut f: impl FnMut(usize, f64)) {
        let mut reader = BitReader::new(&encoded.0);
        for (i, &b) in fit.bits.iter().enumerate() {
            let m = f64::from(fit.mean[i]);
            if b == 0 {
                f(i, m);
                continue;
            }
            let code = reader.read(b);
            let half = f64::from((1u32 << b) - 1) / 2.0;
            f(i, m + (f64::from(code) - half) * f64::from(fit.step[i]));
        }
    }
}

/// Greedy reverse water-filling: `budget` bits over dimensions with these
/// variances, at most [`MAX_BITS`] each.
fn allocate(variance: &[f64], budget: usize) -> Vec<u8> {
    let mut bits = vec![0u8; variance.len()];
    for _ in 0..budget {
        // Error left in a dimension at b bits is about var / 4^b.
        let best = (0..variance.len()).filter(|&i| bits[i] < MAX_BITS).max_by(|&a, &b| {
            let ea = variance[a] / 4f64.powi(i32::from(bits[a]));
            let eb = variance[b] / 4f64.powi(i32::from(bits[b]));
            ea.total_cmp(&eb).then(b.cmp(&a))
        });
        match best {
            Some(i) => bits[i] += 1,
            None => break,
        }
    }
    bits
}

/// Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations.
/// Returns `(eigenvalues, eigenvectors)`, eigenvectors as rows, sorted by
/// decreasing eigenvalue.
fn symmetric_eigen(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let scale: f64 = (0..n).map(|i| a[i * n + i].abs()).sum::<f64>().max(f64::MIN_POSITIVE);
    for _sweep in 0..60 {
        let off: f64 = (0..n)
            .flat_map(|p| ((p + 1)..n).map(move |q| (p, q)))
            .map(|(p, q)| a[p * n + q] * a[p * n + q])
            .sum();
        if off.sqrt() < 1e-12 * scale {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                // Columns of V are eigenvectors.
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| a[y * n + y].total_cmp(&a[x * n + x]));
    let values = order.iter().map(|&j| a[j * n + j]).collect();
    let mut rows = vec![0f64; n * n];
    for (r, &j) in order.iter().enumerate() {
        for k in 0..n {
            rows[r * n + k] = v[k * n + j];
        }
    }
    (values, rows)
}

struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    filled: u32,
}

impl BitWriter {
    fn new(bytes: usize) -> Self {
        Self {
            out: Vec::with_capacity(bytes),
            acc: 0,
            filled: 0,
        }
    }
    fn write(&mut self, value: u32, bits: u8) {
        self.acc |= u64::from(value) << self.filled;
        self.filled += u32::from(bits);
        while self.filled >= 8 {
            self.out.push((self.acc & 0xff) as u8);
            self.acc >>= 8;
            self.filled -= 8;
        }
    }
    fn finish(mut self, bytes: usize) -> Vec<u8> {
        if self.filled > 0 {
            self.out.push((self.acc & 0xff) as u8);
        }
        self.out.resize(bytes, 0);
        self.out
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u64,
    filled: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            acc: 0,
            filled: 0,
        }
    }
    fn read(&mut self, bits: u8) -> u32 {
        while self.filled < u32::from(bits) {
            let byte = self.data.get(self.pos).copied().unwrap_or(0);
            self.acc |= u64::from(byte) << self.filled;
            self.pos += 1;
            self.filled += 8;
        }
        let v = (self.acc & ((1u64 << bits) - 1)) as u32;
        self.acc >>= bits;
        self.filled -= u32::from(bits);
        v
    }
}

impl Codec for MixedCodec {
    fn name(&self) -> &str {
        &self.name
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn encoded_size(&self) -> usize {
        self.bytes
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
        let d = self.dimension;
        let step = (sample.len() as f64 / FIT_SAMPLE as f64).max(1.0);
        let picked: Vec<&Vector> = (0..sample.len().min(FIT_SAMPLE))
            .map(|i| &sample[((i as f64 * step) as usize).min(sample.len() - 1)])
            .collect();
        for v in &picked {
            check_dim(d, v.len())?;
        }
        let n = picked.len() as f64;
        let mut mean = vec![0f64; d];
        for v in &picked {
            for (m, &x) in mean.iter_mut().zip(v.iter()) {
                *m += f64::from(x) / n;
            }
        }
        let rotation = if self.pca {
            let mut cov = vec![0f64; d * d];
            let mut c = vec![0f64; d];
            for v in &picked {
                for (ci, (&x, &m)) in c.iter_mut().zip(v.iter().zip(&mean)) {
                    *ci = f64::from(x) - m;
                }
                for i in 0..d {
                    let ci = c[i];
                    for j in i..d {
                        cov[i * d + j] += ci * c[j];
                    }
                }
            }
            for i in 0..d {
                for j in i..d {
                    cov[i * d + j] /= n;
                    cov[j * d + i] = cov[i * d + j];
                }
            }
            let (_, rows) = symmetric_eigen(cov, d);
            Some(rows.into_iter().map(|x| x as f32).collect::<Vec<f32>>())
        } else {
            None
        };
        let probe = Fitted {
            rotation,
            mean: vec![0.0; d],
            step: vec![0.0; d],
            bits: vec![0; d],
        };
        // Statistics in the (possibly rotated) space.
        let rotated: Vec<Vector> = picked.iter().map(|v| Self::rotate(&probe, v)).collect();
        let mut rmean = vec![0f64; d];
        for y in &rotated {
            for (m, &x) in rmean.iter_mut().zip(y) {
                *m += f64::from(x) / n;
            }
        }
        let mut var = vec![0f64; d];
        for y in &rotated {
            for ((s, &x), &m) in var.iter_mut().zip(y).zip(&rmean) {
                *s += (f64::from(x) - m).powi(2) / n;
            }
        }
        // Plain allocation minimises reconstruction error, Σ Var(e_i). But a
        // dimension's error reaches the score through the query: E[(q·e)²] =
        // Σ E[q_i²]·Var(e_i). With `-w`, queries are assumed to look like the
        // corpus and each dimension's variance is weighted by its mean
        // energy, var + mean².
        let importance: Vec<f64> = if self.weighted {
            var.iter().zip(&rmean).map(|(&v, &m)| v * (v + m * m)).collect()
        } else {
            var.clone()
        };
        let bits = allocate(&importance, self.bytes * 8);
        let step = bits
            .iter()
            .zip(&var)
            .map(|(&b, &v)| (GAUSSIAN_STEP[usize::from(b)] * v.sqrt()) as f32)
            .collect();
        self.fitted = Some(Fitted {
            rotation: probe.rotation,
            mean: rmean.iter().map(|&m| m as f32).collect(),
            step,
            bits,
        });
        Ok(())
    }

    fn params(&self) -> Vec<u8> {
        let Some(f) = &self.fitted else {
            return Vec::new();
        };
        let mut out = vec![u8::from(f.rotation.is_some())];
        out.extend_from_slice(&f.bits);
        for x in f.mean.iter().chain(&f.step).chain(f.rotation.iter().flatten()) {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    fn restore(&mut self, params: &[u8]) -> Result<(), CodecError> {
        let d = self.dimension;
        let bad = |reason: String| CodecError::BadParams {
            codec: self.name.clone(),
            reason,
        };
        let floats = 2 * d + if self.pca { d * d } else { 0 };
        let expected = 1 + d + 4 * floats;
        if params.len() != expected || params[0] != u8::from(self.pca) {
            return Err(bad(format!(
                "expected {expected} bytes of parameters, got {}",
                params.len()
            )));
        }
        let bits = params[1..=d].to_vec();
        let used: usize = bits.iter().map(|&b| usize::from(b)).sum();
        if bits.iter().any(|&b| b > MAX_BITS) || used > self.bytes * 8 {
            return Err(bad(format!("bit allocation uses {used} bits of {}", self.bytes * 8)));
        }
        let f: Vec<f32> = params[1 + d..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        self.fitted = Some(Fitted {
            mean: f[..d].to_vec(),
            step: f[d..2 * d].to_vec(),
            rotation: self.pca.then(|| f[2 * d..].to_vec()),
            bits,
        });
        Ok(())
    }

    fn encode(&self, vector: &[f32]) -> Result<Encoded, CodecError> {
        check_dim(self.dimension, vector.len())?;
        let f = self.state()?;
        let y = Self::rotate(f, vector);
        let mut w = BitWriter::new(self.bytes);
        for (i, &b) in f.bits.iter().enumerate() {
            if b == 0 {
                continue;
            }
            let levels = (1u32 << b) - 1;
            let half = f64::from(levels) / 2.0;
            let step = f64::from(f.step[i]);
            let raw = if step > 0.0 {
                ((f64::from(y[i]) - f64::from(f.mean[i])) / step + half).round()
            } else {
                half.round()
            };
            w.write(raw.clamp(0.0, f64::from(levels)) as u32, b);
        }
        Ok(Encoded(w.finish(self.bytes)))
    }

    fn decode(&self, encoded: &Encoded) -> Result<Vector, CodecError> {
        check_payload(self.bytes, encoded.0.len())?;
        let f = self.state()?;
        let mut y = vec![0f32; self.dimension];
        Self::for_each_value(f, encoded, |i, v| y[i] = v as f32);
        Ok(match &f.rotation {
            None => y,
            Some(r) => {
                // x = Rᵀ y
                let d = self.dimension;
                let mut x = vec![0f64; d];
                for (j, &yj) in y.iter().enumerate() {
                    for (xk, &rjk) in x.iter_mut().zip(&r[j * d..(j + 1) * d]) {
                        *xk += f64::from(rjk) * f64::from(yj);
                    }
                }
                x.into_iter().map(|v| v as f32).collect()
            }
        })
    }

    fn score(&self, query: &[f32], encoded: &Encoded, metric: Metric) -> Result<f32, CodecError> {
        let prepared = self.prepare_query(query)?;
        self.score_prepared(&prepared, encoded, metric)
    }

    fn prepare_query(&self, query: &[f32]) -> Result<Vector, CodecError> {
        check_dim(self.dimension, query.len())?;
        Ok(Self::rotate(self.state()?, query))
    }

    fn score_prepared(&self, prepared: &[f32], encoded: &Encoded, metric: Metric) -> Result<f32, CodecError> {
        check_dim(self.dimension, prepared.len())?;
        check_payload(self.bytes, encoded.0.len())?;
        let f = self.state()?;
        let (mut dot, mut stored_sq, mut query_sq) = (0f64, 0f64, 0f64);
        Self::for_each_value(f, encoded, |i, v| {
            let q = f64::from(prepared[i]);
            dot += q * v;
            stored_sq += v * v;
            query_sq += q * q;
        });
        let v = match metric {
            Metric::Dot => dot,
            Metric::Cosine => {
                let denom = (query_sq * stored_sq).sqrt();
                if denom == 0.0 { 0.0 } else { dot / denom }
            }
            Metric::L2 => (query_sq - 2.0 * dot + stored_sq).max(0.0),
        };
        Ok(v as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance;

    struct XorShift(u64);
    impl XorShift {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// Correlated data: a few strong directions mixed into every coordinate,
    /// so raw dimensions look alike but principal components do not.
    fn correlated(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = XorShift(99);
        let dirs: Vec<Vec<f32>> = (0..4).map(|_| (0..dim).map(|_| rng.next_f32()).collect()).collect();
        (0..n)
            .map(|_| {
                let w: Vec<f32> = (0..4).map(|k| rng.next_f32() * (4 - k) as f32).collect();
                let mut v: Vec<f32> = (0..dim)
                    .map(|i| (0..4).map(|k| w[k] * dirs[k][i]).sum::<f32>() + 0.05 * rng.next_f32())
                    .collect();
                distance::normalize(&mut v);
                v
            })
            .collect()
    }

    fn fitted(dim: usize, bytes: usize, pca: bool, data: &[Vec<f32>]) -> MixedCodec {
        let mut c = MixedCodec::new(dim, bytes, pca, false).unwrap();
        assert!(c.needs_fit());
        c.fit(data).unwrap();
        c
    }

    #[test]
    fn allocation_spends_the_budget_where_the_variance_is() {
        let bits = allocate(&[16.0, 4.0, 1.0, 0.0], 6);
        assert_eq!(bits.iter().map(|&b| usize::from(b)).sum::<usize>(), 6);
        assert!(bits[0] >= bits[1] && bits[1] >= bits[2] && bits[3] == 0, "{bits:?}");
        assert_eq!(allocate(&[1.0, 1.0], 100), vec![8, 8]);
    }

    #[test]
    fn weighting_moves_bits_towards_high_energy_dimensions() {
        let data = correlated(300, 16);
        let mut plain = MixedCodec::new(16, 4, true, false).unwrap();
        let mut weighted = MixedCodec::new(16, 4, true, true).unwrap();
        plain.fit(&data).unwrap();
        weighted.fit(&data).unwrap();
        assert_eq!(weighted.name(), "mixed4-pca-w");
        let (p, w) = (plain.bits().unwrap(), weighted.bits().unwrap());
        assert_eq!(p.iter().map(|&b| u32::from(b)).sum::<u32>(), 32);
        assert_eq!(w.iter().map(|&b| u32::from(b)).sum::<u32>(), 32);
        assert!(w[0] >= p[0], "weighted {w:?} plain {p:?}");
    }

    #[test]
    fn jacobi_recovers_a_known_spectrum() {
        // [[2,1],[1,2]] has eigenvalues 3 and 1.
        let (vals, rows) = symmetric_eigen(vec![2.0, 1.0, 1.0, 2.0], 2);
        assert!((vals[0] - 3.0).abs() < 1e-9 && (vals[1] - 1.0).abs() < 1e-9, "{vals:?}");
        let r = std::f64::consts::FRAC_1_SQRT_2;
        assert!((rows[0].abs() - r).abs() < 1e-9 && (rows[1].abs() - r).abs() < 1e-9);
    }

    #[test]
    fn score_is_consistent_with_own_decode() {
        let data = correlated(300, 32);
        for pca in [false, true] {
            let c = fitted(32, 12, pca, &data);
            for (q, v) in data.iter().zip(data.iter().skip(1)).take(20) {
                let e = c.encode(v).unwrap();
                assert_eq!(e.0.len(), 12);
                let d = c.decode(&e).unwrap();
                for m in [Metric::Dot, Metric::Cosine, Metric::L2] {
                    let a = c.score(q, &e, m).unwrap();
                    let b = distance::score(q, &d, m);
                    assert!((a - b).abs() < 1e-4, "{m:?} pca={pca}: {a} vs {b}");
                }
                let p = c.prepare_query(q).unwrap();
                assert_eq!(
                    c.score_prepared(&p, &e, Metric::Cosine).unwrap(),
                    c.score(q, &e, Metric::Cosine).unwrap()
                );
            }
        }
    }

    #[test]
    fn pca_beats_raw_allocation_on_correlated_data_and_more_bytes_help() {
        let data = correlated(400, 32);
        let err = |c: &MixedCodec| -> f64 {
            data.iter()
                .zip(data.iter().skip(7))
                .take(100)
                .map(|(q, v)| {
                    let e = c.encode(v).unwrap();
                    f64::from((c.score(q, &e, Metric::Cosine).unwrap() - distance::cosine(q, v)).abs())
                })
                .sum::<f64>()
                / 100.0
        };
        // Budgets small enough that the 8-bit cap per dimension does not bind.
        let raw = err(&fitted(32, 4, false, &data));
        let pca = err(&fitted(32, 4, true, &data));
        let pca_small = err(&fitted(32, 2, true, &data));
        assert!(pca < raw, "pca {pca} raw {raw}");
        assert!(pca < pca_small, "4 bytes {pca} vs 2 bytes {pca_small}");
        // PCA puts the bits on the leading components.
        let c = fitted(32, 8, true, &data);
        let bits = c.bits().unwrap();
        assert!(bits[0] >= bits[31], "{bits:?}");
    }

    #[test]
    fn params_roundtrip_and_are_validated() {
        let data = correlated(200, 16);
        for pca in [false, true] {
            let c = fitted(16, 6, pca, &data);
            let mut r = MixedCodec::new(16, 6, pca, false).unwrap();
            r.restore(&c.params()).unwrap();
            let v = &data[3];
            assert_eq!(r.encode(v).unwrap(), c.encode(v).unwrap());
            assert!(
                MixedCodec::new(16, 6, !pca, false)
                    .unwrap()
                    .restore(&c.params())
                    .is_err()
            );
        }
        assert!(MixedCodec::new(16, 0, false, false).is_err());
        assert!(MixedCodec::new(16, 17, false, false).is_err());
        assert!(MixedCodec::new(16, 6, false, false).unwrap().encode(&data[0]).is_err());
    }
}
