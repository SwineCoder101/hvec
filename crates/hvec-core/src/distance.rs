//! Scalar reference kernels. Every codec is tested against these.

use crate::Metric;

/// Inner product of two equal-length slices.
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Euclidean norm.
#[must_use]
pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// Cosine similarity. Returns 0 when either vector has zero norm.
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let denom = norm(a) * norm(b);
    if denom == 0.0 { 0.0 } else { dot(a, b) / denom }
}

/// Squared Euclidean distance.
#[must_use]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Dispatch on a metric.
#[must_use]
pub fn score(a: &[f32], b: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::Dot => dot(a, b),
        Metric::Cosine => cosine(a, b),
        Metric::L2 => l2_squared(a, b),
    }
}

/// Normalise a vector in place to unit length. No-op for the zero vector.
pub fn normalize(a: &mut [f32]) {
    let n = norm(a);
    if n > 0.0 {
        a.iter_mut().for_each(|x| *x /= n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_and_cosine_agree_on_unit_vectors() {
        let mut a = vec![3.0, 4.0];
        let mut b = vec![4.0, 3.0];
        normalize(&mut a);
        normalize(&mut b);
        assert!((dot(&a, &b) - cosine(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn l2_of_identical_vectors_is_zero() {
        let a = [1.0, 2.0, 3.0];
        assert_eq!(l2_squared(&a, &a), 0.0);
    }

    #[test]
    fn cosine_handles_zero_vector() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }
}
