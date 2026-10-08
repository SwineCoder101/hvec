//! Retrieval metrics computed against a baseline ranking.

use std::collections::HashSet;

/// Fraction of the baseline's top-k that appears in the candidate's top-k.
///
/// Both slices are chunk ids in rank order. Returns 1.0 when the baseline is
/// empty, since there is nothing to miss.
#[must_use]
pub fn recall_at_k(baseline: &[i64], candidate: &[i64], k: usize) -> f64 {
    let truth: HashSet<i64> = baseline.iter().take(k).copied().collect();
    if truth.is_empty() {
        return 1.0;
    }
    let found = candidate.iter().take(k).filter(|id| truth.contains(id)).count();
    found as f64 / truth.len() as f64
}

/// Mean absolute difference between paired scores.
#[must_use]
pub fn mean_abs_error(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return f64::NAN;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .sum::<f64>()
        / a.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_counts_overlap_regardless_of_order() {
        assert_eq!(recall_at_k(&[1, 2, 3], &[3, 1, 9], 3), 2.0 / 3.0);
        assert_eq!(recall_at_k(&[1, 2, 3], &[1, 2, 3], 3), 1.0);
        assert_eq!(recall_at_k(&[], &[1], 3), 1.0);
    }

    #[test]
    fn mae_of_identical_is_zero() {
        assert_eq!(mean_abs_error(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        assert!(mean_abs_error(&[1.0], &[]).is_nan());
    }
}
