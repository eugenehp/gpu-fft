//! Shared test helpers compiled into every integration-test binary.
//! Items may not all be used in every binary, so dead_code is suppressed.
#![allow(dead_code)]

/// Maximum absolute error tolerated between GPU f32 results and analytical values.
pub const EPSILON: f32 = 1e-3;

pub fn assert_approx(actual: f32, expected: f32, label: &str) {
    assert!(
        (actual - expected).abs() <= EPSILON,
        "{}: got {:.6}, expected {:.6}  (diff {:.2e})",
        label,
        actual,
        expected,
        (actual - expected).abs(),
    );
}

pub fn assert_slice_approx(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch ({} vs {})",
        label,
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= EPSILON,
            "{}[{}]: got {:.6}, expected {:.6}  (diff {:.2e})",
            label,
            i,
            a,
            e,
            (a - e).abs(),
        );
    }
}
