use std::f32::consts::PI;

use gpu_fft::{fft, ifft};

mod common;
use common::assert_slice_approx;

/// IFFT(FFT(x))[0..n] must recover x; the imaginary half must be ~0 (real input).
#[test]
fn test_roundtrip_arbitrary_signal() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let n = input.len();

    let (real, imag) = fft(&input);
    let output = ifft(&real, &imag);

    assert_slice_approx(&output[..n], &input, "real (round-trip)");
    assert_slice_approx(&output[n..], &[0.0f32; 8], "imag (should be ~0)");
}

/// Round-trip must also work for a signal that contains negative values.
#[test]
fn test_roundtrip_with_negatives() {
    let input = vec![-3.0f32, 1.5, 0.0, -2.0, 4.0, -1.0, 0.5, 2.5];
    let n = input.len();

    let (real, imag) = fft(&input);
    let output = ifft(&real, &imag);

    assert_slice_approx(&output[..n], &input, "real (round-trip with negatives)");
    assert_slice_approx(&output[n..], &[0.0f32; 8], "imag (should be ~0)");
}

/// Round-trip must also work for a pure sine wave (not just arbitrary samples).
#[test]
fn test_roundtrip_sine_wave() {
    let n = 8usize;
    let input: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * i as f32 / n as f32).sin())
        .collect();

    let (real, imag) = fft(&input);
    let output = ifft(&real, &imag);

    assert_slice_approx(&output[..n], &input, "real (sine round-trip)");
    assert_slice_approx(&output[n..], &[0.0f32; 8], "imag (should be ~0)");
}

// ── Radix-4 outer stage coverage ─────────────────────────────────────────────

/// Round-trip for N = 4096 (1 radix-4 outer stage).
#[test]
fn test_roundtrip_large_even() {
    let n = 4096usize;
    let input: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * 7.0 * i as f32 / n as f32).sin())
        .collect();

    let (real, imag) = fft(&input);
    let output = ifft(&real, &imag);

    // Tolerance scales with log₂(N): 5 · log₂(4096) · ε_machine ≈ 5·12·1.2e-7 ≈ 7e-6
    let tol = 5.0 * (n as f32).log2() * f32::EPSILON;
    for i in 0..n {
        assert!(
            (output[i] - input[i]).abs() <= tol,
            "real[{i}]: got {:.6}, expected {:.6} (diff {:.2e}, tol {:.2e})",
            output[i], input[i], (output[i] - input[i]).abs(), tol
        );
    }
}

/// Round-trip for N = 8192 (1 radix-4 + trailing radix-2 outer stage).
#[test]
fn test_roundtrip_large_odd() {
    let n = 8192usize;
    let input: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * 13.0 * i as f32 / n as f32).sin())
        .collect();

    let (real, imag) = fft(&input);
    let output = ifft(&real, &imag);

    let tol = 5.0 * (n as f32).log2() * f32::EPSILON;
    for i in 0..n {
        assert!(
            (output[i] - input[i]).abs() <= tol,
            "real[{i}]: got {:.6}, expected {:.6} (diff {:.2e}, tol {:.2e})",
            output[i], input[i], (output[i] - input[i]).abs(), tol
        );
    }
}
