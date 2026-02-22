use std::f32::consts::PI;

use gpu_fft::fft;

mod common;
use common::{assert_approx, assert_slice_approx};

/// FFT of an impulse [1, 0, 0, …] must be all-ones real and all-zeros imaginary.
///
/// Proof: X[k] = Σ x[n]·e^{-j2πkn/N} = x[0]·e^0 = 1  for all k.
#[test]
fn test_fft_impulse() {
    let n = 8;
    let mut input = vec![0.0f32; n];
    input[0] = 1.0;

    let (real, imag) = fft(&input);

    assert_slice_approx(&real, &[1.0; 8], "real");
    assert_slice_approx(&imag, &[0.0; 8], "imag");
}

/// FFT of a constant (DC) signal [1, 1, …, 1] must be [N, 0, 0, …] real, all-zeros imaginary.
///
/// Proof: X[0] = N·1 = N;  X[k≠0] = Σ e^{-j2πkn/N} = 0 (geometric sum).
#[test]
fn test_fft_dc_signal() {
    let n = 8;
    let input = vec![1.0f32; n];

    let (real, imag) = fft(&input);

    assert_approx(real[0], n as f32, "real[0]");
    assert_approx(imag[0], 0.0, "imag[0]");
    for i in 1..n {
        assert_approx(real[i], 0.0, &format!("real[{}]", i));
        assert_approx(imag[i], 0.0, &format!("imag[{}]", i));
    }
}

/// FFT of x[n] = sin(2π·n/N), N=8, must have power only at bins 1 and N-1.
///
/// Proof via DFT of complex exponentials:
///   sin(2πn/N) = (e^{j2πn/N} - e^{-j2πn/N}) / 2j
///   X[1]   = -jN/2  →  real=0,     imag=-N/2
///   X[N-1] = +jN/2  →  real=0,     imag=+N/2
///   X[k]   = 0  for all other k
#[test]
fn test_fft_single_frequency_sine() {
    let n = 8usize;
    let half_n = n as f32 / 2.0; // 4.0

    let input: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * i as f32 / n as f32).sin())
        .collect();

    let (real, imag) = fft(&input);

    // DC bin
    assert_approx(real[0], 0.0, "real[0] (DC)");
    assert_approx(imag[0], 0.0, "imag[0] (DC)");

    // Positive-frequency peak at bin 1: X[1] = -jN/2
    assert_approx(real[1], 0.0, "real[1]");
    assert_approx(imag[1], -half_n, "imag[1]");

    // All middle bins should be zero
    for k in 2..n - 1 {
        assert_approx(real[k], 0.0, &format!("real[{}]", k));
        assert_approx(imag[k], 0.0, &format!("imag[{}]", k));
    }

    // Negative-frequency mirror at bin N-1: X[N-1] = +jN/2
    assert_approx(real[n - 1], 0.0, &format!("real[{}]", n - 1));
    assert_approx(imag[n - 1], half_n, &format!("imag[{}]", n - 1));
}

/// FFT of the all-zeros vector must be the all-zeros vector.
#[test]
fn test_fft_zero_input() {
    let n = 8;
    let (real, imag) = fft(&vec![0.0f32; n]);
    assert_slice_approx(&real, &[0.0; 8], "real");
    assert_slice_approx(&imag, &[0.0; 8], "imag");
}

/// Linearity: FFT(a·x) = a·FFT(x).
#[test]
fn test_fft_linearity() {
    let scale = 3.0f32;
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let scaled_input: Vec<f32> = input.iter().map(|&x| x * scale).collect();

    let (real_base, imag_base) = fft(&input);
    let (real_scaled, imag_scaled) = fft(&scaled_input);

    let n = real_base.len();
    for i in 0..n {
        assert_approx(
            real_scaled[i],
            real_base[i] * scale,
            &format!("real[{}]", i),
        );
        assert_approx(
            imag_scaled[i],
            imag_base[i] * scale,
            &format!("imag[{}]", i),
        );
    }
}
