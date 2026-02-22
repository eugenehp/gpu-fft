use gpu_fft::ifft;

mod common;
use common::assert_slice_approx;

/// IFFT of a DC spectrum [N, 0, 0, …] must give the constant signal [1, 1, …, 1].
///
/// This is the inverse of `test_fft_dc_signal` in fft.rs.
#[test]
fn test_ifft_dc_spectrum() {
    let n = 8;
    let mut real_in = vec![0.0f32; n];
    real_in[0] = n as f32;
    let imag_in = vec![0.0f32; n];

    let output = ifft(real_in, imag_in);

    // output[0..n]  = real part of reconstructed signal
    // output[n..2n] = imaginary part (should be ~0 for a real spectrum)
    assert_slice_approx(&output[..n], &[1.0f32; 8], "real (constant signal)");
    assert_slice_approx(&output[n..], &[0.0f32; 8], "imag (should be ~0)");
}

/// IFFT of a flat spectrum [1, 1, …, 1] must give an impulse [1, 0, 0, …] in time.
///
/// This is the inverse of `test_fft_impulse` in fft.rs.
#[test]
fn test_ifft_flat_spectrum() {
    let n = 8;
    let real_in = vec![1.0f32; n];
    let imag_in = vec![0.0f32; n];

    let output = ifft(real_in, imag_in);

    let mut expected_real = vec![0.0f32; n];
    expected_real[0] = 1.0;

    assert_slice_approx(&output[..n], &expected_real, "real (impulse)");
    assert_slice_approx(&output[n..], &[0.0f32; 8], "imag (should be ~0)");
}

/// Linearity of IFFT: IFFT(a·X) = a·IFFT(X).
#[test]
fn test_ifft_linearity() {
    let scale = 3.0f32;
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let n = input.len();

    let (real, imag) = gpu_fft::fft(input);
    let output_base = ifft(real.clone(), imag.clone());

    let real_scaled: Vec<f32> = real.iter().map(|&x| x * scale).collect();
    let imag_scaled: Vec<f32> = imag.iter().map(|&x| x * scale).collect();
    let output_scaled = ifft(real_scaled, imag_scaled);

    for i in 0..n {
        common::assert_approx(
            output_scaled[i],
            output_base[i] * scale,
            &format!("scaled real[{}]", i),
        );
    }
}
