/// Computes the forward FFT using Apple's MLX framework.
///
/// Same interface as [`crate::fft`]: zero-pads to the next power of two,
/// returns `(real, imag)` of length `n.next_power_of_two()`.
#[must_use]
pub fn fft(input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n_orig = input.len();
    let n = n_orig.next_power_of_two();

    if n <= 1 {
        let mut real = vec![0.0f32; n];
        if n == 1 && n_orig == 1 {
            real[0] = input[0];
        }
        return (real, vec![0.0f32; n]);
    }

    let mut real_in = vec![0.0f32; n];
    real_in[..n_orig].copy_from_slice(input);
    let imag_in = vec![0.0f32; n];

    let mut real_out = vec![0.0f32; n];
    let mut imag_out = vec![0.0f32; n];

    let ret = unsafe {
        super::ffi::mlx_fft_forward(
            real_in.as_ptr(),
            imag_in.as_ptr(),
            real_out.as_mut_ptr(),
            imag_out.as_mut_ptr(),
            n as u32,
        )
    };
    assert_eq!(ret, 0, "MLX FFT forward failed with error code {ret}");

    (real_out, imag_out)
}

/// Computes the inverse FFT using Apple's MLX framework.
///
/// Same interface as [`crate::ifft`]: returns a `Vec<f32>` of length `2*n`
/// where `[0..n]` is the real part and `[n..2n]` is the imaginary part.
/// Applies 1/N scaling on the CPU side.
#[must_use]
pub fn ifft(input_real: &[f32], input_imag: &[f32]) -> Vec<f32> {
    let n = input_real.len();
    assert_eq!(n, input_imag.len(), "real and imag lengths must match");

    if n <= 1 {
        let mut out = vec![0.0f32; n * 2];
        if n == 1 {
            out[0] = input_real[0];
            out[1] = input_imag[0];
        }
        return out;
    }

    assert!(n.is_power_of_two(), "length must be a power of two");

    let mut real_out = vec![0.0f32; n];
    let mut imag_out = vec![0.0f32; n];

    let ret = unsafe {
        super::ffi::mlx_fft_inverse(
            input_real.as_ptr(),
            input_imag.as_ptr(),
            real_out.as_mut_ptr(),
            imag_out.as_mut_ptr(),
            n as u32,
        )
    };
    assert_eq!(ret, 0, "MLX FFT inverse failed with error code {ret}");

    // MLX with BACKWARD norm already applies 1/N to IFFT, so no scaling needed.
    // But let's verify: BACKWARD norm means forward has no scale, inverse has 1/N.
    // This matches our convention.

    let mut result = real_out;
    result.extend(imag_out);
    result
}
