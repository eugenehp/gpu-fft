/// Computes the Power Spectral Density (PSD) from the real and imaginary components of a DFT.
///
/// Each bin's power is `(real² + imag²) / n`, where `n` is the number of bins.
///
/// For a real-valued input signal the DFT is conjugate-symmetric, so the upper half of the
/// spectrum is a mirror of the lower half. Pass only the first `n/2 + 1` bins to obtain the
/// **one-sided** PSD (recommended); pass all `n` bins for the full two-sided PSD.
///
/// # Parameters
///
/// - `real`: Real parts of the DFT output.
/// - `imag`: Imaginary parts of the DFT output.
///
/// # Returns
///
/// A vector of PSD values with the same length as the inputs.
///
/// # Example
///
/// ```
/// # use gpu_fft::psd::psd;
/// let real = vec![1.0f32, 0.0, 0.0, 0.0];
/// let imag = vec![0.0f32, 0.0, 0.0, 0.0];
/// let psd_values = psd(&real, &imag);
/// assert_eq!(psd_values.len(), 4);
/// ```
#[must_use]
pub fn psd(real: &[f32], imag: &[f32]) -> Vec<f32> {
    assert_eq!(real.len(), imag.len(), "real and imag must have the same length");
    let n = real.len() as f32;
    real.iter()
        .zip(imag)
        // power = real² + imag²  (sqrt then square is a no-op, removed)
        .map(|(&r, &im)| (r * r + im * im) / n)
        .collect()
}
