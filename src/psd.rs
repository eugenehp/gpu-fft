/// Computes the Power Spectral Density (PSD) from the real and imaginary components of a signal.
///
/// The Power Spectral Density is a measure of the power of a signal per unit frequency. This function
/// calculates the PSD by first computing the magnitude of the complex numbers represented by the
/// real and imaginary components, and then calculating the power from the magnitude. The result is
/// normalized by the number of points in the input vectors.
///
/// # Parameters
///
/// - `real`: A vector of `f32` values representing the real parts of the complex signal.
/// - `imag`: A vector of `f32` values representing the imaginary parts of the complex signal.
///
/// # Returns
///
/// A vector of `f32` values representing the Power Spectral Density of the input signal. The length
/// of the output vector will be the same as the input vectors.
///
/// # Example
///
/// ```
/// # use gpu_fft::psd::psd;
/// let real = vec![1.0f32, 0.0, 0.0, 0.0];
/// let imag = vec![0.0f32, 0.0, 0.0, 0.0];
/// let psd_values = psd(real, imag);
/// assert_eq!(psd_values.len(), 4);
/// ```
///
/// # Note
///
/// The normalization step (dividing by `n`) is optional and can be adjusted based on specific
/// requirements or conventions in your application.
pub fn psd(real: Vec<f32>, imag: Vec<f32>) -> Vec<f32> {
    let n = real.len();
    let mut psd = Vec::with_capacity(n);

    for i in 0..n {
        // Calculate the magnitude
        let magnitude = (real[i] * real[i] + imag[i] * imag[i]).sqrt();
        // Calculate the power
        let power = magnitude * magnitude;
        // Normalize the power (optional, depending on your needs)
        psd.push(power / n as f32); // Normalization by the number of points
    }

    psd
}
