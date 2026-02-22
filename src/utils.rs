use std::f32::consts::PI;

/// Generates a sine wave signal based on the specified frequency, sample rate, and duration.
///
/// # Parameters
///
/// - `frequency`: The frequency of the sine wave in Hertz (Hz).
/// - `sample_rate`: The number of samples per second.
/// - `duration`: The duration of the sine wave in seconds.
///
/// # Returns
///
/// A vector of `f32` samples. Its length equals `(sample_rate * duration) as usize`.
///
/// # Example
///
/// ```
/// # use gpu_fft::utils::generate_sine_wave;
/// let frequency = 440.0f32;    // A4 note
/// let sample_rate = 44100.0f32; // CD quality
/// let duration = 1.0f32;        // 1 second
/// let sine_wave = generate_sine_wave(frequency, sample_rate, duration);
/// assert_eq!(sine_wave.len(), 44100);
/// ```
#[must_use]
pub fn generate_sine_wave(frequency: f32, sample_rate: f32, duration: f32) -> Vec<f32> {
    let num_samples = (sample_rate * duration) as usize;
    (0..num_samples)
        .map(|n| (2.0 * PI * frequency * n as f32 / sample_rate).sin())
        .collect()
}

/// Returns the frequency (in Hz) corresponding to each bin of a full (two-sided) DFT output.
///
/// Bin `k` maps to `k * sample_rate / n` Hz.  The upper half of the returned frequencies
/// (`> sample_rate / 2`) represent negative frequencies — they are the conjugate mirrors
/// of the lower half and carry no additional information for real-valued signals.
///
/// For real signals, prefer [`calculate_one_sided_frequencies`] instead.
///
/// # Example
///
/// ```
/// # use gpu_fft::utils::calculate_frequencies;
/// let frequencies = calculate_frequencies(1024, 44100.0);
/// assert_eq!(frequencies.len(), 1024);
/// assert_eq!(frequencies[0], 0.0);
/// ```
#[must_use]
pub fn calculate_frequencies(n: usize, sample_rate: f32) -> Vec<f32> {
    (0..n).map(|k| k as f32 * sample_rate / n as f32).collect()
}

/// Returns the `n_total / 2 + 1` unique positive-frequency bins (0 Hz … Nyquist) for a
/// real-valued DFT of `n_total` samples at `sample_rate` Hz.
///
/// For a real input the DFT spectrum is conjugate-symmetric, so only the first half plus
/// the DC and Nyquist bins are unique.  Use this together with slicing the PSD to the
/// same length to avoid spurious mirror-image peaks.
///
/// # Example
///
/// ```
/// # use gpu_fft::utils::calculate_one_sided_frequencies;
/// let freqs = calculate_one_sided_frequencies(1000, 200.0);
/// assert_eq!(freqs.len(), 501);       // n/2 + 1
/// assert_eq!(freqs[0], 0.0);          // DC
/// assert!((freqs[500] - 100.0).abs() < 1e-4); // Nyquist = sample_rate / 2
/// ```
#[must_use]
pub fn calculate_one_sided_frequencies(n_total: usize, sample_rate: f32) -> Vec<f32> {
    (0..=n_total / 2)
        .map(|k| k as f32 * sample_rate / n_total as f32)
        .collect()
}

/// Finds the dominant frequencies in a Power Spectral Density (PSD) by looking for local
/// peaks above a threshold.
///
/// A peak is a bin whose value exceeds both immediate neighbours and the threshold.
/// The first and last bins are never reported (they cannot be local peaks).
///
/// For real-valued signals, pass only the **one-sided** PSD (first `n/2 + 1` bins) and the
/// matching frequencies from [`calculate_one_sided_frequencies`] to avoid spurious
/// mirror-image peaks in the upper half of the spectrum.
///
/// # Example
///
/// ```
/// # use gpu_fft::utils::find_dominant_frequencies;
/// let psd = vec![0.1f32, 0.5, 0.3, 0.7, 0.2];
/// let frequencies = vec![0.0f32, 100.0, 200.0, 300.0, 400.0];
/// let dominant = find_dominant_frequencies(&psd, &frequencies, 0.4);
/// // Bins 1 (100 Hz) and 3 (300 Hz) are local peaks above the threshold.
/// assert_eq!(dominant.len(), 2);
/// assert_eq!(dominant[0].0, 100.0);
/// assert_eq!(dominant[1].0, 300.0);
/// ```
#[must_use]
pub fn find_dominant_frequencies(
    psd: &[f32],
    frequencies: &[f32],
    threshold: f32,
) -> Vec<(f32, f32)> {
    assert_eq!(psd.len(), frequencies.len(), "psd and frequencies must have the same length");
    (1..psd.len().saturating_sub(1))
        .filter(|&i| psd[i] > psd[i - 1] && psd[i] > psd[i + 1] && psd[i] > threshold)
        .map(|i| (frequencies[i], psd[i]))
        .collect()
}
