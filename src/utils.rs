/// Generates a sine wave signal based on the specified frequency, sample rate, and duration.
///
/// This function creates a vector of samples representing a sine wave. The sine wave is generated
/// using the formula: `sin(2 * π * frequency * time)`, where `time` is calculated based on the
/// sample rate and the sample index.
///
/// # Parameters
///
/// - `frequency`: The frequency of the sine wave in Hertz (Hz).
/// - `sample_rate`: The number of samples per second (samples/second).
/// - `duration`: The duration of the sine wave in seconds.
///
/// # Returns
///
/// A vector of `f32` values representing the samples of the generated sine wave. The length of the
/// output vector will be equal to the number of samples calculated as `sample_rate * duration`.
///
/// # Example
///
/// ```rust
/// let frequency = 440.0; // A4 note
/// let sample_rate = 44100.0; // CD quality
/// let duration = 1.0; // 1 second
/// let sine_wave = generate_sine_wave(frequency, sample_rate, duration);
/// ```
pub fn generate_sine_wave(frequency: f32, sample_rate: f32, duration: f32) -> Vec<f32> {
    let num_samples = (sample_rate * duration) as usize;
    let mut sine_wave = Vec::with_capacity(num_samples);

    for n in 0..num_samples {
        let sample = (2.0 * PI * frequency * (n as f32 / sample_rate)).sin();
        sine_wave.push(sample);
    }

    sine_wave
}

/// Calculates the frequency values corresponding to the given number of samples and sample rate.
///
/// This function generates a vector of frequency values based on the number of samples and the
/// sample rate. The frequencies are calculated as `k * sample_rate / n`, where `k` is the index
/// of the frequency bin.
///
/// # Parameters
///
/// - `n`: The number of frequency bins (samples).
/// - `sample_rate`: The sample rate in Hertz (Hz).
///
/// # Returns
///
/// A vector of `f32` values representing the frequency values for each bin. The length of the
/// output vector will be equal to `n`.
///
/// # Example
///
/// ```rust
/// let n = 1024; // Number of frequency bins
/// let sample_rate = 44100.0; // Sample rate in Hz
/// let frequencies = calculate_frequencies(n, sample_rate);
/// ```
pub fn calculate_frequencies(n: usize, sample_rate: f32) -> Vec<f32> {
    (0..n).map(|k| k as f32 * sample_rate / n as f32).collect()
}

/// Finds the dominant frequencies in the Power Spectral Density (PSD) based on a threshold.
///
/// This function identifies the dominant frequencies in the provided PSD by checking for peaks
/// in the PSD values that exceed a specified threshold. A peak is defined as a point that is
/// greater than its immediate neighbors.
///
/// # Parameters
///
/// - `psd`: A vector of `f32` values representing the Power Spectral Density of the signal.
/// - `frequencies`: A vector of `f32` values representing the frequency values corresponding to
///   the PSD.
/// - `threshold`: A threshold value for identifying dominant frequencies. Only peaks above this
///   threshold will be considered.
///
/// # Returns
///
/// A vector of tuples, where each tuple contains a dominant frequency and its corresponding PSD
/// value. The first element of the tuple is the frequency, and the second element is the PSD value.
///
/// # Example
///
/// ```rust
/// let psd = vec![0.1, 0.5, 0.3, 0.7, 0.2]; // Example PSD values
/// let frequencies = vec![0.0, 100.0, 200.0, 300.0, 400.0]; // Corresponding frequencies
/// let threshold = 0.4; // Threshold for dominance
/// let dominant_freqs = find_dominant_frequencies(psd, frequencies, threshold);
/// ```
pub fn find_dominant_frequencies(
    psd: Vec<f32>,
    frequencies: Vec<f32>,
    threshold: f32,
) -> Vec<(f32, f32)> {
    let mut dominant_frequencies = Vec::new();

    for i in 1..(psd.len() - 1) {
        // Check if the current point is a peak
        if psd[i] > psd[i - 1] && psd[i] > psd[i + 1] && psd[i] > threshold {
            dominant_frequencies.push((frequencies[i], psd[i]));
        }
    }

    dominant_frequencies
}
