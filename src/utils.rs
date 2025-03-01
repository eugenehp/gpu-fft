use std::f32::consts::PI;

pub fn generate_sine_wave(frequency: f32, sample_rate: f32, duration: f32) -> Vec<f32> {
    let num_samples = (sample_rate * duration) as usize;
    let mut sine_wave = Vec::with_capacity(num_samples);

    for n in 0..num_samples {
        let sample = (2.0 * PI * frequency * (n as f32 / sample_rate)).sin();
        sine_wave.push(sample);
    }

    sine_wave
}

pub fn calculate_frequencies(n: usize, sample_rate: f32) -> Vec<f32> {
    (0..n).map(|k| k as f32 * sample_rate / n as f32).collect()
}

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
