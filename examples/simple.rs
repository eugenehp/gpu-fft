use std::time::Instant;

use gpu_fft::{psd, utils};

fn main() {
    let sample_rate = 200.0f32; // Hz
    let frequency = 15.0f32;   // Hz
    let duration = 5.0f32;     // seconds  →  1 000 samples
    let threshold = 100.0f32;  // minimum power to report a dominant frequency

    // ── Generate input signal ─────────────────────────────────────────────────
    let input = utils::generate_sine_wave(frequency, sample_rate, duration);

    println!("Input");
    println!("  samples    : {}", input.len());
    println!("  sample rate: {sample_rate} Hz");
    println!("  frequency  : {frequency} Hz");
    println!("  first 5    : {:.4?}", &input[..5]);
    println!();

    // ── FFT ───────────────────────────────────────────────────────────────────
    let t = Instant::now();
    let (real, imag) = gpu_fft::fft(&input);
    println!("FFT  completed in {:?}", t.elapsed());

    // ── Spectral analysis (one-sided) ─────────────────────────────────────────
    // PSD is computed over all N bins so the normalization denominator is N.
    // For a real-valued signal X[N-k] = X[k]*, so only the first N/2+1 bins
    // (0 Hz … Nyquist) are unique — scanning the full spectrum would report
    // every peak twice.  We slice the PSD before the peak-search step.
    let n = real.len();
    let n_unique = n / 2 + 1;

    let spectrum = psd::psd(&real, &imag);          // length N, normalised by N
    let frequencies = utils::calculate_one_sided_frequencies(n, sample_rate); // length N/2+1
    let dominant = utils::find_dominant_frequencies(&spectrum[..n_unique], &frequencies, threshold);

    println!("Dominant frequencies (0 … {:.0} Hz):", sample_rate / 2.0);
    if dominant.is_empty() {
        println!("  (none above threshold {threshold})");
    } else {
        for (freq, power) in &dominant {
            println!("  {:>8.2} Hz  power {:>12.2}", freq, power);
        }
    }
    println!();

    // ── IFFT ──────────────────────────────────────────────────────────────────
    let t = Instant::now();
    let output = gpu_fft::ifft(&real, &imag);
    println!("IFFT completed in {:?}", t.elapsed());

    // output[0..n]  = reconstructed real signal
    // output[n..2n] = reconstructed imaginary signal (≈ 0 for real inputs)
    let reconstructed = &output[..n];

    // Round-trip error: expected ~2·N·ε for f32, where ε ≈ 1.2e-7
    let max_err = input
        .iter()
        .zip(reconstructed)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // For an O(N²) DFT each output bin accumulates N f32 multiply-adds.
    // The per-element error is O(N·ε); the max over N elements is typically
    // 3–5× that, so anything below ~5·N·ε is expected.
    let acceptable = 5.0 * n as f32 * f32::EPSILON;
    let verdict = if max_err <= acceptable { "✓" } else { "✗ unexpectedly large" };
    println!("Round-trip max error : {max_err:.2e}  (limit 5·N·ε = {acceptable:.2e})  {verdict}");
}
