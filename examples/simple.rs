use std::time::Instant;

use gpu_fft::{psd, utils};

fn main() {
    let sample_rate = 200.0f32; // Hz
    let frequency = 15.0f32;   // Hz
    let duration = 5.0f32;     // seconds  →  1 000 samples (zero-padded to 1 024)
    let threshold = 100.0f32;  // minimum PSD power to report a dominant frequency

    // ── Generate input signal ─────────────────────────────────────────────────
    let input = utils::generate_sine_wave(frequency, sample_rate, duration);
    let n_orig = input.len(); // 1 000 — original length before zero-padding

    println!("Input");
    println!("  samples    : {n_orig}  (zero-padded to {} for FFT)", n_orig.next_power_of_two());
    println!("  sample rate: {sample_rate} Hz");
    println!("  frequency  : {frequency} Hz");
    println!("  first 5    : {:.4?}", &input[..5]);
    println!();

    // ── FFT ───────────────────────────────────────────────────────────────────
    // Non-power-of-two inputs are zero-padded internally; output length is
    // n_orig.next_power_of_two() (here: 1 024).
    let t = Instant::now();
    let (real, imag) = gpu_fft::fft(&input);
    println!("FFT  completed in {:?}", t.elapsed());

    let n = real.len(); // padded length (1 024)

    // ── Spectral analysis (one-sided) ─────────────────────────────────────────
    // For a real-valued signal X[N-k] = X[k]*, so only bins 0 … N/2 are unique.
    // Compute PSD over all N bins (correct 1/N normalisation), then slice off
    // the redundant upper half before peak-searching.
    let n_unique = n / 2 + 1; // 513 unique bins (0 Hz … Nyquist)

    let spectrum    = psd::psd(&real, &imag);
    let frequencies = utils::calculate_one_sided_frequencies(n, sample_rate);
    let dominant    = utils::find_dominant_frequencies(&spectrum[..n_unique], &frequencies, threshold);

    println!("Dominant frequencies (0 … {:.0} Hz):", sample_rate / 2.0);
    if dominant.is_empty() {
        println!("  (none above threshold {threshold})");
    } else {
        for (freq, power) in &dominant {
            println!("  {:>8.2} Hz  power {:>10.2}", freq, power);
        }
    }
    println!();

    // ── IFFT ──────────────────────────────────────────────────────────────────
    let t = Instant::now();
    let output = gpu_fft::ifft(&real, &imag);
    println!("IFFT completed in {:?}", t.elapsed());

    // output[0..n]  = reconstructed real signal (length N = 1 024)
    // output[n..2n] = reconstructed imaginary signal (≈ 0 for real inputs)
    //
    // Compare only the first n_orig samples — the zero-padded tail is expected
    // to be ~0 and carries no information from the original signal.
    let reconstructed = &output[..n_orig];

    // For a radix-2 FFT + IFFT, accumulated f32 round-trip error is O(log₂N · ε).
    let max_err   = input.iter().zip(reconstructed).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let acceptable = 5.0 * (n as f32).log2() * f32::EPSILON;
    let verdict   = if max_err <= acceptable { "✓" } else { "✗ unexpectedly large" };

    println!(
        "Round-trip max error : {max_err:.2e}  (limit 5·log₂N·ε = {acceptable:.2e})  {verdict}"
    );
}
