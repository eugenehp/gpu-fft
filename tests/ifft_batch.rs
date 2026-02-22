use std::f32::consts::PI;

use gpu_fft::{fft, fft_batch, ifft, ifft_batch};

mod common;
use common::assert_slice_approx;

// ── Round-trip: ifft_batch(fft_batch(x)) == x ────────────────────────────────

/// IFFT(FFT(x))[0..n] must recover x for every signal in the batch.
#[test]
fn test_ifft_batch_roundtrip_arbitrary() {
    let signals: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![-3.0, 1.5, 0.0, -2.0, 4.0, -1.0, 0.5, 2.5],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let n = signals[0].len();

    let spectra  = fft_batch(&signals);
    let recovered = ifft_batch(&spectra);
    assert_eq!(recovered.len(), signals.len());

    for (b, (signal, output)) in signals.iter().zip(recovered.iter()).enumerate() {
        assert_slice_approx(&output[..n], signal, &format!("batch[{b}] real (round-trip)"));
        assert_slice_approx(&output[n..], &vec![0.0f32; n], &format!("batch[{b}] imag (~0)"));
    }
}

// ── Correctness vs. scalar ifft ───────────────────────────────────────────────

/// Each `ifft_batch` output must match the scalar `ifft` result exactly.
#[test]
fn test_ifft_batch_matches_single() {
    let signals: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0, 1.0],
        vec![1.0, -1.0, 1.0, -1.0],
        vec![0.5, 0.25, 0.125, 0.0625],
    ];

    let spectra = fft_batch(&signals);
    let batch_out = ifft_batch(&spectra);
    assert_eq!(batch_out.len(), signals.len());

    for (b, (re, im)) in spectra.iter().enumerate() {
        let scalar_out = ifft(re, im);
        assert_slice_approx(&batch_out[b], &scalar_out, &format!("batch[{b}]"));
    }
}

// ── Impulse spectrum → unit impulse ──────────────────────────────────────────

/// IFFT of an all-ones spectrum must recover a unit impulse [1, 0, 0, …].
#[test]
fn test_ifft_batch_all_ones_spectrum() {
    let n       = 8usize;
    let count   = 3;
    // All-ones real spectrum, zero imaginary.
    let spectra: Vec<(Vec<f32>, Vec<f32>)> = (0..count)
        .map(|_| (vec![1.0f32; n], vec![0.0f32; n]))
        .collect();

    let results = ifft_batch(&spectra);
    assert_eq!(results.len(), count);

    // IFFT([1,1,…,1]) = [1, 0, 0, …] (unit impulse).
    let mut expected = vec![0.0f32; n];
    expected[0] = 1.0;

    for (b, output) in results.iter().enumerate() {
        assert_slice_approx(&output[..n], &expected, &format!("batch[{b}] real"));
        assert_slice_approx(&output[n..], &vec![0.0f32; n], &format!("batch[{b}] imag"));
    }
}

// ── Sine-wave round-trip ──────────────────────────────────────────────────────

#[test]
fn test_ifft_batch_roundtrip_sine() {
    let n = 8usize;
    let signal: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * i as f32 / n as f32).sin())
        .collect();

    let batch = vec![signal.clone(); 4];
    let spectra = fft_batch(&batch);
    let recovered = ifft_batch(&spectra);

    for (b, output) in recovered.iter().enumerate() {
        assert_slice_approx(&output[..n], &signal, &format!("batch[{b}] real (sine round-trip)"));
        assert_slice_approx(&output[n..], &vec![0.0f32; n], &format!("batch[{b}] imag (~0)"));
    }
}

// ── Empty batch ───────────────────────────────────────────────────────────────

#[test]
fn test_ifft_batch_empty() {
    let results = ifft_batch(&[]);
    assert!(results.is_empty());
}

// ── Single-element batch ──────────────────────────────────────────────────────

#[test]
fn test_ifft_batch_single_element() {
    let signal = vec![1.0f32, 2.0, 3.0, 4.0];
    let (re, im) = fft(&signal);
    let scalar_out = ifft(&re, &im);

    let batch_out = ifft_batch(&[(re, im)]);
    assert_eq!(batch_out.len(), 1);
    assert_slice_approx(&batch_out[0], &scalar_out, "single-element batch");
}

// ── Large batch ───────────────────────────────────────────────────────────────

/// Round-trip 64 identical signals; every output must recover the original.
#[test]
fn test_ifft_batch_large_roundtrip() {
    let n      = 64usize;
    let count  = 64;
    let signal: Vec<f32> = (0..n).map(|i| i as f32).collect();

    let batch   = vec![signal.clone(); count];
    let spectra = fft_batch(&batch);
    let recovered = ifft_batch(&spectra);
    assert_eq!(recovered.len(), count);

    for (b, output) in recovered.iter().enumerate() {
        assert_slice_approx(&output[..n], &signal, &format!("batch[{b}] real (large round-trip)"));
        assert_slice_approx(&output[n..], &vec![0.0f32; n], &format!("batch[{b}] imag (~0)"));
    }
}

// ── Signal independence ───────────────────────────────────────────────────────

/// Each signal in a mixed batch must be recovered independently.
#[test]
fn test_ifft_batch_signal_independence() {
    let signal_a = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let signal_b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let batch = vec![
        signal_a.clone(),
        signal_b.clone(),
        signal_a.clone(),
        signal_b.clone(),
    ];

    let spectra   = fft_batch(&batch);
    let recovered = ifft_batch(&spectra);
    let n         = signal_a.len();

    for (b, output) in recovered.iter().enumerate() {
        let expected = if b % 2 == 0 { &signal_a } else { &signal_b };
        assert_slice_approx(&output[..n], expected, &format!("batch[{b}] real"));
        assert_slice_approx(&output[n..], &vec![0.0f32; n], &format!("batch[{b}] imag"));
    }
}

// ── 1/N scaling correctness ───────────────────────────────────────────────────

/// IFFT must apply the 1/N normalization correctly: IFFT(FFT([a,…,a]))[0] = a.
#[test]
fn test_ifft_batch_scaling() {
    // DC signal: [a, a, a, a] → FFT → [4a, 0, 0, 0] → IFFT → [a, a, a, a].
    for &amplitude in &[1.0f32, 2.0, 0.5, 10.0] {
        let signal = vec![amplitude; 8];
        let spectra = fft_batch(&[signal.clone()]);
        let result  = ifft_batch(&spectra);

        let n = signal.len();
        assert_slice_approx(
            &result[0][..n],
            &signal,
            &format!("DC amplitude={amplitude} real"),
        );
    }
}
