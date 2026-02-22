use std::f32::consts::PI;

use gpu_fft::{fft, fft_batch};

mod common;
use common::{assert_approx, assert_slice_approx};

// ── Correctness vs. single-signal FFT ────────────────────────────────────────

/// Each output of `fft_batch` must exactly match the result of calling `fft` on
/// the same signal individually.
#[test]
fn test_fft_batch_matches_single() {
    let signals: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],             // impulse
        vec![1.0, 1.0, 1.0, 1.0],             // DC
        vec![0.0f32, 1.0, 0.0, -1.0],         // alternating
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ];

    let batch_results = fft_batch(&signals);
    assert_eq!(batch_results.len(), signals.len());

    for (b, signal) in signals.iter().enumerate() {
        let (exp_real, exp_imag) = fft(signal);
        let (got_real, got_imag) = &batch_results[b];

        assert_slice_approx(got_real, &exp_real, &format!("batch[{b}] real"));
        assert_slice_approx(got_imag, &exp_imag, &format!("batch[{b}] imag"));
    }
}

// ── Impulse signals ───────────────────────────────────────────────────────────

/// FFT of an impulse [1, 0, 0, …] must be all-ones real, all-zeros imaginary.
#[test]
fn test_fft_batch_impulses() {
    let n = 8;
    let batch: Vec<Vec<f32>> = (0..4)
        .map(|_| {
            let mut s = vec![0.0f32; n];
            s[0] = 1.0;
            s
        })
        .collect();

    let results = fft_batch(&batch);
    assert_eq!(results.len(), 4);

    for (b, (real, imag)) in results.iter().enumerate() {
        assert_slice_approx(real, &[1.0; 8], &format!("batch[{b}] real"));
        assert_slice_approx(imag, &[0.0; 8], &format!("batch[{b}] imag"));
    }
}

// ── DC signals ────────────────────────────────────────────────────────────────

/// FFT of [1, 1, …, 1] must give [N, 0, …, 0] real, all-zeros imaginary.
#[test]
fn test_fft_batch_dc_signals() {
    let n = 8;
    let batch = vec![vec![1.0f32; n]; 3];

    let results = fft_batch(&batch);
    assert_eq!(results.len(), 3);

    for (b, (real, imag)) in results.iter().enumerate() {
        assert_approx(real[0], n as f32, &format!("batch[{b}] real[0]"));
        assert_approx(imag[0], 0.0,      &format!("batch[{b}] imag[0]"));
        for i in 1..n {
            assert_approx(real[i], 0.0, &format!("batch[{b}] real[{i}]"));
            assert_approx(imag[i], 0.0, &format!("batch[{b}] imag[{i}]"));
        }
    }
}

// ── Single-frequency sine waves ───────────────────────────────────────────────

/// Batched FFT of sin(2π·n/N) must put power only at bins 1 and N−1.
#[test]
fn test_fft_batch_single_frequency_sine() {
    let n     = 8usize;
    let half  = n as f32 / 2.0;
    let signal: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * i as f32 / n as f32).sin())
        .collect();

    // Run multiple identical signals in one batch.
    let batch = vec![signal; 5];
    let results = fft_batch(&batch);
    assert_eq!(results.len(), 5);

    for (b, (real, imag)) in results.iter().enumerate() {
        // DC bin
        assert_approx(real[0], 0.0, &format!("batch[{b}] real[0]"));
        assert_approx(imag[0], 0.0, &format!("batch[{b}] imag[0]"));
        // Bin 1: X[1] = −jN/2
        assert_approx(real[1], 0.0,   &format!("batch[{b}] real[1]"));
        assert_approx(imag[1], -half, &format!("batch[{b}] imag[1]"));
        // Middle bins: zero
        for k in 2..n - 1 {
            assert_approx(real[k], 0.0, &format!("batch[{b}] real[{k}]"));
            assert_approx(imag[k], 0.0, &format!("batch[{b}] imag[{k}]"));
        }
        // Bin N−1: X[N−1] = +jN/2
        assert_approx(real[n - 1], 0.0,  &format!("batch[{b}] real[{}]", n - 1));
        assert_approx(imag[n - 1], half, &format!("batch[{b}] imag[{}]", n - 1));
    }
}

// ── All-zero input ────────────────────────────────────────────────────────────

/// FFT of the zero vector must be the zero vector.
#[test]
fn test_fft_batch_zero_input() {
    let batch = vec![vec![0.0f32; 8]; 4];
    let results = fft_batch(&batch);

    for (b, (real, imag)) in results.iter().enumerate() {
        assert_slice_approx(real, &[0.0; 8], &format!("batch[{b}] real"));
        assert_slice_approx(imag, &[0.0; 8], &format!("batch[{b}] imag"));
    }
}

// ── Linearity across signals ──────────────────────────────────────────────────

/// FFT(a·x) == a·FFT(x) must hold for every signal in the batch independently.
#[test]
fn test_fft_batch_linearity() {
    let scale   = 3.0f32;
    let base    = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let scaled: Vec<f32> = base.iter().map(|&x| x * scale).collect();

    let results = fft_batch(&[base.clone(), scaled.clone()]);
    assert_eq!(results.len(), 2);

    let (real_base, imag_base)     = &results[0];
    let (real_scaled, imag_scaled) = &results[1];
    let n = real_base.len();

    for i in 0..n {
        assert_approx(real_scaled[i], real_base[i] * scale, &format!("real[{i}]"));
        assert_approx(imag_scaled[i], imag_base[i] * scale, &format!("imag[{i}]"));
    }
}

// ── Empty batch ───────────────────────────────────────────────────────────────

#[test]
fn test_fft_batch_empty() {
    let results = fft_batch(&[]);
    assert!(results.is_empty());
}

// ── Single-element batch ──────────────────────────────────────────────────────

/// A batch of size 1 must equal the result of the scalar `fft`.
#[test]
fn test_fft_batch_single_element() {
    let signal = vec![1.0f32, -1.0, 2.0, 0.5];
    let (exp_real, exp_imag) = fft(&signal);

    let results = fft_batch(&[signal]);
    assert_eq!(results.len(), 1);

    assert_slice_approx(&results[0].0, &exp_real, "single-element batch real");
    assert_slice_approx(&results[0].1, &exp_imag, "single-element batch imag");
}

// ── Large batch ───────────────────────────────────────────────────────────────

/// Process 64 independent impulses; every result must be all-ones real / all-zeros imag.
#[test]
fn test_fft_batch_large() {
    let n       = 64usize;
    let count   = 64usize;
    let mut impulse = vec![0.0f32; n];
    impulse[0] = 1.0;

    let batch: Vec<Vec<f32>> = (0..count).map(|_| impulse.clone()).collect();
    let results = fft_batch(&batch);
    assert_eq!(results.len(), count);

    for (b, (real, imag)) in results.iter().enumerate() {
        for k in 0..n {
            assert_approx(real[k], 1.0, &format!("batch[{b}] real[{k}]"));
            assert_approx(imag[k], 0.0, &format!("batch[{b}] imag[{k}]"));
        }
    }
}

// ── Mixed signals — independence check ───────────────────────────────────────

/// Each signal in a mixed batch must be unaffected by its neighbours.
#[test]
fn test_fft_batch_signal_independence() {
    // Two very different signals interleaved in the same batch.
    let impulse  = { let mut v = vec![0.0f32; 8]; v[0] = 1.0; v };
    let dc       = vec![1.0f32; 8];
    let batch    = vec![impulse.clone(), dc.clone(), impulse.clone(), dc.clone()];
    let results  = fft_batch(&batch);

    let (ref_impulse_r, ref_impulse_i) = fft(&impulse);
    let (ref_dc_r,      ref_dc_i)      = fft(&dc);

    for (b, (real, imag)) in results.iter().enumerate() {
        if b % 2 == 0 {
            assert_slice_approx(real, &ref_impulse_r, &format!("impulse batch[{b}] real"));
            assert_slice_approx(imag, &ref_impulse_i, &format!("impulse batch[{b}] imag"));
        } else {
            assert_slice_approx(real, &ref_dc_r, &format!("dc batch[{b}] real"));
            assert_slice_approx(imag, &ref_dc_i, &format!("dc batch[{b}] imag"));
        }
    }
}
