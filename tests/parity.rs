//! Cross-backend numeric parity tests.
//!
//! Runs the same inputs through WGPU and MLX backends and asserts
//! element-wise agreement within EPSILON.
//!
//! Compile and run with:
//!   cargo test --features wgpu,mlx -- parity
#![cfg(all(target_os = "macos", feature = "wgpu", feature = "mlx"))]

mod common;

use std::f32::consts::PI;

use common::EPSILON;

type WgpuRuntime = cubecl::wgpu::WgpuRuntime;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn fft_wgpu(input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    gpu_fft::fft::fft::<WgpuRuntime>(&Default::default(), input)
}

fn fft_mlx(input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    gpu_fft::mlx::fft::fft(input)
}

fn ifft_wgpu(real: &[f32], imag: &[f32]) -> Vec<f32> {
    gpu_fft::ifft::ifft::<WgpuRuntime>(&Default::default(), real, imag)
}

fn ifft_mlx(real: &[f32], imag: &[f32]) -> Vec<f32> {
    gpu_fft::mlx::fft::ifft(real, imag)
}

fn assert_fft_parity(label: &str, a: &(Vec<f32>, Vec<f32>), b: &(Vec<f32>, Vec<f32>), eps: f32) {
    assert_eq!(a.0.len(), b.0.len(), "{label}: length mismatch");
    for (i, (&ar, &br)) in a.0.iter().zip(b.0.iter()).enumerate() {
        assert!(
            (ar - br).abs() <= eps,
            "{label} real[{i}]: {ar:.6} vs {br:.6} (diff {:.2e})",
            (ar - br).abs()
        );
    }
    for (i, (&ai, &bi)) in a.1.iter().zip(b.1.iter()).enumerate() {
        assert!(
            (ai - bi).abs() <= eps,
            "{label} imag[{i}]: {ai:.6} vs {bi:.6} (diff {:.2e})",
            (ai - bi).abs()
        );
    }
}

fn assert_vec_parity(label: &str, a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (av - bv).abs() <= eps,
            "{label}[{i}]: {av:.6} vs {bv:.6} (diff {:.2e})",
            (av - bv).abs()
        );
    }
}

fn sine_wave(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (2.0 * PI * i as f32 / n as f32).sin())
        .collect()
}

// ── Forward FFT parity ──────────────────────────────────────────────────────

#[test]
fn parity_fft_impulse() {
    let input = {
        let mut v = vec![0.0f32; 1024];
        v[0] = 1.0;
        v
    };
    let wgpu = fft_wgpu(&input);
    let mlx  = fft_mlx(&input);
    assert_fft_parity("wgpu-vs-mlx impulse", &wgpu, &mlx, EPSILON);
}

#[test]
fn parity_fft_dc() {
    let input = vec![1.0f32; 1024];
    let wgpu = fft_wgpu(&input);
    let mlx  = fft_mlx(&input);
    assert_fft_parity("wgpu-vs-mlx DC", &wgpu, &mlx, EPSILON);
}

#[test]
fn parity_fft_sine() {
    let input = sine_wave(1024);
    let wgpu = fft_wgpu(&input);
    let mlx  = fft_mlx(&input);
    assert_fft_parity("wgpu-vs-mlx sine", &wgpu, &mlx, EPSILON);
}

#[test]
fn parity_fft_arbitrary() {
    let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let wgpu = fft_wgpu(&input);
    let mlx  = fft_mlx(&input);
    assert_fft_parity("wgpu-vs-mlx arb", &wgpu, &mlx, EPSILON);
}

#[test]
fn parity_fft_large_4096() {
    let input = sine_wave(4096);
    let eps = EPSILON * 12.0;
    let wgpu = fft_wgpu(&input);
    let mlx  = fft_mlx(&input);
    assert_fft_parity("wgpu-vs-mlx 4k", &wgpu, &mlx, eps);
}

#[test]
fn parity_fft_large_16384() {
    let input = sine_wave(16384);
    let eps = EPSILON * 14.0;
    let wgpu = fft_wgpu(&input);
    let mlx  = fft_mlx(&input);
    assert_fft_parity("wgpu-vs-mlx 16k", &wgpu, &mlx, eps);
}

// ── Round-trip parity ───────────────────────────────────────────────────────

#[test]
fn parity_roundtrip_sine() {
    let input = sine_wave(1024);

    let (re_w, im_w) = fft_wgpu(&input);
    let (re_x, im_x) = fft_mlx(&input);

    let rt_wgpu = ifft_wgpu(&re_w, &im_w);
    let rt_mlx  = ifft_mlx(&re_x, &im_x);

    let n = input.len();
    assert_vec_parity("rt wgpu-vs-mlx", &rt_wgpu[..n], &rt_mlx[..n], EPSILON);

    for (i, &v) in input.iter().enumerate() {
        assert!(
            (rt_wgpu[i] - v).abs() <= EPSILON,
            "wgpu roundtrip[{i}]: {:.6} vs {:.6}", rt_wgpu[i], v
        );
        assert!(
            (rt_mlx[i] - v).abs() <= EPSILON,
            "mlx roundtrip[{i}]: {:.6} vs {:.6}", rt_mlx[i], v
        );
    }
}

#[test]
fn parity_roundtrip_arbitrary() {
    let input: Vec<f32> = (0..256).map(|x| (x as f32 * 0.1).sin()).collect();

    let (re_w, im_w) = fft_wgpu(&input);
    let (re_x, im_x) = fft_mlx(&input);

    let rt_wgpu = ifft_wgpu(&re_w, &im_w);
    let rt_mlx  = ifft_mlx(&re_x, &im_x);

    let n = input.len();
    let eps = EPSILON * 8.0;
    assert_vec_parity("rt arb wgpu-vs-mlx", &rt_wgpu[..n], &rt_mlx[..n], eps);
}
