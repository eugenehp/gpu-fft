use std::f32::consts::PI;
use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

type Runtime = cubecl::wgpu::WgpuRuntime;

/// Input sizes (number of samples) swept by every benchmark group.
/// Kept modest because the current DFT kernel is O(N²).
const SIZES: &[usize] = &[64, 256, 1_024, 4_096];

/// Single-frequency sine wave — more realistic than a ramp and signal-independent
/// in terms of DFT computation time.
fn sine_wave(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (2.0 * PI * i as f32 / n as f32).sin())
        .collect()
}

// ── FFT ──────────────────────────────────────────────────────────────────────

/// Measures raw FFT throughput across several input sizes.
///
/// The device is created once outside the loop so that GPU initialisation and
/// shader compilation are not included in the timing. `iter_batched` clones
/// the input in an un-timed setup phase so that memcpy cost is excluded.
fn bench_fft(c: &mut Criterion) {
    let device = cubecl::wgpu::WgpuDevice::default();
    let mut group = c.benchmark_group("fft");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for &n in SIZES {
        let input = sine_wave(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &input, |b, input| {
            b.iter_batched(
                || input.clone(),
                |inp| gpu_fft::fft::fft::<Runtime>(&device, black_box(inp)),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ── IFFT ─────────────────────────────────────────────────────────────────────

/// Measures raw IFFT throughput across several input sizes.
///
/// The FFT is pre-computed outside the timed loop so only the IFFT kernel
/// itself is measured.
fn bench_ifft(c: &mut Criterion) {
    let device = cubecl::wgpu::WgpuDevice::default();
    let mut group = c.benchmark_group("ifft");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for &n in SIZES {
        // Pre-compute the spectrum — only IFFT is timed below.
        let (real, imag) = gpu_fft::fft::fft::<Runtime>(&device, sine_wave(n));
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(real, imag),
            |b, (real, imag)| {
                b.iter_batched(
                    || (real.clone(), imag.clone()),
                    |(re, im)| {
                        gpu_fft::ifft::ifft::<Runtime>(&device, black_box(re), black_box(im))
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ── Round-trip ────────────────────────────────────────────────────────────────

/// Measures the combined FFT → IFFT pipeline latency.
///
/// Useful for gauging end-to-end cost when the application needs both
/// transforms (e.g. convolution in the frequency domain).
fn bench_roundtrip(c: &mut Criterion) {
    let device = cubecl::wgpu::WgpuDevice::default();
    let mut group = c.benchmark_group("roundtrip");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for &n in SIZES {
        let input = sine_wave(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &input, |b, input| {
            b.iter_batched(
                || input.clone(),
                |inp| {
                    let (re, im) = gpu_fft::fft::fft::<Runtime>(&device, black_box(inp));
                    gpu_fft::ifft::ifft::<Runtime>(&device, re, im)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_fft, bench_ifft, bench_roundtrip);
criterion_main!(benches);
