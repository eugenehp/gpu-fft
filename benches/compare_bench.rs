use std::f32::consts::PI;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

type WgpuRuntime = cubecl::wgpu::WgpuRuntime;

const SIZES: &[usize] = &[256, 1_024, 4_096, 16_384, 65_536];

fn sine_wave(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (2.0 * PI * i as f32 / n as f32).sin())
        .collect()
}

// ── Compare forward FFT ─────────────────────────────────────────────────────

fn bench_compare_fft(c: &mut Criterion) {
    let device = cubecl::wgpu::WgpuDevice::default();
    let mut group = c.benchmark_group("compare_fft");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for &n in SIZES {
        let input = sine_wave(n);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("wgpu", n), &input, |b, input| {
            b.iter_batched(
                || input.clone(),
                |inp| gpu_fft::fft::fft::<WgpuRuntime>(&device, &inp),
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("mlx", n), &input, |b, input| {
            b.iter_batched(
                || input.clone(),
                |inp| gpu_fft::mlx::fft::fft(&inp),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ── Compare inverse FFT ─────────────────────────────────────────────────────

fn bench_compare_ifft(c: &mut Criterion) {
    let device = cubecl::wgpu::WgpuDevice::default();
    let mut group = c.benchmark_group("compare_ifft");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for &n in SIZES {
        let (real, imag) = gpu_fft::fft::fft::<WgpuRuntime>(&device, &sine_wave(n));
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("wgpu", n),
            &(real.clone(), imag.clone()),
            |b, (real, imag)| {
                b.iter_batched(
                    || (real.clone(), imag.clone()),
                    |(re, im)| gpu_fft::ifft::ifft::<WgpuRuntime>(&device, &re, &im),
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mlx", n),
            &(real.clone(), imag.clone()),
            |b, (real, imag)| {
                b.iter_batched(
                    || (real.clone(), imag.clone()),
                    |(re, im)| gpu_fft::mlx::fft::ifft(&re, &im),
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ── Compare round-trip (FFT → IFFT) ─────────────────────────────────────────

fn bench_compare_roundtrip(c: &mut Criterion) {
    let device = cubecl::wgpu::WgpuDevice::default();
    let mut group = c.benchmark_group("compare_roundtrip");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for &n in SIZES {
        let input = sine_wave(n);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("wgpu", n), &input, |b, input| {
            b.iter_batched(
                || input.clone(),
                |inp| {
                    let (re, im) = gpu_fft::fft::fft::<WgpuRuntime>(&device, &inp);
                    gpu_fft::ifft::ifft::<WgpuRuntime>(&device, &re, &im)
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("mlx", n), &input, |b, input| {
            b.iter_batched(
                || input.clone(),
                |inp| {
                    let (re, im) = gpu_fft::mlx::fft::fft(&inp);
                    gpu_fft::mlx::fft::ifft(&re, &im)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_compare_fft,
    bench_compare_ifft,
    bench_compare_roundtrip,
);
criterion_main!(benches);
