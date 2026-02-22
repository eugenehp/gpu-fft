use std::f32::consts::PI;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

type Runtime = cubecl::wgpu::WgpuRuntime;

// ── Benchmark parameters ──────────────────────────────────────────────────────

/// Signal lengths swept by every single-transform benchmark (powers of two).
/// O(N log N) — even N = 65 536 completes in milliseconds on a GPU.
const SIZES: &[usize] = &[256, 1_024, 4_096, 16_384, 65_536];

/// Batch sizes swept when signal length is held constant.
const BATCH_SIZES: &[usize] = &[1, 4, 16, 64];

/// Signal length used when sweeping batch size in isolation.
const BATCH_N: usize = 4_096;

/// Batch size used when sweeping signal length in isolation.
const BATCH_FIXED: usize = 16;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn sine_wave(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (2.0 * PI * i as f32 / n as f32).sin())
        .collect()
}

fn make_batch(batch_size: usize, n: usize) -> Vec<Vec<f32>> {
    (0..batch_size).map(|_| sine_wave(n)).collect()
}

/// Pre-compute a batch of FFT spectra so only the IFFT is timed.
fn make_spectra(
    device: &cubecl::wgpu::WgpuDevice,
    batch_size: usize,
    n: usize,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    make_batch(batch_size, n)
        .iter()
        .map(|s| gpu_fft::fft::fft::<Runtime>(device, s))
        .collect()
}

// ── Scalar FFT ────────────────────────────────────────────────────────────────

/// Raw FFT throughput for individual signals across a range of sizes.
///
/// The GPU device is created once so that adapter initialisation and shader
/// compilation are excluded. `iter_batched` clones the input in an un-timed
/// setup phase to exclude host-side memcpy from the measurement.
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
                |inp| gpu_fft::fft::fft::<Runtime>(&device, &inp),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ── Scalar IFFT ───────────────────────────────────────────────────────────────

/// Raw IFFT throughput for individual spectra across a range of sizes.
///
/// The spectrum is pre-computed outside the timed loop so only the inverse
/// butterfly passes and the 1/N CPU scaling are measured.
fn bench_ifft(c: &mut Criterion) {
    let device = cubecl::wgpu::WgpuDevice::default();
    let mut group = c.benchmark_group("ifft");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for &n in SIZES {
        let (real, imag) = gpu_fft::fft::fft::<Runtime>(&device, &sine_wave(n));
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(real, imag),
            |b, (real, imag)| {
                b.iter_batched(
                    || (real.clone(), imag.clone()),
                    |(re, im)| gpu_fft::ifft::ifft::<Runtime>(&device, &re, &im),
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ── Scalar round-trip ─────────────────────────────────────────────────────────

/// Combined FFT → IFFT pipeline latency for individual signals.
///
/// Useful for end-to-end cost estimation when the application needs both
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
                    let (re, im) = gpu_fft::fft::fft::<Runtime>(&device, &inp);
                    gpu_fft::ifft::ifft::<Runtime>(&device, &re, &im)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ── Batch FFT — sweep batch size ──────────────────────────────────────────────

/// Batched FFT throughput as batch size grows, with signal length fixed at
/// `BATCH_N`.
///
/// Throughput is the total number of elements processed per second
/// (`batch_size × n`), so the benefit of amortising kernel-launch overhead
/// over many signals shows up as rising throughput with batch size.
fn bench_fft_batch_size(c: &mut Criterion) {
    let device = cubecl::wgpu::WgpuDevice::default();
    let mut group = c.benchmark_group("fft_batch/batch_size");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for &bs in BATCH_SIZES {
        let batch = make_batch(bs, BATCH_N);
        group.throughput(Throughput::Elements((bs * BATCH_N) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(bs), &batch, |b, batch| {
            b.iter_batched(
                || batch.clone(),
                |b| gpu_fft::fft::fft_batch::<Runtime>(&device, &b),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ── Batch FFT — sweep signal length ──────────────────────────────────────────

/// Batched FFT throughput across signal lengths with batch size fixed at
/// `BATCH_FIXED`.
///
/// Lets you compare per-sample cost of the batch path vs the scalar path at
/// each transform size (run alongside `bench_fft` to compute the ratio).
fn bench_fft_batch_signal_len(c: &mut Criterion) {
    let device = cubecl::wgpu::WgpuDevice::default();
    let mut group = c.benchmark_group("fft_batch/signal_len");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for &n in SIZES {
        let batch = make_batch(BATCH_FIXED, n);
        group.throughput(Throughput::Elements((BATCH_FIXED * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &batch, |b, batch| {
            b.iter_batched(
                || batch.clone(),
                |b| gpu_fft::fft::fft_batch::<Runtime>(&device, &b),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ── Batch FFT vs sequential scalar FFT ───────────────────────────────────────

/// Head-to-head comparison: `fft_batch` vs a CPU loop of `fft` calls, both
/// processing `batch_size` signals of `BATCH_N` elements.
///
/// The "sequential" variant measures the realistic cost a caller would pay
/// without the batch API; any throughput advantage of "batch" comes from
/// reduced kernel-launch overhead and better GPU utilisation.
fn bench_fft_batch_vs_sequential(c: &mut Criterion) {
    let device = cubecl::wgpu::WgpuDevice::default();
    let mut group = c.benchmark_group("fft_batch_vs_sequential");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for &bs in BATCH_SIZES {
        let batch = make_batch(bs, BATCH_N);
        group.throughput(Throughput::Elements((bs * BATCH_N) as u64));

        // batched — single dispatch per stage
        group.bench_with_input(
            BenchmarkId::new("batch", bs),
            &batch,
            |b, batch| {
                b.iter_batched(
                    || batch.clone(),
                    |b| gpu_fft::fft::fft_batch::<Runtime>(&device, &b),
                    BatchSize::SmallInput,
                );
            },
        );

        // sequential — one dispatch per signal per stage
        group.bench_with_input(
            BenchmarkId::new("sequential", bs),
            &batch,
            |b, batch| {
                b.iter_batched(
                    || batch.clone(),
                    |b| {
                        b.iter()
                            .map(|s| gpu_fft::fft::fft::<Runtime>(&device, s))
                            .collect::<Vec<_>>()
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ── Batch IFFT — sweep batch size ────────────────────────────────────────────

/// Mirrors `bench_fft_batch_size` but for the inverse transform.
///
/// Spectra are pre-computed outside the timed loop so only the IFFT butterflies
/// and 1/N CPU scaling are measured.
fn bench_ifft_batch_size(c: &mut Criterion) {
    let device = cubecl::wgpu::WgpuDevice::default();
    let mut group = c.benchmark_group("ifft_batch/batch_size");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for &bs in BATCH_SIZES {
        let spectra = make_spectra(&device, bs, BATCH_N);
        group.throughput(Throughput::Elements((bs * BATCH_N) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(bs), &spectra, |b, spectra| {
            b.iter_batched(
                || spectra.clone(),
                |sp| gpu_fft::ifft::ifft_batch::<Runtime>(&device, &sp),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ── Batch IFFT — sweep signal length ─────────────────────────────────────────

/// Mirrors `bench_fft_batch_signal_len` but for the inverse transform.
fn bench_ifft_batch_signal_len(c: &mut Criterion) {
    let device = cubecl::wgpu::WgpuDevice::default();
    let mut group = c.benchmark_group("ifft_batch/signal_len");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for &n in SIZES {
        let spectra = make_spectra(&device, BATCH_FIXED, n);
        group.throughput(Throughput::Elements((BATCH_FIXED * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &spectra, |b, spectra| {
            b.iter_batched(
                || spectra.clone(),
                |sp| gpu_fft::ifft::ifft_batch::<Runtime>(&device, &sp),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ── Batch IFFT vs sequential scalar IFFT ─────────────────────────────────────

/// Head-to-head comparison: `ifft_batch` vs a CPU loop of `ifft` calls.
fn bench_ifft_batch_vs_sequential(c: &mut Criterion) {
    let device = cubecl::wgpu::WgpuDevice::default();
    let mut group = c.benchmark_group("ifft_batch_vs_sequential");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for &bs in BATCH_SIZES {
        let spectra = make_spectra(&device, bs, BATCH_N);
        group.throughput(Throughput::Elements((bs * BATCH_N) as u64));

        // batched
        group.bench_with_input(
            BenchmarkId::new("batch", bs),
            &spectra,
            |b, spectra| {
                b.iter_batched(
                    || spectra.clone(),
                    |sp| gpu_fft::ifft::ifft_batch::<Runtime>(&device, &sp),
                    BatchSize::SmallInput,
                );
            },
        );

        // sequential
        group.bench_with_input(
            BenchmarkId::new("sequential", bs),
            &spectra,
            |b, spectra| {
                b.iter_batched(
                    || spectra.clone(),
                    |sp| {
                        sp.iter()
                            .map(|(re, im)| gpu_fft::ifft::ifft::<Runtime>(&device, re, im))
                            .collect::<Vec<_>>()
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ── Batch round-trip ──────────────────────────────────────────────────────────

/// End-to-end `fft_batch` → `ifft_batch` pipeline latency across batch sizes.
///
/// Useful for frequency-domain processing workloads (e.g. block convolution)
/// where both transforms are required.
fn bench_roundtrip_batch(c: &mut Criterion) {
    let device = cubecl::wgpu::WgpuDevice::default();
    let mut group = c.benchmark_group("roundtrip_batch");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for &bs in BATCH_SIZES {
        let batch = make_batch(bs, BATCH_N);
        group.throughput(Throughput::Elements((bs * BATCH_N) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(bs), &batch, |b, batch| {
            b.iter_batched(
                || batch.clone(),
                |b| {
                    let spectra = gpu_fft::fft::fft_batch::<Runtime>(&device, &b);
                    gpu_fft::ifft::ifft_batch::<Runtime>(&device, &spectra)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ── Batch round-trip — sweep signal length ────────────────────────────────────

/// End-to-end `fft_batch` → `ifft_batch` pipeline latency across signal lengths
/// with batch size fixed at `BATCH_FIXED`.
fn bench_roundtrip_batch_signal_len(c: &mut Criterion) {
    let device = cubecl::wgpu::WgpuDevice::default();
    let mut group = c.benchmark_group("roundtrip_batch/signal_len");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for &n in SIZES {
        let batch = make_batch(BATCH_FIXED, n);
        group.throughput(Throughput::Elements((BATCH_FIXED * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &batch, |b, batch| {
            b.iter_batched(
                || batch.clone(),
                |b| {
                    let spectra = gpu_fft::fft::fft_batch::<Runtime>(&device, &b);
                    gpu_fft::ifft::ifft_batch::<Runtime>(&device, &spectra)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    // scalar baselines
    bench_fft,
    bench_ifft,
    bench_roundtrip,
    // batch FFT
    bench_fft_batch_size,
    bench_fft_batch_signal_len,
    bench_fft_batch_vs_sequential,
    // batch IFFT
    bench_ifft_batch_size,
    bench_ifft_batch_signal_len,
    bench_ifft_batch_vs_sequential,
    // batch round-trip
    bench_roundtrip_batch,
    bench_roundtrip_batch_signal_len,
);
criterion_main!(benches);
