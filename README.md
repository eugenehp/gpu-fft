# GPU-FFT

A Rust library for GPU-accelerated FFT and IFFT built on [CubeCL](https://github.com/tracel-ai/cubecl).

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Benchmarks](#benchmarks)
- [License](#license)

## Features

- **Cooley-Tukey radix-2 DIT FFT/IFFT** — O(N log₂ N), runs `log₂ N` butterfly-stage
  kernel dispatches of N/2 threads each instead of a single O(N²) DFT pass
- **Automatic zero-padding** — non-power-of-two inputs are padded to the next power of two
- **Power Spectral Density** — one-sided PSD with correct 1/N normalisation
- **Dominant frequency detection** — local-peak search above a threshold
- **wgpu and CUDA backends** via CubeCL feature flags

## Roadmap

- [x] GPU kernel (CubeCL / wgpu)
- [x] Precomputed twiddle factors (WIP branch)
- [x] Cooley-Tukey radix-2 FFT — O(N log N)
- [x] Shared-memory tiling — inner stages fused into one launch; 10 → 1 dispatch for N ≤ 1 024
- [ ] Batched FFT (multiple signals in one kernel launch)
- [ ] Mixed-radix / radix-4 — halve the outer-stage count for large N

## Requirements

- Rust 1.84 or later
- A Vulkan, Metal, or DX12 GPU (wgpu backend) **or** an NVIDIA GPU (CUDA backend)

## Installation

```bash
cargo add gpu_fft -F wgpu
```

## Usage

```bash
cargo run --example simple -F wgpu
```

The example generates a 15 Hz sine wave, runs FFT, identifies the dominant
frequency in the one-sided spectrum, then reconstructs the signal with IFFT and
reports the round-trip error.

### Example Output

```
Input
  samples    : 1000  (zero-padded to 1024 for FFT)
  sample rate: 200 Hz
  frequency  : 15 Hz
  first 5    : [0.0000, 0.4540, 0.8090, 0.9877, 0.9511]

FFT  completed in 5.21ms
Dominant frequencies (0 … 100 Hz):
     15.04 Hz  power       243.16

IFFT completed in 1.84ms
Round-trip max error : 3.58e-6  (limit 5·log₂N·ε = 5.96e-6)  ✓
```

> The first run includes GPU shader compilation (~50 ms one-time cost per
> kernel variant). Subsequent calls reuse the compiled shaders.

## Benchmarks

### Performance

![Latency](bench-results/charts/latency.svg)

![Throughput](bench-results/charts/throughput.svg)

_Charts and the table below are regenerated automatically by `./scripts/bench.sh`.
Shaded bands show the Criterion 95 % confidence interval._

| Benchmark | N | Mean | Throughput |
|-----------|--:|-----:|------------|
| fft |    64 | 249 µs | 257 Kelem/s |
| fft |   256 | 396 µs | 646 Kelem/s |
| fft | 1 024 | 417 µs | 2.46 Melem/s |
| fft | 4 096 | 487 µs | 8.42 Melem/s |
| fft | 16 384 | 579 µs | 28.3 Melem/s |
| fft | 65 536 | 976 µs | 67.2 Melem/s |
| | | | |
| ifft |    64 | 250 µs | 256 Kelem/s |
| ifft |   256 | 395 µs | 647 Kelem/s |
| ifft | 1 024 | 402 µs | 2.55 Melem/s |
| ifft | 4 096 | 474 µs | 8.65 Melem/s |
| ifft | 16 384 | 657 µs | 24.9 Melem/s |
| ifft | 65 536 | 1.14 ms | 57.5 Melem/s |
| | | | |
| roundtrip |    64 | 505 µs | 127 Kelem/s |
| roundtrip |   256 | 792 µs | 323 Kelem/s |
| roundtrip | 1 024 | 803 µs | 1.27 Melem/s |
| roundtrip | 4 096 | 942 µs | 4.35 Melem/s |
| roundtrip | 16 384 | 1.17 ms | 14.0 Melem/s |
| roundtrip | 65 536 | 2.14 ms | 30.7 Melem/s |

Full results (95 % CI, std dev, raw Criterion output) → [`bench-results/latest.md`](bench-results/latest.md)

### Running benchmarks

```shell
# Run everything, generate charts, save report
./scripts/bench.sh

# Single size
cargo bench --features wgpu --bench fft_bench -- fft/65536

# Save a named Criterion baseline, then compare after changes
./scripts/bench.sh -- --save-baseline before
./scripts/bench.sh -- --baseline before
```

Results are saved to `bench-results/latest.md` and archived under
`bench-results/archive/`.

## Algorithm

### FFT

1. **Bit-reverse permute** the input on the CPU (O(N), negligible).
2. Upload to GPU.
3. **Inner stages** (where `half_stride < TILE_SIZE/2 = 512`) — fused into a
   **single kernel launch** using workgroup shared memory.  Each workgroup of
   512 threads loads 1 024 elements into `SharedMemory<f32>`, runs all ≤ 10
   butterfly stages with `sync_cube()` barriers between stages, then writes
   back.  This eliminates ~10 × 65 µs of kernel-launch overhead per call.
4. **Outer stages** — one kernel launch each, reading/writing global memory.
   ```
   W  = exp(-jπ · k / half_stride)     (forward; positive for inverse)
   out[i] = in[i] + W · in[j]
   out[j] = in[i] − W · in[j]
   ```
5. Read back.

Total launches per transform:

| N | Inner | Outer | Total |
|---|------:|------:|------:|
| ≤ 1 024 | 1 | 0 | **1** |
| 4 096 | 1 | 2 | **3** |
| 65 536 | 1 | 6 | **7** |

Each unique `(tile, stages, direction)` / `(N, half_stride, direction)` triple
compiles to a separate specialised kernel cached by CubeCL after the first run.

### IFFT

Same butterfly with positive twiddle factors, followed by a CPU-side 1/N divide.

## License

This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.

## Copyright

© 2025-2026, Eugene Hauptmann
