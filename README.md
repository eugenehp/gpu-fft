# GPU-FFT

A Rust library for GPU-accelerated FFT and IFFT built on [CubeCL](https://github.com/tracel-ai/cubecl).

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Benchmarks](#benchmarks)
- [Algorithm](#algorithm)
- [License](#license)

## Features

- **Cooley-Tukey radix-2 DIT FFT/IFFT** — O(N log₂ N), runs `log₂ N` butterfly-stage
  kernel dispatches of N/2 threads each instead of a single O(N²) DFT pass
- **Batched FFT/IFFT** — process many signals in a single GPU pass; kernel-launch
  overhead is amortised across the whole batch regardless of batch size
- **Automatic zero-padding** — non-power-of-two inputs are padded to the next power of two
- **Power Spectral Density** — one-sided PSD with correct 1/N normalisation
- **Dominant frequency detection** — local-peak search above a threshold
- **wgpu and CUDA backends** via CubeCL feature flags

## Roadmap

- [x] GPU kernel (CubeCL / wgpu)
- [x] Precomputed twiddle factors (WIP branch)
- [x] Cooley-Tukey radix-2 FFT — O(N log N)
- [x] Shared-memory tiling — inner stages fused into one launch; 10 → 1 dispatch for N ≤ 1 024
- [x] Batched FFT/IFFT — multiple signals in one kernel launch
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

### API

#### Scalar (single signal)

```rust
use gpu_fft::{fft, ifft};

// Forward transform — zero-pads to the next power of two automatically.
let signal = vec![1.0f32, 0.0, 0.0, 0.0];
let (real, imag) = fft(&signal);           // (Vec<f32>, Vec<f32>), length = 4

// Inverse transform — pass the direct output of fft unchanged.
let recovered = ifft(&real, &imag);        // Vec<f32>, length = 2N
let time_domain = &recovered[..signal.len()];
let imaginary   = &recovered[signal.len()..]; // ≈ 0 for real inputs
```

#### Batched (many signals, one GPU pass)

```rust
use gpu_fft::{fft_batch, ifft_batch};

// All signals are zero-padded to the same length (next power of two of the
// longest signal) and processed in a single set of kernel launches.
let signals: Vec<Vec<f32>> = vec![
    vec![1.0, 0.0, 0.0, 0.0],  // impulse → all-ones spectrum
    vec![1.0, 1.0, 1.0, 1.0],  // DC      → [4, 0, 0, 0]
    vec![0.0, 1.0, 0.0, -1.0], // alternating
];

// Returns one (real, imag) pair per signal, each of length n.
let spectra: Vec<(Vec<f32>, Vec<f32>)> = fft_batch(&signals);

// Inverse — pass the direct output of fft_batch.
// Returns one Vec<f32> of length 2n per signal: [real | imag].
let recovered: Vec<Vec<f32>> = ifft_batch(&spectra);
```

> `fft_batch` and `ifft_batch` accept slices of any length; all signals are
> padded to the same power-of-two length determined by the longest signal in
> the batch.

## Benchmarks

### Scalar baselines

| Latency | Throughput |
|---------|------------|
| ![Latency vs N](bench-results/charts/latency.svg) | ![Throughput vs N](bench-results/charts/throughput.svg) |

| Benchmark | N | Mean | Throughput |
|-----------|--:|-----:|------------|
| fft |    64 |  249 µs |  257 Kelem/s |
| fft |   256 |  390 µs |  657 Kelem/s |
| fft | 1 024 |  666 µs | 1.54 Melem/s |
| fft | 4 096 |  557 µs | 7.35 Melem/s |
| fft | 16 384 |  697 µs | 23.50 Melem/s |
| fft | 65 536 | 1.12 ms | 58.66 Melem/s |
| | | | |
| ifft |    64 |  250 µs |  256 Kelem/s |
| ifft |   256 |  467 µs |  548 Kelem/s |
| ifft | 1 024 |  436 µs | 2.35 Melem/s |
| ifft | 4 096 |  894 µs | 4.58 Melem/s |
| ifft | 16 384 |  744 µs | 22.01 Melem/s |
| ifft | 65 536 | 1.18 ms | 55.63 Melem/s |
| | | | |
| roundtrip |    64 |  505 µs |  127 Kelem/s |
| roundtrip |   256 |  785 µs |  326 Kelem/s |
| roundtrip | 1 024 |  806 µs | 1.27 Melem/s |
| roundtrip | 4 096 |  934 µs | 4.38 Melem/s |
| roundtrip | 16 384 | 1.32 ms | 12.40 Melem/s |
| roundtrip | 65 536 | 2.15 ms | 30.44 Melem/s |

### Batch vs scalar — throughput at fixed N, batch = 16

![Throughput vs N](bench-results/charts/batch_signal.svg)

_Solid lines: scalar baseline.  Dashed lines: batch=16.
 At N = 65 536 the batch FFT path delivers **131 Melem/s vs 58.7 Melem/s** — 2.2×
 the scalar throughput — because 16 signals share the same set of kernel launches._

### Batch FFT — signal-length sweep (batch = 16 fixed)

| N | Mean | Throughput |
|--:|-----:|------------|
| 256 | 431 µs | 9.50 Melem/s |
| 1 024 | 506 µs | 32.40 Melem/s |
| 4 096 | 869 µs | 75.41 Melem/s |
| 16 384 | 2.24 ms | 116.81 Melem/s |
| 65 536 | 7.98 ms | 131.48 Melem/s |

### Batch IFFT — signal-length sweep (batch = 16 fixed)

| N | Mean | Throughput |
|--:|-----:|------------|
| 256 | 428 µs | 9.58 Melem/s |
| 1 024 | 499 µs | 32.80 Melem/s |
| 4 096 | 913 µs | 71.75 Melem/s |
| 16 384 | 2.78 ms | 94.32 Melem/s |
| 65 536 | 10.64 ms | 98.54 Melem/s |

### Batch round-trip — signal-length sweep (batch = 16 fixed)

| N | Mean | Throughput |
|--:|-----:|------------|
| 256 | 861 µs | 4.76 Melem/s |
| 1 024 | 1.02 ms | 15.98 Melem/s |
| 4 096 | 1.85 ms | 35.46 Melem/s |
| 16 384 | 5.44 ms | 48.16 Melem/s |
| 65 536 | 19.72 ms | 53.19 Melem/s |

### Batch throughput vs batch size — N = 4 096 fixed

![Throughput vs batch size](bench-results/charts/batch_size.svg)

| Benchmark | Batch | Mean | Throughput |
|-----------|------:|-----:|------------|
| fft_batch |  1 |  467 µs |  8.76 Melem/s |
| fft_batch |  4 |  546 µs | 30.03 Melem/s |
| fft_batch | 16 |  885 µs | 74.06 Melem/s |
| fft_batch | 64 | 1.99 ms | 131.74 Melem/s |
| | | | |
| ifft_batch |  1 |  467 µs |  8.77 Melem/s |
| ifft_batch |  4 |  553 µs | 29.63 Melem/s |
| ifft_batch | 16 |  900 µs | 72.81 Melem/s |
| ifft_batch | 64 | 2.15 ms | 122.17 Melem/s |
| | | | |
| roundtrip_batch |  1 | 1.24 ms |  3.30 Melem/s |
| roundtrip_batch |  4 | 1.43 ms | 11.43 Melem/s |
| roundtrip_batch | 16 | 2.05 ms | 31.90 Melem/s |
| roundtrip_batch | 64 | 4.91 ms | 53.37 Melem/s |

### Batch vs sequential — N = 4 096 fixed

![Batch vs sequential throughput](bench-results/charts/vs_sequential.svg)

Processing 64 signals sequentially costs **29.62 ms** for FFT and **59.79 ms** for
round-trip; the batch equivalents take **1.95 ms** and **4.91 ms** — **15.2×** and
**12.2× speedups** — because kernel-launch overhead is paid once per stage, not
once per signal per stage.

#### FFT

| Method | Batch | Mean | Throughput | vs sequential |
|--------|------:|-----:|------------|--------------|
| fft_batch |  1 |  465 µs |  8.81 Melem/s | 1.0× |
| fft_batch |  4 |  539 µs | 30.41 Melem/s | 3.4× |
| fft_batch | 16 |  875 µs | 74.91 Melem/s | 8.5× |
| fft_batch | 64 | 1.95 ms | 134.23 Melem/s | **15.2×** |
| | | | | |
| sequential |  1 |  464 µs |  8.83 Melem/s | 1.0× |
| sequential |  4 | 1.85 ms |  8.83 Melem/s | — |
| sequential | 16 | 7.43 ms |  8.82 Melem/s | — |
| sequential | 64 | 29.62 ms |  8.85 Melem/s | — |

#### IFFT

| Method | Batch | Mean | Throughput | vs sequential |
|--------|------:|-----:|------------|--------------|
| ifft_batch |  1 |  468 µs |  8.76 Melem/s | 1.0× |
| ifft_batch |  4 |  601 µs | 27.26 Melem/s | 3.1× |
| ifft_batch | 16 |  967 µs | 67.74 Melem/s | 7.7× |
| ifft_batch | 64 | 2.97 ms | 88.22 Melem/s | **10.1×** |
| | | | | |
| sequential |  1 |  464 µs |  8.83 Melem/s | 1.0× |
| sequential |  4 | 2.45 ms |  6.70 Melem/s | — |
| sequential | 16 | 7.59 ms |  8.63 Melem/s | — |
| sequential | 64 | 37.13 ms |  7.06 Melem/s | — |

#### Round-trip (FFT → IFFT)

| Method | Batch | Mean | Throughput | vs sequential |
|--------|------:|-----:|------------|--------------|
| roundtrip_batch |  1 | 1.24 ms |  3.30 Melem/s | 0.75× |
| roundtrip_batch |  4 | 1.43 ms | 11.43 Melem/s | 2.6× |
| roundtrip_batch | 16 | 2.05 ms | 31.90 Melem/s | 7.3× |
| roundtrip_batch | 64 | 4.91 ms | 53.37 Melem/s | **12.2×** |
| | | | | |
| sequential |  1 |  934 µs |  4.38 Melem/s | 1.0× |
| sequential |  4 | 3.74 ms |  4.38 Melem/s | — |
| sequential | 16 | 14.95 ms |  4.38 Melem/s | — |
| sequential | 64 | 59.79 ms |  4.38 Melem/s | — |

_Sequential baseline: N calls to the scalar `roundtrip` (same total data).
Batch=1 is slightly slower than scalar due to batch-path bookkeeping overhead._

### Running benchmarks

```shell
# Run everything, generate charts, save report
./scripts/bench.sh

# Single benchmark group or specific size
cargo bench --features wgpu --bench fft_bench -- fft/65536
cargo bench --features wgpu --bench fft_bench -- "fft_batch/batch_size"
cargo bench --features wgpu --bench fft_bench -- "fft_batch/signal_len"
cargo bench --features wgpu --bench fft_bench -- "fft_batch_vs_sequential"
cargo bench --features wgpu --bench fft_bench -- "ifft_batch/batch_size"
cargo bench --features wgpu --bench fft_bench -- "ifft_batch/signal_len"
cargo bench --features wgpu --bench fft_bench -- "ifft_batch_vs_sequential"
cargo bench --features wgpu --bench fft_bench -- "roundtrip_batch"
cargo bench --features wgpu --bench fft_bench -- "roundtrip_batch/signal_len"

# Save a named Criterion baseline, then compare after changes
./scripts/bench.sh -- --save-baseline before
./scripts/bench.sh -- --baseline before
```

Results are saved to `bench-results/latest.md` and archived under
`bench-results/archive/`.

_Charts and the tables above are regenerated automatically by `./scripts/bench.sh`.
Shaded bands on charts show the Criterion 95% confidence interval._

Full results (95% CI, std dev, raw Criterion output) → [`bench-results/latest.md`](bench-results/latest.md)

## Algorithm

### FFT (single signal)

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

### IFFT (single signal)

Same butterfly with positive twiddle factors, followed by a CPU-side 1/N divide.

### Batched FFT/IFFT

`fft_batch` and `ifft_batch` process a batch of `B` signals of length `N` in
the same number of kernel launches as a single transform — the inner and outer
stage dispatches are sized to cover all `B` signals at once.

**CPU side** (un-timed):
1. Pad all signals to the same length (next power of two of the longest).
2. Bit-reverse permute each signal.
3. Pack into a flat GPU buffer of size `B × N`.

**GPU side** (same launch count as scalar):

- **Inner stages** — `butterfly_inner_batch` is launched with
  `B × tiles_per_signal` workgroups (each workgroup handles one tile of one
  signal):
  ```
  local       = ABSOLUTE_POS % (tile/2)      thread within tile
  tile_global = ABSOLUTE_POS / (tile/2)      global tile index
  signal      = tile_global / tiles_per_signal
  tile_in_sig = tile_global % tiles_per_signal
  base        = signal * N + tile_in_sig * tile
  ```
  Shared memory per workgroup is identical to the scalar kernel (≤ 8 KiB).

- **Outer stages** — `butterfly_stage_batch` is launched with enough
  workgroups to cover all `B × N/2` butterfly pairs at once:
  ```
  signal = ABSOLUTE_POS / (N/2)
  pos    = ABSOLUTE_POS % (N/2)
  offset = signal * N
  ```
  `batch_size` is a `#[comptime]` parameter so the guard `tid < B × (N/2)` is
  a compile-time constant with no runtime branch cost.

**Result**: kernel-launch overhead (`~65 µs` per dispatch on typical hardware)
is paid once per stage, not once per signal per stage.  For `B = 64` signals
and `N = 4 096` (3 launches) the saving is `(64 − 1) × 3 × 65 µs ≈ 12 ms` —
consistent with the observed **13.8× throughput gain** over sequential calls.

Total launches per batched transform (identical to scalar regardless of B):

| N | Inner | Outer | Total |
|---|------:|------:|------:|
| ≤ 1 024 | 1 | 0 | **1** |
| 4 096 | 1 | 2 | **3** |
| 65 536 | 1 | 6 | **7** |

## License

This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.

## Copyright

© 2025-2026, Eugene Hauptmann
