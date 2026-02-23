# Changelog

All notable changes to **gpu-fft** are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.1.0] — 2026-02-22

### Added

#### Batched FFT / IFFT
- **`fft_batch` / `ifft_batch` public API** — accepts `&[Vec<f32>]`; pads all
  signals to the same power-of-two length, packs them into a flat GPU buffer,
  and runs the full butterfly pipeline in a single set of kernel launches
  regardless of batch size
- `butterfly_inner_batch` CubeCL kernel — extends the shared-memory inner pass
  to cover `B × tiles_per_signal` workgroups simultaneously
- `butterfly_stage_batch` CubeCL kernel — extends each outer-stage dispatch to
  cover all `B × N/2` butterfly pairs at once; `batch_size` is a `#[comptime]`
  parameter so the thread-guard is a compile-time constant
- **Test suite** — `tests/fft_batch.rs` (9 cases: impulse, DC, alternating,
  random, mixed lengths) and `tests/ifft_batch.rs` (7 cases)

#### Radix-4 outer butterfly
- **`butterfly_stage_radix4`** (scalar) and **`butterfly_stage_radix4_batch`**
  (batched) CubeCL kernels — each of the N/4 threads loads four elements
  `{p, p+q, p+2q, p+3q}` and executes two radix-2 butterfly stages entirely in
  registers, eliminating one global-memory round-trip per pair of outer stages
- **Analytical W₂b** — the second twiddle is derived from the first without an
  extra trig call: `cos₂b = +sin₂a`, `sin₂b = −cos₂a` (forward); sign flips
  for the inverse transform
- **Updated outer loop** in `fft`, `ifft`, `fft_batch`, `ifft_batch`:
  `while s + 1 < m` dispatches radix-4; trailing `if s < m` dispatches a
  single radix-2 when the remaining outer-stage count is odd
- **Launch count reduction:**

  | N | v1.0 launches | v1.1 launches | Saved |
  |---|:---:|:---:|:---:|
  | ≤ 1 024 | 1 | 1 | — |
  | 2 048 | 2 | 2 | — |
  | 4 096 | 3 | **2** | 1 |
  | 8 192 | 4 | **3** | 1 |
  | 16 384 | 5 | **4** | 1 |
  | 65 536 | 7 | **4** | 3 |

#### Benchmarks (`benches/fft_bench.rs`)
- **9 new Criterion groups** covering the batched path:
  `fft_batch/batch_size`, `fft_batch/signal_len`, `fft_batch_vs_sequential`,
  `ifft_batch/batch_size`, `ifft_batch/signal_len`, `ifft_batch_vs_sequential`,
  `roundtrip_batch`, `roundtrip_batch/signal_len`
- **6 radix-4 groups** (`fft_radix4_outer`, `ifft_radix4_outer`,
  `roundtrip_radix4_outer`, `fft_batch_radix4_outer`, `ifft_batch_radix4_outer`,
  `roundtrip_batch_radix4_outer`) sweeping `RADIX4_OUTER_SIZES` =
  {2 048, 4 096, 8 192, 16 384, 65 536} — one size per outer-stage dispatch
  pattern (trailing r2 · pure r4 · mixed r4+r2)
- `bench.sh` forwards extra flags (e.g. `--save-baseline`, `--baseline`) to
  the Criterion binary for before/after comparisons

#### Charts & report generation (`scripts/export_bench.py`)
- Full chart pipeline built from scratch: colour palette, marker shapes,
  human-readable labels, signal-length / batch-size / vs-sequential /
  radix-4 chart families
- **7 SVG charts** auto-generated on every bench run:
  `latency.svg`, `throughput.svg`, `batch_signal.svg`, `batch_size.svg`,
  `vs_sequential.svg`, `radix4_outer.svg`, `radix4_batch_outer.svg`
- `_TABLE_SECTIONS` partitions the Markdown summary table into labelled
  groups; a catch-all appends any unrecognised groups automatically
- Image paths in generated reports now use `../charts/<name>` so archive
  files under `bench-results/archive/` render correctly

#### Tests
- `test_fft_impulse_large_even` (N = 4 096) — single radix-4 outer dispatch
- `test_fft_impulse_large_odd` (N = 8 192) — mixed r4 + trailing r2
- `test_fft_dc_large` (N = 4 096) — DC bin correctness at large N
- `test_roundtrip_large_even` / `test_roundtrip_large_odd` — round-trip
  max error bounded by `5 · log₂(N) · ε_machine`

#### Documentation
- **Algorithm section** — radix-4 butterfly formula and W₂b derivation written
  out in full; launch-count tables updated for both scalar and batched paths
- **Benchmark tables** — scalar baselines, all batch signal-length/batch-size
  sweeps, batch-vs-sequential with speedup column, two radix-4 outer tables
  (scalar + batch=16)
- **Citation section** — BibTeX `@software` entry and APA plain-text
- **Roadmap** — radix-4 item marked complete; batch item marked complete

### Changed
- Outer butterfly loop in `fft` / `ifft` / `fft_batch` / `ifft_batch` replaced
  from sequential single-stage radix-2 dispatches to paired radix-4 dispatches
  with an optional trailing radix-2

### Performance

Measured 2026-02-22, commit `b231b99`, wgpu backend.

| Benchmark | v1.0.0 | v1.1.0 | Change |
|-----------|-------:|-------:|-------:|
| `fft` N = 4 096 | 557 µs · 7.35 Melem/s | **447 µs · 9.16 Melem/s** | −20% latency |
| `fft` N = 65 536 | 1.12 ms · 58.7 Melem/s | **940 µs · 69.7 Melem/s** | −16% latency |
| `ifft` N = 4 096 | 894 µs · 4.58 Melem/s | **476 µs · 8.61 Melem/s** | −47% latency |
| `ifft` N = 65 536 | 1.18 ms · 55.6 Melem/s | **1.12 ms · 58.8 Melem/s** | −5% latency |
| `fft_batch` N = 65 536, B = 16 | — | **7.90 ms · 133 Melem/s** | new |
| `fft_batch` vs sequential, B = 64 | — | **13.5×** | new |
| `roundtrip_batch` vs sequential, B = 64 | — | **14.6×** | new |

---

## [1.0.0] — 2026-02-22

Initial public release.

### Added
- **Cooley-Tukey radix-2 DIT FFT and IFFT** — O(N log₂ N), `log₂ N` kernel
  dispatches of N/2 threads each
- **Shared-memory inner-stage fusion** — all butterfly stages where
  `half_stride < TILE_SIZE/2 = 512` are fused into a single kernel launch
  using workgroup shared memory and `sync_cube()` barriers; for N ≤ 1 024 the
  entire transform is a single dispatch
- **Automatic zero-padding** — non-power-of-two inputs padded to the next
  power of two transparently
- **Power Spectral Density** — one-sided PSD with correct 1/N normalisation
- **Dominant frequency detection** — local-peak search with configurable
  amplitude threshold
- **wgpu backend** (Vulkan / Metal / DX12) and **CUDA backend** via CubeCL
  feature flags (`-F wgpu` / `-F cuda`)
- Scalar benchmark groups `fft`, `ifft`, `roundtrip` over
  N ∈ {256, 1 024, 4 096, 16 384, 65 536}
- `latency.svg` and `throughput.svg` charts
