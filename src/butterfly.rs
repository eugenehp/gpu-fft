// Shared Cooley-Tukey radix-2 DIT butterfly kernels and helpers.
// Used by both fft.rs and ifft.rs.
use cubecl::prelude::*;
use std::f32::consts::PI;

// ── Outer butterfly stage ─────────────────────────────────────────────────────

/// Single DIT butterfly stage over global memory.
///
/// N/2 threads are launched; thread `tid` handles the disjoint pair `(i, j)`:
///
/// ```text
/// k    = tid % half_stride
/// i    = (tid / half_stride) * (2 * half_stride) + k
/// j    = i + half_stride
///
/// W    = exp(sign · jπ · k / half_stride)
///          sign = -1  →  forward FFT
///          sign = +1  →  inverse FFT
///
/// out[i] = in[i] + W · in[j]
/// out[j] = in[i] - W · in[j]
/// ```
///
/// Used for **outer** stages where `half_stride ≥ TILE_SIZE / 2` — those stages
/// span workgroup-tile boundaries and cannot be handled in shared memory.
#[cube(launch)]
pub fn butterfly_stage<F: Float>(
    real: &mut Array<F>,
    imag: &mut Array<F>,
    #[comptime] n: usize,
    #[comptime] half_stride: usize,
    #[comptime] forward: bool,
) {
    let tid = ABSOLUTE_POS;
    if tid < n / 2 {
        let k = tid % half_stride;
        let i = (tid / half_stride) * (half_stride * 2) + k;
        let j = i + half_stride;

        let sign = if forward { F::new(-1.0) } else { F::new(1.0) };
        let angle = sign * F::new(PI) * F::cast_from(k) / F::cast_from(half_stride);
        let cos_a = F::cos(angle);
        let sin_a = F::sin(angle);

        let ur = real[i];
        let ui = imag[i];
        let vr = cos_a * real[j] - sin_a * imag[j];
        let vi = sin_a * real[j] + cos_a * imag[j];

        real[i] = ur + vr;
        imag[i] = ui + vi;
        real[j] = ur - vr;
        imag[j] = ui - vi;
    }
}

// ── Inner (shared-memory) butterfly kernel ────────────────────────────────────

/// Multi-stage DIT butterfly kernel using workgroup shared memory.
///
/// Each workgroup of `tile / 2` threads owns a contiguous tile of `tile` elements.
/// All `stages` butterfly stages for that tile are processed in a **single launch**,
/// with a `sync_units()` workgroup barrier between consecutive stages.  Because the
/// data never leaves shared memory between stages, inter-stage global-memory
/// round-trips are eliminated — replacing `stages` separate kernel dispatches with
/// one.
///
/// ### Usage
///
/// Call this kernel with:
/// - `CubeCount = (N / tile).max(1)` workgroups
/// - `CubeDim  = tile / 2` threads per workgroup
/// - `stages   = log₂(tile).min(log₂(N))` — the number of inner stages to fuse
///
/// Shared-memory budget: `2 * tile * sizeof(F)` bytes per workgroup.
/// With `tile = TILE_SIZE = 1024` and F = f32: **8 192 bytes** — within the
/// WebGPU / Vulkan / Metal 16 384-byte minimum guaranteed per workgroup.
#[cube(launch)]
pub fn butterfly_inner<F: Float>(
    real: &mut Array<F>,
    imag: &mut Array<F>,
    #[comptime] tile: usize,   // elements per workgroup (= 2 × threads per WG)
    #[comptime] stages: usize, // number of butterfly stages to fuse in this kernel
    #[comptime] forward: bool,
) {
    // Allocate two shared-memory buffers of `tile` scalars each.
    // Size is comptime → the GPU driver allocates it statically.
    let mut s_real = SharedMemory::<F>::new(tile);
    let mut s_imag = SharedMemory::<F>::new(tile);

    let half_tile = tile / 2; // comptime = tile / 2

    // ABSOLUTE_POS is the global thread index (0 .. N/2 - 1).
    // local : thread index within this tile (0 .. half_tile - 1)
    // base  : first global element of this tile
    let tid   = ABSOLUTE_POS;
    let local = tid % half_tile;
    let base  = (tid / half_tile) * tile;

    // ── Load: each thread loads 2 elements from global → shared ──────────────
    s_real[local]             = real[base + local];
    s_real[local + half_tile] = real[base + local + half_tile];
    s_imag[local]             = imag[base + local];
    s_imag[local + half_tile] = imag[base + local + half_tile];

    sync_cube(); // ensure all threads have loaded before the first butterfly

    // ── Fused butterfly stages in shared memory ───────────────────────────────
    // `stages` is comptime → the loop is unrolled by the CubeCL compiler.
    // In each unrolled iteration `s` is a compile-time constant, so
    // `1 << s` becomes a literal and `% hs` / `/ hs` become bitwise ops.
    for s in 0..stages {
        let hs = 1_usize << s; // half-stride for this stage (1, 2, 4, …)

        let k = local % hs;
        let i = (local / hs) * (hs * 2) + k; // local index of upper element
        let j = i + hs;                        // local index of lower element

        let sign = if forward { F::new(-1.0) } else { F::new(1.0) };
        let angle = sign * F::new(PI) * F::cast_from(k) / F::cast_from(hs);
        let cos_a = F::cos(angle);
        let sin_a = F::sin(angle);

        let ur = s_real[i];
        let ui = s_imag[i];
        let vr = cos_a * s_real[j] - sin_a * s_imag[j];
        let vi = sin_a * s_real[j] + cos_a * s_imag[j];

        s_real[i] = ur + vr;
        s_imag[i] = ui + vi;
        s_real[j] = ur - vr;
        s_imag[j] = ui - vi;

        sync_cube(); // barrier before next stage reads shared memory
    }

    // ── Write back: each thread stores 2 elements from shared → global ────────
    real[base + local]             = s_real[local];
    real[base + local + half_tile] = s_real[local + half_tile];
    imag[base + local]             = s_imag[local];
    imag[base + local + half_tile] = s_imag[local + half_tile];
}

// ── Batched outer butterfly stage ─────────────────────────────────────────────

/// Batched single DIT butterfly stage over global memory.
///
/// Processes `batch_size` independent signals of length `n` packed end-to-end in
/// `real` / `imag` (signal `b` starts at offset `b * n`).
///
/// Uses a flat 1-D dispatch — `ABSOLUTE_POS` (usize) encodes both the signal
/// index and the butterfly-pair position within it:
///
/// ```text
/// signal = tid / (n / 2)
/// pos    = tid % (n / 2)
/// ```
///
/// ### Launch parameters
/// ```text
/// CubeCount = (outer_wg, 1, 1)
///             where outer_wg = ceil(batch_size * n/2 / WORKGROUP_SIZE)
/// CubeDim   = (WORKGROUP_SIZE, 1, 1)
/// ```
///
/// `batch_size` is comptime so the guard `tid < batch_size * (n/2)` can be
/// evaluated as a compile-time constant, keeping the per-stage kernel cost down.
#[cube(launch)]
pub fn butterfly_stage_batch<F: Float>(
    real: &mut Array<F>,
    imag: &mut Array<F>,
    #[comptime] n: usize,
    #[comptime] half_stride: usize,
    #[comptime] batch_size: usize,
    #[comptime] forward: bool,
) {
    let tid = ABSOLUTE_POS; // usize — flat global thread index

    if tid < batch_size * (n / 2) {
        let signal = tid / (n / 2);
        let pos    = tid % (n / 2);
        let offset = signal * n;

        let k = pos % half_stride;
        let i = (pos / half_stride) * (half_stride * 2) + k;
        let j = i + half_stride;

        let sign  = if forward { F::new(-1.0) } else { F::new(1.0) };
        let angle = sign * F::new(PI) * F::cast_from(k) / F::cast_from(half_stride);
        let cos_a = F::cos(angle);
        let sin_a = F::sin(angle);

        let ur = real[offset + i];
        let ui = imag[offset + i];
        let vr = cos_a * real[offset + j] - sin_a * imag[offset + j];
        let vi = sin_a * real[offset + j] + cos_a * imag[offset + j];

        real[offset + i] = ur + vr;
        imag[offset + i] = ui + vi;
        real[offset + j] = ur - vr;
        imag[offset + j] = ui - vi;
    }
}

// ── Batched inner (shared-memory) butterfly kernel ────────────────────────────

/// Multi-stage DIT butterfly kernel using shared memory — batched over many signals.
///
/// Uses a flat 1-D dispatch: each workgroup of `tile/2` threads handles exactly
/// one tile of one signal.  `ABSOLUTE_POS` (usize) encodes both the signal and
/// tile within the signal:
///
/// ```text
/// local       = ABSOLUTE_POS % half_tile          (thread position within tile)
/// tile_global = ABSOLUTE_POS / half_tile           (global tile index)
/// signal      = tile_global / tiles_per_signal
/// tile_in_sig = tile_global % tiles_per_signal
/// base        = signal * n + tile_in_sig * tile
/// ```
///
/// `tiles_per_signal = (n / tile).max(1)` — derived from the comptime params `n`
/// and `tile`; no separate `batch_size` comptime param is needed because the
/// dispatch count already encodes it.
///
/// ### Launch parameters
/// ```text
/// CubeCount = (tiles_per_signal * batch_size, 1, 1)
/// CubeDim   = (tile / 2, 1, 1)   — one thread per butterfly pair in the tile
/// ```
///
/// Shared-memory budget: `2 * tile * sizeof(F)` bytes — same as the scalar kernel.
#[cube(launch)]
pub fn butterfly_inner_batch<F: Float>(
    real: &mut Array<F>,
    imag: &mut Array<F>,
    #[comptime] n: usize,      // per-signal length (power-of-two, padded)
    #[comptime] tile: usize,   // elements per workgroup tile (≤ n, power-of-two)
    #[comptime] stages: usize, // number of butterfly stages to fuse
    #[comptime] forward: bool,
) {
    let mut s_real = SharedMemory::<F>::new(tile);
    let mut s_imag = SharedMemory::<F>::new(tile);

    let half_tile        = tile / 2;        // threads per workgroup (comptime)
    let tiles_per_signal = (n / tile).max(1); // comptime

    let tid         = ABSOLUTE_POS;
    let local       = tid % half_tile;          // local thread index within tile
    let tile_global = tid / half_tile;          // global tile index across all signals
    let signal      = tile_global / tiles_per_signal;
    let tile_in_sig = tile_global % tiles_per_signal;
    let base        = signal * n + tile_in_sig * tile;

    // ── Load from global → shared ─────────────────────────────────────────────
    s_real[local]             = real[base + local];
    s_real[local + half_tile] = real[base + local + half_tile];
    s_imag[local]             = imag[base + local];
    s_imag[local + half_tile] = imag[base + local + half_tile];

    sync_cube();

    // ── Fused butterfly stages in shared memory ───────────────────────────────
    for s in 0..stages {
        let hs = 1_usize << s;

        let k = local % hs;
        let i = (local / hs) * (hs * 2) + k;
        let j = i + hs;

        let sign  = if forward { F::new(-1.0) } else { F::new(1.0) };
        let angle = sign * F::new(PI) * F::cast_from(k) / F::cast_from(hs);
        let cos_a = F::cos(angle);
        let sin_a = F::sin(angle);

        let ur = s_real[i];
        let ui = s_imag[i];
        let vr = cos_a * s_real[j] - sin_a * s_imag[j];
        let vi = sin_a * s_real[j] + cos_a * s_imag[j];

        s_real[i] = ur + vr;
        s_imag[i] = ui + vi;
        s_real[j] = ur - vr;
        s_imag[j] = ui - vi;

        sync_cube();
    }

    // ── Write back shared → global ────────────────────────────────────────────
    real[base + local]             = s_real[local];
    real[base + local + half_tile] = s_real[local + half_tile];
    imag[base + local]             = s_imag[local];
    imag[base + local + half_tile] = s_imag[local + half_tile];
}

// ── CPU helper ────────────────────────────────────────────────────────────────

/// Bit-reversal permutation index: reverses the lowest `bits` bits of `x`.
///
/// Used to rearrange the input into the order required by the DIT FFT before
/// uploading to the GPU.  Running this O(N) pass on the CPU is negligible
/// compared to the GPU kernel time.
#[inline]
pub fn bit_reverse(mut x: usize, bits: u32) -> usize {
    let mut r = 0usize;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}
