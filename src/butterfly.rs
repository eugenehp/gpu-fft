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
