// Shared Cooley-Tukey DIT butterfly kernels and helpers.
// Radix-2 kernels are used for inner (shared-memory) stages and the optional
// trailing stage when the number of outer stages is odd.
// Radix-4 kernels fuse two consecutive radix-2 stages into one dispatch,
// halving kernel-launch overhead for the outer (global-memory) stages.
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

// ── Radix-4 outer butterfly stage ────────────────────────────────────────────

/// Single DIT radix-4 butterfly stage over global memory.
///
/// Combines two consecutive radix-2 stages (half-strides `q` and `2q`) into a
/// **single** kernel dispatch.  Each of the N/4 threads handles a group of 4
/// elements at positions `{p, p+q, p+2q, p+3q}` and computes both stages
/// entirely in registers before writing results back — eliminating one
/// kernel-launch round-trip compared to two separate radix-2 dispatches.
///
/// ### Stage-1 (half-stride `q`)
/// ```text
/// W1 = exp(sign · jπ · k / q)          k = p % q
///
/// u0 = in[p]    + W1 · in[p+q]
/// u1 = in[p]    − W1 · in[p+q]
/// u2 = in[p+2q] + W1 · in[p+3q]
/// u3 = in[p+2q] − W1 · in[p+3q]
/// ```
///
/// ### Stage-2 (half-stride `2q`)
/// ```text
/// W2a = exp(sign · jπ · k / (2q))
/// W2b = W2a · exp(sign · jπ/2)   →  cos₂b = neg_sign · sin₂a
///                                     sin₂b = sign     · cos₂a
///
/// y0 = u0 + W2a · u2   →  out[p]
/// y2 = u0 − W2a · u2   →  out[p+2q]
/// y1 = u1 + W2b · u3   →  out[p+q]
/// y3 = u1 − W2b · u3   →  out[p+3q]
/// ```
///
/// ### Launch parameters
/// ```text
/// CubeCount = (ceil(N/4 / WORKGROUP_SIZE), 1, 1)
/// CubeDim   = (WORKGROUP_SIZE, 1, 1)
/// ```
///
/// Each unique `(n, q, forward)` triple compiles to a specialised kernel cached
/// by CubeCL, consistent with the radix-2 `butterfly_stage` approach.
#[cube(launch)]
pub fn butterfly_stage_radix4<F: Float>(
    real: &mut Array<F>,
    imag: &mut Array<F>,
    #[comptime] n: usize,
    #[comptime] q: usize,       // quarter-stride = half-stride of the lower stage
    #[comptime] forward: bool,
) {
    let tid = ABSOLUTE_POS;
    if tid < n / 4 {
        let k     = tid % q;
        let group = tid / q;
        let p     = group * (q * 4) + k;

        // ── Load 4 elements ───────────────────────────────────────────────────
        let ar = real[p];
        let ai = imag[p];
        let br = real[p + q];
        let bi = imag[p + q];
        let cr = real[p + q * 2];
        let ci = imag[p + q * 2];
        let dr = real[p + q * 3];
        let di = imag[p + q * 3];

        // ── Stage-1 twiddle: W1 = exp(sign · jπ · k / q) ────────────────────
        let sign    = if forward { F::new(-1.0) } else { F::new(1.0) };
        let angle1  = sign * F::new(PI) * F::cast_from(k) / F::cast_from(q);
        let cos1    = F::cos(angle1);
        let sin1    = F::sin(angle1);

        let w1b_r = cos1 * br - sin1 * bi;
        let w1b_i = sin1 * br + cos1 * bi;
        let w1d_r = cos1 * dr - sin1 * di;
        let w1d_i = sin1 * dr + cos1 * di;

        // ── Stage-1 outputs (held in registers, no global write) ──────────────
        let u0r = ar + w1b_r;   let u0i = ai + w1b_i;
        let u1r = ar - w1b_r;   let u1i = ai - w1b_i;
        let u2r = cr + w1d_r;   let u2i = ci + w1d_i;
        let u3r = cr - w1d_r;   let u3i = ci - w1d_i;

        // ── Stage-2 twiddles ──────────────────────────────────────────────────
        // W2a = exp(sign · jπ · k / (2q))
        let angle2a = sign * F::new(PI) * F::cast_from(k) / F::cast_from(q * 2);
        let cos2a   = F::cos(angle2a);
        let sin2a   = F::sin(angle2a);

        // W2b = W2a · exp(sign · jπ/2), derived without a second trig call:
        //   forward  (sign = −1): angle2b = angle2a − π/2
        //                          cos2b =  sin2a,  sin2b = −cos2a
        //   inverse  (sign = +1): angle2b = angle2a + π/2
        //                          cos2b = −sin2a,  sin2b =  cos2a
        //   In both cases: cos2b = neg_sign · sin2a,  sin2b = sign · cos2a
        let neg_sign = if forward { F::new(1.0) } else { F::new(-1.0) };
        let cos2b    = neg_sign * sin2a;
        let sin2b    = sign * cos2a;

        // W2a · u2
        let w2a_u2r = cos2a * u2r - sin2a * u2i;
        let w2a_u2i = sin2a * u2r + cos2a * u2i;
        // W2b · u3
        let w2b_u3r = cos2b * u3r - sin2b * u3i;
        let w2b_u3i = sin2b * u3r + cos2b * u3i;

        // ── Stage-2 outputs → write back to global memory ─────────────────────
        real[p]         = u0r + w2a_u2r;
        imag[p]         = u0i + w2a_u2i;
        real[p + q * 2] = u0r - w2a_u2r;
        imag[p + q * 2] = u0i - w2a_u2i;
        real[p + q]     = u1r + w2b_u3r;
        imag[p + q]     = u1i + w2b_u3i;
        real[p + q * 3] = u1r - w2b_u3r;
        imag[p + q * 3] = u1i - w2b_u3i;
    }
}

// ── Batched radix-4 outer butterfly stage ─────────────────────────────────────

/// Batched DIT radix-4 butterfly stage over global memory.
///
/// Processes `batch_size` independent signals of length `n` packed end-to-end
/// in `real` / `imag` (signal `b` starts at byte offset `b * n`).
///
/// Uses a flat 1-D dispatch — `ABSOLUTE_POS` encodes both the signal index and
/// the butterfly-group position within it:
///
/// ```text
/// signal = tid / (n / 4)
/// pos    = tid % (n / 4)
/// ```
///
/// `pos` is then handled identically to the scalar `butterfly_stage_radix4`
/// (the same two-stage radix-4 butterfly in registers).
///
/// ### Launch parameters
/// ```text
/// CubeCount = (ceil(batch_size × N/4 / WORKGROUP_SIZE), 1, 1)
/// CubeDim   = (WORKGROUP_SIZE, 1, 1)
/// ```
///
/// `batch_size` is comptime so the guard `tid < batch_size × (n/4)` is a
/// compile-time constant with no runtime branch cost.
#[cube(launch)]
pub fn butterfly_stage_radix4_batch<F: Float>(
    real: &mut Array<F>,
    imag: &mut Array<F>,
    #[comptime] n: usize,
    #[comptime] q: usize,
    #[comptime] batch_size: usize,
    #[comptime] forward: bool,
) {
    let tid = ABSOLUTE_POS;
    if tid < batch_size * (n / 4) {
        let signal = tid / (n / 4);
        let pos    = tid % (n / 4);
        let offset = signal * n;

        let k     = pos % q;
        let group = pos / q;
        let p     = group * (q * 4) + k;

        // ── Load 4 elements ───────────────────────────────────────────────────
        let ar = real[offset + p];
        let ai = imag[offset + p];
        let br = real[offset + p + q];
        let bi = imag[offset + p + q];
        let cr = real[offset + p + q * 2];
        let ci = imag[offset + p + q * 2];
        let dr = real[offset + p + q * 3];
        let di = imag[offset + p + q * 3];

        // ── Stage-1 twiddle ───────────────────────────────────────────────────
        let sign    = if forward { F::new(-1.0) } else { F::new(1.0) };
        let angle1  = sign * F::new(PI) * F::cast_from(k) / F::cast_from(q);
        let cos1    = F::cos(angle1);
        let sin1    = F::sin(angle1);

        let w1b_r = cos1 * br - sin1 * bi;
        let w1b_i = sin1 * br + cos1 * bi;
        let w1d_r = cos1 * dr - sin1 * di;
        let w1d_i = sin1 * dr + cos1 * di;

        let u0r = ar + w1b_r;   let u0i = ai + w1b_i;
        let u1r = ar - w1b_r;   let u1i = ai - w1b_i;
        let u2r = cr + w1d_r;   let u2i = ci + w1d_i;
        let u3r = cr - w1d_r;   let u3i = ci - w1d_i;

        // ── Stage-2 twiddles ──────────────────────────────────────────────────
        let angle2a = sign * F::new(PI) * F::cast_from(k) / F::cast_from(q * 2);
        let cos2a   = F::cos(angle2a);
        let sin2a   = F::sin(angle2a);

        let neg_sign = if forward { F::new(1.0) } else { F::new(-1.0) };
        let cos2b    = neg_sign * sin2a;
        let sin2b    = sign * cos2a;

        let w2a_u2r = cos2a * u2r - sin2a * u2i;
        let w2a_u2i = sin2a * u2r + cos2a * u2i;
        let w2b_u3r = cos2b * u3r - sin2b * u3i;
        let w2b_u3i = sin2b * u3r + cos2b * u3i;

        // ── Write back ────────────────────────────────────────────────────────
        real[offset + p]         = u0r + w2a_u2r;
        imag[offset + p]         = u0i + w2a_u2i;
        real[offset + p + q * 2] = u0r - w2a_u2r;
        imag[offset + p + q * 2] = u0i - w2a_u2i;
        real[offset + p + q]     = u1r + w2b_u3r;
        imag[offset + p + q]     = u1i + w2b_u3i;
        real[offset + p + q * 3] = u1r - w2b_u3r;
        imag[offset + p + q * 3] = u1i - w2b_u3i;
    }
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
