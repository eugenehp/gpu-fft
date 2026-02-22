use cubecl::prelude::*;

use crate::butterfly::{
    bit_reverse,
    butterfly_inner, butterfly_inner_batch,
    butterfly_stage, butterfly_stage_batch,
    butterfly_stage_radix4, butterfly_stage_radix4_batch,
};
use crate::{TILE_BITS, TILE_SIZE, WORKGROUP_SIZE};

/// Computes the Cooley-Tukey radix-2 DIT FFT of `input`.
///
/// If `input.len()` is not a power of two the signal is **zero-padded** to the
/// next power of two. Both returned vectors have length `input.len().next_power_of_two()`.
///
/// ### Launch strategy
///
/// All stages where `half_stride < TILE_SIZE / 2` are fused into a **single**
/// `butterfly_inner` dispatch using workgroup shared memory — eliminating the
/// per-stage kernel-launch overhead that dominates small-N performance.
/// The remaining outer stages use `butterfly_stage_radix4` (two radix-2 stages
/// per dispatch) where possible, falling back to `butterfly_stage` for a single
/// trailing stage when the outer-stage count is odd.
///
/// | N        | Inner | Outer        | Total |
/// |----------|------:|-------------:|------:|
/// | ≤ 1 024  | 1     | 0            | **1** |
/// | 4 096    | 1     | 1 (r4)       | **2** |
/// | 65 536   | 1     | 3 (r4)       | **4** |
///
/// # Example
///
/// ```ignore
/// use cubecl::wgpu::WgpuRuntime;
/// use gpu_fft::fft::fft;
/// let (real, imag) = fft::<WgpuRuntime>(&Default::default(), &[1.0f32, 0.0, 0.0, 0.0]);
/// ```
#[must_use]
pub fn fft<R: Runtime>(device: &R::Device, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n_orig = input.len();
    let n = n_orig.next_power_of_two();

    // Edge case: trivial transform for zero or single element.
    if n <= 1 {
        let mut real = vec![0.0f32; n];
        if n == 1 && n_orig == 1 {
            real[0] = input[0];
        }
        return (real, vec![0.0f32; n]);
    }

    let m = n.ilog2() as usize;

    // ── Bit-reverse permute the input on the CPU (O(N)) ───────────────────────
    let mut real = vec![0.0f32; n];
    for (i, &v) in input.iter().enumerate() {
        real[bit_reverse(i, m as u32)] = v;
    }
    let imag = vec![0.0f32; n];

    let client = R::client(device);
    let real_handle = client.create_from_slice(f32::as_bytes(&real));
    let imag_handle = client.create_from_slice(f32::as_bytes(&imag));

    // ── Inner stages: fused into one launch via shared memory ─────────────────
    // inner_stages = how many stages fit inside a TILE_SIZE-element workgroup tile.
    // tile         = actual tile size (≤ TILE_SIZE; equals N when N < TILE_SIZE).
    let inner_stages = m.min(TILE_BITS);
    let tile         = TILE_SIZE.min(n);     // comptime specialisation value
    let num_tiles    = (n / TILE_SIZE).max(1) as u32;
    let wg_threads   = (n / 2).min(TILE_SIZE / 2) as u32; // threads per workgroup

    unsafe {
        butterfly_inner::launch::<f32, R>(
            &client,
            CubeCount::Static(num_tiles, 1, 1),
            CubeDim::new_1d(wg_threads),
            ArrayArg::from_raw_parts::<f32>(&real_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&imag_handle, n, 1),
            tile,         // comptime
            inner_stages, // comptime
            true,         // comptime — forward FFT
        )
        .expect("FFT inner (shared-memory) launch failed")
    };

    // ── Outer stages: radix-4 pairs, then one radix-2 if the count is odd ────
    // Two consecutive radix-2 stages (strides q and 2q) are fused into a single
    // radix-4 dispatch, halving the number of kernel launches for large N.
    let outer_wg_r4 = ((n / 4) as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let outer_wg_r2 = ((n / 2) as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    let mut s = inner_stages;
    while s + 1 < m {
        let q = 1_usize << s;   // quarter-stride = half-stride of the lower stage
        unsafe {
            butterfly_stage_radix4::launch::<f32, R>(
                &client,
                CubeCount::Static(outer_wg_r4, 1, 1),
                CubeDim::new_1d(WORKGROUP_SIZE),
                ArrayArg::from_raw_parts::<f32>(&real_handle, n, 1),
                ArrayArg::from_raw_parts::<f32>(&imag_handle, n, 1),
                n,    // comptime
                q,    // comptime — specialises kernel for this stage pair
                true, // comptime — forward FFT
            )
            .expect("FFT outer radix-4 butterfly launch failed")
        };
        s += 2;
    }
    // Trailing radix-2 stage when (m − inner_stages) is odd.
    if s < m {
        let hs = 1_usize << s;
        unsafe {
            butterfly_stage::launch::<f32, R>(
                &client,
                CubeCount::Static(outer_wg_r2, 1, 1),
                CubeDim::new_1d(WORKGROUP_SIZE),
                ArrayArg::from_raw_parts::<f32>(&real_handle, n, 1),
                ArrayArg::from_raw_parts::<f32>(&imag_handle, n, 1),
                n,    // comptime
                hs,   // comptime
                true, // comptime — forward FFT
            )
            .expect("FFT outer radix-2 trailing butterfly launch failed")
        };
    }

    let real_out = f32::from_bytes(&client.read_one(real_handle)).to_vec();
    let imag_out = f32::from_bytes(&client.read_one(imag_handle)).to_vec();

    (real_out, imag_out)
}

/// Computes the Cooley-Tukey radix-2 DIT FFT for a **batch** of signals in a
/// single GPU pass.
///
/// All signals are zero-padded to the same length: the next power-of-two of the
/// **longest** signal.  Every other signal is padded to that same length so the
/// batch forms a rectangular `batch_size × n` matrix in GPU memory.
///
/// Returns one `(real, imag)` pair per input signal, each of length `n`.
///
/// ### Performance
///
/// All `batch_size` signals are processed simultaneously using a 2-D kernel
/// dispatch — the Y-dimension of the grid indexes the signal and the X-dimension
/// covers butterfly pairs within a signal.  This amortises kernel-launch overhead
/// over the entire batch.
///
/// ### Panics
///
/// Does not panic.  An empty batch returns an empty `Vec`.
///
/// # Example
///
/// ```ignore
/// use cubecl::wgpu::WgpuRuntime;
/// use gpu_fft::fft::fft_batch;
/// let signals = vec![vec![1.0f32, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
/// let results = fft_batch::<WgpuRuntime>(&Default::default(), &signals);
/// assert_eq!(results.len(), 2);
/// ```
#[must_use]
pub fn fft_batch<R: Runtime>(device: &R::Device, signals: &[Vec<f32>]) -> Vec<(Vec<f32>, Vec<f32>)> {
    if signals.is_empty() {
        return Vec::new();
    }

    let batch_size = signals.len();
    let max_len    = signals.iter().map(|s| s.len()).max().unwrap_or(0);

    // Edge case: all signals are empty or length 0/1.
    let n_raw = max_len.next_power_of_two().max(1);
    if n_raw <= 1 {
        return signals
            .iter()
            .map(|s| {
                let mut real = vec![0.0f32; n_raw];
                if n_raw == 1 && !s.is_empty() {
                    real[0] = s[0];
                }
                (real, vec![0.0f32; n_raw])
            })
            .collect();
    }

    let n = n_raw;
    let m = n.ilog2() as usize;

    // ── Bit-reverse permute each signal on the CPU and pack flat ──────────────
    let mut real_flat = vec![0.0f32; batch_size * n];
    let     imag_flat = vec![0.0f32; batch_size * n];

    for (b, signal) in signals.iter().enumerate() {
        let base = b * n;
        for (i, &v) in signal.iter().enumerate() {
            real_flat[base + bit_reverse(i, m as u32)] = v;
        }
    }

    let client  = R::client(device);
    let total   = batch_size * n;
    let real_handle = client.create_from_slice(f32::as_bytes(&real_flat));
    let imag_handle = client.create_from_slice(f32::as_bytes(&imag_flat));

    // ── Inner stages: one flat 1D dispatch covers all tiles in all signals ──────
    // Each workgroup = one tile (tile/2 threads). Total workgroups = tiles_per_signal * batch_size.
    let inner_stages     = m.min(TILE_BITS);
    let tile             = TILE_SIZE.min(n);
    let tiles_per_signal = (n / tile).max(1);
    let wg_count         = (tiles_per_signal * batch_size) as u32;
    let wg_threads       = (tile / 2) as u32;

    unsafe {
        butterfly_inner_batch::launch::<f32, R>(
            &client,
            CubeCount::Static(wg_count, 1, 1),
            CubeDim::new_1d(wg_threads),
            ArrayArg::from_raw_parts::<f32>(&real_handle, total, 1),
            ArrayArg::from_raw_parts::<f32>(&imag_handle, total, 1),
            n,            // comptime — per-signal length
            tile,         // comptime — tile size
            inner_stages, // comptime — stages fused per tile
            true,         // comptime — forward FFT
        )
        .expect("FFT batch inner (shared-memory) launch failed")
    };

    // ── Outer stages: radix-4 pairs, then one radix-2 if the count is odd ────
    let total_groups_r4 = batch_size * (n / 4);
    let total_pairs_r2  = batch_size * (n / 2);
    let outer_wg_r4 = ((total_groups_r4 as u32) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let outer_wg_r2 = ((total_pairs_r2  as u32) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    let mut s = inner_stages;
    while s + 1 < m {
        let q = 1_usize << s;
        unsafe {
            butterfly_stage_radix4_batch::launch::<f32, R>(
                &client,
                CubeCount::Static(outer_wg_r4, 1, 1),
                CubeDim::new_1d(WORKGROUP_SIZE),
                ArrayArg::from_raw_parts::<f32>(&real_handle, total, 1),
                ArrayArg::from_raw_parts::<f32>(&imag_handle, total, 1),
                n,          // comptime
                q,          // comptime
                batch_size, // comptime
                true,       // comptime — forward FFT
            )
            .expect("FFT batch outer radix-4 butterfly launch failed")
        };
        s += 2;
    }
    if s < m {
        let hs = 1_usize << s;
        unsafe {
            butterfly_stage_batch::launch::<f32, R>(
                &client,
                CubeCount::Static(outer_wg_r2, 1, 1),
                CubeDim::new_1d(WORKGROUP_SIZE),
                ArrayArg::from_raw_parts::<f32>(&real_handle, total, 1),
                ArrayArg::from_raw_parts::<f32>(&imag_handle, total, 1),
                n,          // comptime
                hs,         // comptime
                batch_size, // comptime
                true,       // comptime — forward FFT
            )
            .expect("FFT batch outer radix-2 trailing butterfly launch failed")
        };
    }

    let real_out = f32::from_bytes(&client.read_one(real_handle)).to_vec();
    let imag_out = f32::from_bytes(&client.read_one(imag_handle)).to_vec();

    // ── Unpack flat buffer into per-signal pairs ──────────────────────────────
    (0..batch_size)
        .map(|b| {
            let start = b * n;
            let end   = start + n;
            (real_out[start..end].to_vec(), imag_out[start..end].to_vec())
        })
        .collect()
}
