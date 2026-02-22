use cubecl::prelude::*;

use crate::butterfly::{bit_reverse, butterfly_inner, butterfly_stage};
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
/// The remaining outer stages (one per stage) use `butterfly_stage` over global
/// memory.
///
/// | N        | Launches |
/// |----------|----------|
/// | ≤ 1 024  | 1        |
/// | 4 096    | 3        |
/// | 65 536   | 7        |
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

    // ── Outer stages: one launch per stage over global memory ─────────────────
    let outer_wg = ((n / 2) as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    for s in inner_stages..m {
        let hs = 1_usize << s;
        unsafe {
            butterfly_stage::launch::<f32, R>(
                &client,
                CubeCount::Static(outer_wg, 1, 1),
                CubeDim::new_1d(WORKGROUP_SIZE),
                ArrayArg::from_raw_parts::<f32>(&real_handle, n, 1),
                ArrayArg::from_raw_parts::<f32>(&imag_handle, n, 1),
                n,    // comptime
                hs,   // comptime — unique kernel per stage
                true, // comptime — forward FFT
            )
            .expect("FFT outer butterfly launch failed")
        };
    }

    let real_out = f32::from_bytes(&client.read_one(real_handle)).to_vec();
    let imag_out = f32::from_bytes(&client.read_one(imag_handle)).to_vec();

    (real_out, imag_out)
}
