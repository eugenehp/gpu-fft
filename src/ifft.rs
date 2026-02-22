use cubecl::prelude::*;

use crate::butterfly::{bit_reverse, butterfly_inner, butterfly_stage};
use crate::{TILE_BITS, TILE_SIZE, WORKGROUP_SIZE};

/// Computes the Cooley-Tukey radix-2 DIT IFFT of `(input_real, input_imag)`.
///
/// Both slices must have the **same power-of-two length** (the direct output of
/// [`fft`][crate::fft::fft]). Uses the same inner/outer launch strategy as `fft`:
/// inner stages are fused into a single shared-memory dispatch; outer stages use
/// one global-memory dispatch each.  After the butterflies, a CPU-side 1/N divide
/// is applied.
///
/// # Returns
///
/// `Vec<f32>` of length `2 * N`:
/// - `[0..N]`  — reconstructed real signal
/// - `[N..2N]` — reconstructed imaginary signal (≈ 0 for real-valued inputs)
///
/// # Panics
///
/// Panics if the slice lengths differ or are not a power of two.
///
/// # Example
///
/// ```ignore
/// use cubecl::wgpu::WgpuRuntime;
/// use gpu_fft::ifft::ifft;
/// let real = vec![1.0f32, 0.0, 0.0, 0.0];
/// let imag = vec![0.0f32, 0.0, 0.0, 0.0];
/// let output = ifft::<WgpuRuntime>(&Default::default(), &real, &imag);
/// ```
#[must_use]
pub fn ifft<R: Runtime>(
    device: &R::Device,
    input_real: &[f32],
    input_imag: &[f32],
) -> Vec<f32> {
    assert_eq!(
        input_real.len(),
        input_imag.len(),
        "ifft: real and imag slices must have the same length"
    );
    let n = input_real.len();
    assert!(
        n.is_power_of_two(),
        "ifft: input length {n} is not a power of two (pass the direct output of fft)"
    );

    // Edge case: trivial inverse transform.
    if n <= 1 {
        let mut out = input_real.to_vec();
        out.extend_from_slice(input_imag);
        return out;
    }

    let m = n.ilog2() as usize;

    // ── Bit-reverse permute the input on the CPU (O(N)) ───────────────────────
    let mut real = vec![0.0f32; n];
    let mut imag = vec![0.0f32; n];
    for i in 0..n {
        let j = bit_reverse(i, m as u32);
        real[j] = input_real[i];
        imag[j] = input_imag[i];
    }

    let client = R::client(device);
    let real_handle = client.create_from_slice(f32::as_bytes(&real));
    let imag_handle = client.create_from_slice(f32::as_bytes(&imag));

    // ── Inner stages: fused into one launch via shared memory ─────────────────
    let inner_stages = m.min(TILE_BITS);
    let tile         = TILE_SIZE.min(n);
    let num_tiles    = (n / TILE_SIZE).max(1) as u32;
    let wg_threads   = (n / 2).min(TILE_SIZE / 2) as u32;

    unsafe {
        butterfly_inner::launch::<f32, R>(
            &client,
            CubeCount::Static(num_tiles, 1, 1),
            CubeDim::new_1d(wg_threads),
            ArrayArg::from_raw_parts::<f32>(&real_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&imag_handle, n, 1),
            tile,         // comptime
            inner_stages, // comptime
            false,        // comptime — inverse FFT (positive twiddle)
        )
        .expect("IFFT inner (shared-memory) launch failed")
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
                n,     // comptime
                hs,    // comptime — unique kernel per stage
                false, // comptime — inverse FFT
            )
            .expect("IFFT outer butterfly launch failed")
        };
    }

    // ── Read back and apply 1/N scaling on the CPU (O(N)) ────────────────────
    let mut real_out = f32::from_bytes(&client.read_one(real_handle)).to_vec();
    let mut imag_out = f32::from_bytes(&client.read_one(imag_handle)).to_vec();

    let scale = (n as f32).recip();
    for v in &mut real_out {
        *v *= scale;
    }
    for v in &mut imag_out {
        *v *= scale;
    }

    // Return [real[0..n] ++ imag[0..n]] — same layout as before.
    real_out.extend_from_slice(&imag_out);
    real_out
}
