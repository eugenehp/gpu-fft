use cubecl::prelude::*;

use crate::butterfly::{bit_reverse, butterfly_inner, butterfly_inner_batch, butterfly_stage, butterfly_stage_batch};
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

/// Computes the Cooley-Tukey radix-2 DIT IFFT for a **batch** of complex spectra
/// in a single GPU pass.
///
/// Each element of `signals` is a `(real, imag)` pair produced by [`fft_batch`]
/// (or by calling [`fft`][crate::fft::fft] repeatedly).  All pairs must share
/// the **same power-of-two length** — pass the direct output of
/// [`fft_batch`][crate::fft::fft_batch] unchanged.
///
/// Returns one `Vec<f32>` per input signal, each of length `2 * n`:
/// - `[0..n]`  — reconstructed real signal
/// - `[n..2n]` — reconstructed imaginary signal (≈ 0 for real-valued inputs)
///
/// ### Panics
///
/// Panics if any pair has mismatched lengths, or if the shared length is not a
/// power of two.  An empty batch returns an empty `Vec`.
///
/// # Example
///
/// ```ignore
/// use cubecl::wgpu::WgpuRuntime;
/// use gpu_fft::{fft::fft_batch, ifft::ifft_batch};
/// let signals = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
/// let spectra = fft_batch::<WgpuRuntime>(&Default::default(), &signals);
/// let pairs: Vec<_> = spectra.into_iter().collect();
/// let recovered = ifft_batch::<WgpuRuntime>(&Default::default(), &pairs);
/// ```
#[must_use]
pub fn ifft_batch<R: Runtime>(
    device: &R::Device,
    signals: &[(Vec<f32>, Vec<f32>)],
) -> Vec<Vec<f32>> {
    if signals.is_empty() {
        return Vec::new();
    }

    let batch_size = signals.len();

    // Validate: all pairs must have identical power-of-two lengths.
    let n = signals[0].0.len();
    for (b, (re, im)) in signals.iter().enumerate() {
        assert_eq!(
            re.len(), im.len(),
            "ifft_batch: signal {b}: real and imag slices have different lengths"
        );
        assert_eq!(
            re.len(), n,
            "ifft_batch: all signals must have the same length (expected {n}, got {})", re.len()
        );
    }
    assert!(
        n.is_power_of_two(),
        "ifft_batch: signal length {n} is not a power of two"
    );

    // Edge case: trivial inverse transform.
    if n <= 1 {
        return signals
            .iter()
            .map(|(re, im)| {
                let mut out = re.clone();
                out.extend_from_slice(im);
                out
            })
            .collect();
    }

    let m = n.ilog2() as usize;

    // ── Bit-reverse permute each signal on the CPU and pack flat ──────────────
    let mut real_flat = vec![0.0f32; batch_size * n];
    let mut imag_flat = vec![0.0f32; batch_size * n];

    for (b, (input_real, input_imag)) in signals.iter().enumerate() {
        let base = b * n;
        for i in 0..n {
            let j = bit_reverse(i, m as u32);
            real_flat[base + j] = input_real[i];
            imag_flat[base + j] = input_imag[i];
        }
    }

    let client  = R::client(device);
    let total   = batch_size * n;
    let real_handle = client.create_from_slice(f32::as_bytes(&real_flat));
    let imag_handle = client.create_from_slice(f32::as_bytes(&imag_flat));

    // ── Inner stages: one flat 1D dispatch covers all tiles in all signals ──────
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
            false,        // comptime — inverse FFT
        )
        .expect("IFFT batch inner (shared-memory) launch failed")
    };

    // ── Outer stages: one flat 1D dispatch per stage ──────────────────────────
    let total_pairs = batch_size * (n / 2);
    let outer_wg    = ((total_pairs as u32) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    for s in inner_stages..m {
        let hs = 1_usize << s;
        unsafe {
            butterfly_stage_batch::launch::<f32, R>(
                &client,
                CubeCount::Static(outer_wg, 1, 1),
                CubeDim::new_1d(WORKGROUP_SIZE),
                ArrayArg::from_raw_parts::<f32>(&real_handle, total, 1),
                ArrayArg::from_raw_parts::<f32>(&imag_handle, total, 1),
                n,          // comptime — per-signal length
                hs,         // comptime — half-stride for this stage
                batch_size, // comptime — number of signals in the batch
                false,      // comptime — inverse FFT
            )
            .expect("IFFT batch outer butterfly launch failed")
        };
    }

    // ── Read back and apply 1/N scaling on the CPU ────────────────────────────
    let mut real_out = f32::from_bytes(&client.read_one(real_handle)).to_vec();
    let mut imag_out = f32::from_bytes(&client.read_one(imag_handle)).to_vec();

    let scale = (n as f32).recip();
    for v in real_out.iter_mut() { *v *= scale; }
    for v in imag_out.iter_mut() { *v *= scale; }

    // ── Unpack: each output is [real[0..n] ++ imag[0..n]] ────────────────────
    (0..batch_size)
        .map(|b| {
            let start = b * n;
            let end   = start + n;
            let mut out = real_out[start..end].to_vec();
            out.extend_from_slice(&imag_out[start..end]);
            out
        })
        .collect()
}
