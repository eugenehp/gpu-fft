pub mod fft;
pub mod ifft;
pub mod psd;
pub mod utils;

// Shared Cooley-Tukey butterfly kernel and helpers; not part of the public API.
pub(crate) mod butterfly;
// Work-in-progress precomputed-twiddle path; not yet wired into the public API.
#[allow(dead_code)]
pub(crate) mod twiddles;

// 1024 threads per workgroup saturates most desktop GPUs and is the maximum
// allowed by Metal / Vulkan / WebGPU on typical hardware.
pub(crate) const WORKGROUP_SIZE: u32 = 1024;

// Shared-memory tile for the inner (fused) butterfly kernel.
// Each workgroup loads TILE_SIZE elements into two SharedMemory<f32> arrays:
//   2 × TILE_SIZE × 4 bytes = 8 192 bytes < 16 384 byte WebGPU minimum.
// TILE_THREADS = TILE_SIZE / 2 = the number of threads per workgroup in the
// inner kernel (one thread per butterfly pair).
// TILE_BITS    = log₂(TILE_SIZE) = the number of stages that fit inside one tile.
pub(crate) const TILE_SIZE: usize = 1024;
pub(crate) const TILE_BITS: usize = 10; // log₂(TILE_SIZE) = log₂(1024)

#[cfg(feature = "wgpu")]
type Runtime = cubecl::wgpu::WgpuRuntime;

#[cfg(feature = "cuda")]
type Runtime = cubecl::cuda::CudaRuntime;

/// Computes the Cooley-Tukey radix-2 FFT of a real-valued signal.
///
/// Runs in **O(N log₂ N)** on the GPU using `log₂ N` butterfly-stage kernel
/// dispatches of N/2 threads each.
///
/// If `input.len()` is not a power of two the signal is zero-padded to the
/// next power of two before the transform.  Both returned vectors have length
/// `input.len().next_power_of_two()`.
///
/// # Example
///
/// ```no_run
/// use gpu_fft::fft;
/// let input = vec![0.0f32, 1.0, 0.0, 0.0];
/// let (real, imag) = fft(&input);
/// assert_eq!(real.len(), 4); // already a power of two
/// ```
#[must_use]
pub fn fft(input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    fft::fft::<Runtime>(&Default::default(), input)
}

/// Computes the Cooley-Tukey radix-2 FFT of a **batch** of real-valued signals
/// in a single GPU pass.
///
/// All signals are zero-padded to the next power-of-two of the **longest** signal
/// so they share a common length `n`.  The batch is processed with a 2-D kernel
/// dispatch — the Y-dimension selects the signal, and the X-dimension covers
/// butterfly pairs within a signal.
///
/// Returns one `(real, imag)` pair per input signal, each of length `n`.
///
/// # Example
///
/// ```no_run
/// use gpu_fft::fft_batch;
/// let signals = vec![
///     vec![1.0f32, 0.0, 0.0, 0.0], // impulse → all-ones spectrum
///     vec![1.0f32, 1.0, 1.0, 1.0], // DC      → [4, 0, 0, 0]
/// ];
/// let results = fft_batch(&signals);
/// assert_eq!(results.len(), 2);
/// ```
#[must_use]
pub fn fft_batch(signals: &[Vec<f32>]) -> Vec<(Vec<f32>, Vec<f32>)> {
    fft::fft_batch::<Runtime>(&Default::default(), signals)
}

/// Computes the Cooley-Tukey radix-2 IFFT of a complex spectrum.
///
/// Runs in **O(N log₂ N)** using `log₂ N` butterfly-stage kernels with
/// positive twiddle factors, followed by a CPU-side 1/N scaling pass.
///
/// Both slices must have the **same power-of-two length** — i.e. pass the
/// direct output of [`fft`] unchanged.
///
/// # Returns
///
/// A `Vec<f32>` of length `2 * N`:
/// - `output[0..N]` — reconstructed real signal
/// - `output[N..2N]` — reconstructed imaginary signal (≈ 0 for real inputs)
///
/// # Example
///
/// ```no_run
/// use gpu_fft::ifft;
/// let real = vec![0.0f32, 1.0, 0.0, 0.0];
/// let imag = vec![0.0f32, 0.0, 0.0, 0.0];
/// let output = ifft(&real, &imag);
/// let reconstructed = &output[..4]; // real part
/// ```
#[must_use]
pub fn ifft(input_real: &[f32], input_imag: &[f32]) -> Vec<f32> {
    ifft::ifft::<Runtime>(&Default::default(), input_real, input_imag)
}

/// Computes the Cooley-Tukey radix-2 IFFT for a **batch** of complex spectra
/// in a single GPU pass.
///
/// Each element of `signals` is a `(real, imag)` pair — the direct output of
/// [`fft_batch`].  All pairs must share the **same power-of-two length**.
///
/// Returns one `Vec<f32>` per input signal, each of length `2 * n`:
/// - `[0..n]`  — reconstructed real signal
/// - `[n..2n]` — reconstructed imaginary signal (≈ 0 for real-valued inputs)
///
/// # Example
///
/// ```no_run
/// use gpu_fft::{fft_batch, ifft_batch};
/// let signals = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
/// let spectra = fft_batch(&signals);
/// let recovered = ifft_batch(&spectra);
/// ```
#[must_use]
pub fn ifft_batch(signals: &[(Vec<f32>, Vec<f32>)]) -> Vec<Vec<f32>> {
    ifft::ifft_batch::<Runtime>(&Default::default(), signals)
}

