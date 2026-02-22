pub mod fft;
pub mod ifft;
pub mod psd;
// Work-in-progress precomputed-twiddle path; not yet wired into the public API.
#[allow(dead_code)]
pub(crate) mod twiddles;
pub mod utils;

// The general advice for WebGPU is to choose a workgroup size of 64
// Common sizes are 32, 64, 128, 256, or 512 threads per workgroup.
// Apple Metal supports a maximum workgroup size of 1024 threads.
pub(crate) const WORKGROUP_SIZE: u32 = 1024;

#[cfg(feature = "wgpu")]
type Runtime = cubecl::wgpu::WgpuRuntime;

#[cfg(feature = "cuda")]
type Runtime = cubecl::cuda::CudaRuntime;

/// Computes the Fast Fourier Transform (FFT) of the given input vector.
///
/// This function takes a vector of real numbers as input and returns a tuple
/// containing two vectors: the real and imaginary parts of the FFT result.
///
/// # Parameters
///
/// - `input`: A vector of `f32` representing the input signal in the time domain.
///
/// # Returns
///
/// A tuple containing two vectors:
/// - A vector of `f32` representing the real part of the FFT output.
/// - A vector of `f32` representing the imaginary part of the FFT output.
///
/// # Example
///
/// ```no_run
/// use gpu_fft::fft;
/// let input = vec![0.0f32, 1.0, 0.0, 0.0];
/// let (real, imag) = fft(&input);
/// ```
#[must_use]
pub fn fft(input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    fft::fft::<Runtime>(&Default::default(), input)
}

/// Computes the Inverse Fast Fourier Transform (IFFT) of the given real and imaginary parts.
///
/// This function takes the real and imaginary parts of a frequency domain signal
/// and returns the corresponding time domain signal as a vector of real numbers.
///
/// # Parameters
///
/// - `input_real`: A vector of `f32` representing the real part of the frequency domain signal.
/// - `input_imag`: A vector of `f32` representing the imaginary part of the frequency domain signal.
///
/// # Returns
///
/// A vector of `f32` representing the reconstructed time domain signal.
///
/// # Example
///
/// ```no_run
/// use gpu_fft::ifft;
/// let real = vec![0.0f32, 1.0, 0.0, 0.0];
/// let imag = vec![0.0f32, 0.0, 0.0, 0.0];
/// let time_domain = ifft(&real, &imag);
/// ```
#[must_use]
pub fn ifft(input_real: &[f32], input_imag: &[f32]) -> Vec<f32> {
    ifft::ifft::<Runtime>(&Default::default(), input_real, input_imag)
}

