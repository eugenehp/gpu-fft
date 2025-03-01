use cubecl::prelude::*;
use std::f32::consts::PI;

use crate::WORKGROUP_SIZE;

/// Performs an Inverse Fast Fourier Transform (IFFT) on the input data.
///
/// This kernel computes the IFFT of a given input array of complex numbers represented as
/// separate real and imaginary parts. The IFFT is computed using the inverse of the Cooley-Tukey
/// algorithm, which is efficient for large datasets.
///
/// # Parameters
///
/// - `input_real`: An array of real parts of complex numbers represented as lines of type
///   `Line<F>`, where `F` is a floating-point type. The input array should contain `n` real
///   components.
/// - `input_imag`: An array of imaginary parts of complex numbers represented as lines of type
///   `Line<F>`. The input array should contain `n` imaginary components.
/// - `output`: A mutable reference to an array of lines where the output of the IFFT will be
///   stored. The output will contain interleaved real and imaginary parts.
/// - `n`: The number of complex samples in the input arrays. This value is provided at compile-time.
///
/// # Safety
///
/// This function is marked as `unsafe` because it performs raw pointer operations and assumes
/// that the input and output arrays are correctly sized and aligned. The caller must ensure
/// that the input data is valid and that the output array has sufficient space to store
/// the results.
///
/// # Example
///
/// ```rust
/// let input_real = vec![1.0, 0.0, 0.0, 0.0]; // Example real input
/// let input_imag = vec![0.0, 0.0, 0.0, 0.0]; // Example imaginary input
/// let output = ifft::<YourRuntimeType>(device, input_real, input_imag);
/// ```
///
/// # Returns
///
/// This function does not return a value directly. Instead, it populates the `output` array
/// with the interleaved real and imaginary parts of the IFFT result.
#[cube(launch_unchecked)]
fn ifft_kernel<F: Float>(
    input_real: &Array<Line<F>>,
    input_imag: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < n {
        let mut sum_real = Line::<F>::new(F::new(0.0));
        let mut sum_imag = Line::<F>::new(F::new(0.0));

        for k in 0..n {
            let angle =
                F::new(2.0) * F::new(PI) * F::cast_from(k) * F::cast_from(idx) / F::cast_from(n);
            let (cos_angle, sin_angle) = (F::cos(angle), F::sin(angle));
            sum_real += input_real[k] * Line::new(cos_angle) - input_imag[k] * Line::new(sin_angle);
            sum_imag += input_real[k] * Line::new(sin_angle) + input_imag[k] * Line::new(cos_angle);
        }

        let n_line = Line::<F>::new(F::cast_from(n));

        // Scale the output by 1/n
        output[idx] = sum_real / n_line;
        output[idx + n] = sum_imag / n_line;
    }
}

/// Computes the Inverse Fast Fourier Transform (IFFT) of a vector of real and imaginary input data.
///
/// This function initializes the IFFT computation on the provided input vectors, launching
/// the IFFT kernel to perform the transformation. The input data is expected to be in the
/// form of separate real and imaginary components of complex numbers.
///
/// # Parameters
///
/// - `device`: A reference to the device on which the IFFT computation will be performed.
/// - `input_real`: A vector of `f32` values representing the real parts of the input data.
/// - `input_imag`: A vector of `f32` values representing the imaginary parts of the input data.
///
/// # Returns
///
/// A vector of `f32` values representing the interleaved output of the IFFT, which contains
/// both the real and imaginary parts of the result.
///
/// # Example
///
/// ```rust
/// let input_real = vec![1.0, 0.0, 0.0, 0.0]; // Example real input
/// let input_imag = vec![0.0, 0.0, 0.0, 0.0]; // Example imaginary input
/// let output = ifft::<YourRuntimeType>(device, input_real, input_imag);
/// ```
///
/// # Safety
///
/// This function uses unsafe operations to interact with the underlying runtime and device.
/// The caller must ensure that the input data is valid and that the device is properly set up
/// for computation.
pub fn ifft<R: Runtime>(
    device: &R::Device,
    input_real: Vec<f32>,
    input_imag: Vec<f32>,
) -> Vec<f32> {
    let client = R::client(device);
    let n = input_real.len();

    let real_handle = client.create(f32::as_bytes(&input_real));
    let imag_handle = client.create(f32::as_bytes(&input_imag));
    let output_handle = client.empty(n * 2 * core::mem::size_of::<f32>()); // Assuming output is interleaved

    let num_workgroups = (n as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    unsafe {
        ifft_kernel::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(num_workgroups, 1, 1),
            CubeDim::new(WORKGROUP_SIZE, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&real_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&imag_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, n * 2, 1),
            n as u32,
        )
    };

    let output_bytes = client.read_one(output_handle.binding());
    let output = f32::from_bytes(&output_bytes);

    output.into()
}
