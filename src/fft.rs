use cubecl::prelude::*;
use std::f32::consts::PI;

use crate::WORKGROUP_SIZE;

/// Performs a Fast Fourier Transform (FFT) on the input data.
///
/// This kernel computes the FFT of a given input array of complex numbers represented as
/// separate real and imaginary parts. The FFT is computed using the Cooley-Tukey algorithm,
/// which is efficient for large datasets.
///
/// # Parameters
///
/// - `input`: An array of complex numbers represented as lines of type `Line<F>`, where `F`
///   is a floating-point type. The input array should contain `n` complex numbers.
/// - `real_output`: A mutable reference to an array of lines where the real parts of the
///   FFT output will be stored.
/// - `imag_output`: A mutable reference to an array of lines where the imaginary parts of
///   the FFT output will be stored.
/// - `n`: The number of complex samples in the input array. This value is provided at compile-time.
///
/// # Safety
///
/// This function is marked as `unsafe` because it performs raw pointer operations and assumes
/// that the input and output arrays are correctly sized and aligned. The caller must ensure
/// that the input data is valid and that the output arrays have sufficient space to store
/// the results.
///
/// # Example
///
/// ```ignore
/// use cubecl::wgpu::WgpuRuntime;
/// use gpu_fft::fft::fft;
/// let device = Default::default();
/// let input = vec![1.0f32, 0.0, 0.0, 0.0];
/// let (real, imag) = fft::<WgpuRuntime>(&device, input);
/// ```
///
/// # Returns
///
/// This function does not return a value directly. Instead, it populates the `output` array
/// with the real and imaginary parts of the FFT result interleaved.
#[cube(launch)]
fn fft_kernel<F: Float>(input: &Array<Line<F>>, output: &mut Array<Line<F>>, #[comptime] n: usize) {
    let idx = ABSOLUTE_POS;
    if idx < n {
        let mut real = Line::<F>::new(F::new(0.0));
        let mut imag = Line::<F>::new(F::new(0.0));

        // Precompute the angle increment
        let angle_increment = -2.0 * PI / n as f32;

        // #[unroll(true)]
        for k in 0..n {
            let angle = F::cast_from(angle_increment) * F::cast_from(k) * F::cast_from(idx);
            let (cos_angle, sin_angle) = (F::cos(angle), F::sin(angle));

            // Combine the multiplication and addition
            real += input[k] * Line::new(cos_angle);
            imag += input[k] * Line::new(sin_angle);
        }

        // Store the real and imaginary parts in an interleaved manner
        output[idx * 2] = Line::new(F::cast_from(real)); // Real part
        output[idx * 2 + 1] = Line::new(F::cast_from(imag)); // Imaginary part
    }
}

/// Computes the Fast Fourier Transform (FFT) of a vector of f32 input data.
///
/// This function initializes the FFT computation on the provided input vector, launching
/// the FFT kernel to perform the transformation. The input data is expected to be in the
/// form of real numbers, which are treated as complex numbers with zero imaginary parts.
///
/// # Parameters
///
/// - `device`: A reference to the device on which the FFT computation will be performed.
/// - `input`: A vector of `f32` values representing the real parts of the input data.
///
/// # Returns
///
/// A tuple containing two vectors:
/// - A vector of `f32` values representing the real parts of the FFT output.
/// - A vector of `f32` values representing the imaginary parts of the FFT output.
///
/// # Example
///
/// ```ignore
/// use cubecl::wgpu::WgpuRuntime;
/// use gpu_fft::fft::fft;
/// let device = Default::default();
/// let input = vec![1.0f32, 0.0, 0.0, 0.0];
/// let (real, imag) = fft::<WgpuRuntime>(&device, input);
/// ```
///
/// # Safety
///
/// This function uses unsafe operations to interact with the underlying runtime and device.
/// The caller must ensure that the input data is valid and that the device is properly set up
/// for computation.
pub fn fft<R: Runtime>(device: &R::Device, input: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
    let client = R::client(device);
    let n = input.len();

    let input_handle = client.create_from_slice(f32::as_bytes(&input));
    let output_handle = client.empty(n * 2 * core::mem::size_of::<f32>()); // Adjust for interleaved output

    let num_workgroups = (n as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    unsafe {
        fft_kernel::launch::<f32, R>(
            &client,
            CubeCount::Static(num_workgroups, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            ArrayArg::from_raw_parts::<f32>(&input_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, n * 2, 1), // Adjust for interleaved output
            n,
        )
        .expect("FFT kernel launch failed")
    };

    let output_bytes = client.read_one(output_handle);
    let output = f32::from_bytes(&output_bytes);

    // Split the interleaved output into real and imaginary parts
    let real: Vec<f32> = output.iter().step_by(2).cloned().collect();
    let imag: Vec<f32> = output.iter().skip(1).step_by(2).cloned().collect();

    // println!(
    //     "real {:?}..{:?}",
    //     &real[0..10],
    //     &real[real.len() - 10..real.len() - 1]
    // );
    // println!(
    //     "imag {:?}..{:?}",
    //     &imag[0..10],
    //     &imag[imag.len() - 10..imag.len() - 1]
    // );

    (real, imag)
}
