use cubecl::prelude::*;
use std::f32::consts::PI;

#[cube(launch_unchecked)]
fn ifft_kernel<F: Float>(input_real: &Array<Line<F>>, input_imag: &Array<Line<F>>, output: &mut Array<Line<F>>, #[comptime] n: u32) {
    let idx = ABSOLUTE_POS;
    if idx < n {
        let mut sum_real = Line::<F>::new(F::new(0.0));
        let mut sum_imag = Line::<F>::new(F::new(0.0));

        for k in 0..n {
            let angle = F::new(2.0) * F::new(PI) * F::cast_from(k) * F::cast_from(idx) / F::cast_from(n);
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

pub fn ifft<R: Runtime>(device: &R::Device, input_real: Vec<f32>, input_imag: Vec<f32>) -> Vec<f32>{
    let client = R::client(device);
    let n = input_real.len();
    
    let real_handle = client.create(f32::as_bytes(&input_real));
    let imag_handle = client.create(f32::as_bytes(&input_imag));
    let output_handle = client.empty(n * 2 * core::mem::size_of::<f32>()); // Assuming output is interleaved

    unsafe {
        ifft_kernel::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(n as u32, 1, 1),
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