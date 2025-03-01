use cubecl::prelude::*;
use std::f32::consts::PI;

#[cube(launch_unchecked)]
fn fft_kernel<F: Float>(input: &Array<Line<F>>, real_output: &mut Array<Line<F>>, imag_output: &mut Array<Line<F>>, #[comptime] n: u32) {
    let idx = ABSOLUTE_POS;
    if idx < n {
        let mut real = Line::<F>::new(F::new(0.0));
        let mut imag = Line::<F>::new(F::new(0.0));
        

        for k in 0..n {
            let angle = F::new(-2.0) * F::new(PI) * F::cast_from(k) * F::cast_from(idx) / F::cast_from(n);
            
            let (cos_angle, sin_angle) = (F::cos(angle), F::sin(angle));
            real += input[k] * Line::new(cos_angle);
            imag += input[k] * Line::new(sin_angle);
        }

        real_output[idx] = real;
        imag_output[idx] = imag;
    }
}

pub fn launch<R: Runtime>(device: &R::Device, input:Vec<f32>) {
    let client = R::client(device);
    let n = input.len();
    
    let input_handle = client.create(f32::as_bytes(&input));

    let real_handle = client.empty(n * core::mem::size_of::<f32>());
    let imag_handle = client.empty(n * core::mem::size_of::<f32>());

    unsafe {
        fft_kernel::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(n as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&real_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&imag_handle, n, 1),
            n as u32,
        )
    };

    let real_bytes = client.read_one(real_handle.binding());
    let real = f32::from_bytes(&real_bytes);
    
    let imag_bytes = client.read_one(imag_handle.binding());
    let imag = f32::from_bytes(&imag_bytes);

    println!("Runtime: {:?}", R::name());

    // Print the FFT output
    for (i, (real, imag)) in real.iter().zip(imag).enumerate() {
        println!("Output[{}]: Real: {}, Imag: {}", i, real, imag);
    }
}
