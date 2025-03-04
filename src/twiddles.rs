use cubecl::prelude::*;
use std::f32::consts::PI;

use crate::WORKGROUP_SIZE;

#[cube(launch_unchecked)]
fn precompute_twiddles<F: Float>(twiddles: &mut Array<Line<F>>, #[comptime] n: u32) {
    let idx = ABSOLUTE_POS;
    if idx < 2 * n {
        // Precompute twiddles for the given index
        let angle_factor = F::new(-2.0) * F::new(PI) * F::cast_from(idx) / F::cast_from(n);
        for k in 0..n {
            let angle = angle_factor * F::cast_from(k);
            let (cos_angle, sin_angle) = (F::cos(angle), F::sin(angle));
            twiddles[k * 2] = Line::new(cos_angle); // Real part
            twiddles[k * 2 + 1] = Line::new(sin_angle); // Imaginary part
        }
    }
}

#[cube(launch_unchecked)]
fn fft_kernel<F: Float>(
    input: &Array<Line<F>>, // array of real floats, there are no imaginary floats here
    twiddles: &Array<Line<F>>, // twiddles array is now correctly sized
    real_output: &mut Array<Line<F>>,
    imag_output: &mut Array<Line<F>>,
    #[comptime] n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < 2 * n {
        let mut real = Line::<F>::new(F::new(0.0));
        let mut imag = Line::<F>::new(F::new(0.0));

        // Ensure we are accessing the correct index for input
        let input_real = input[idx * 2]; // Access the real part of the input

        for k in 0..n {
            let twiddle_real = twiddles[k * 2]; // Real part of the twiddle factor
            let twiddle_imag = twiddles[k * 2 + 1]; // Imaginary part of the twiddle factor

            // Accumulate the results
            real += input_real * twiddle_real;
            imag += input_real * twiddle_imag;
        }

        // Store the results in the output arrays
        real_output[idx] = real;
        imag_output[idx] = imag;
    }
}

pub fn fft<R: Runtime>(device: &R::Device, input: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
    let client = R::client(device);
    let n = input.len();

    let input_handle = client.create(f32::as_bytes(&input));
    let real_handle = client.empty(n * core::mem::size_of::<f32>());
    let imag_handle = client.empty(n * core::mem::size_of::<f32>());
    let twiddles_handle = client.empty(n * core::mem::size_of::<f32>());

    let num_workgroups = (n as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    unsafe {
        precompute_twiddles::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(num_workgroups, 1, 1),
            CubeDim::new(WORKGROUP_SIZE, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&twiddles_handle, (n * 2) as usize, 1),
            n as u32,
        )
    };

    let twiddles_bytes = client.read_one(twiddles_handle.clone().binding());
    let twiddles = f32::from_bytes(&twiddles_bytes);
    println!("twiddles[{}] - {:#?}", twiddles.len(), &twiddles[0..10]);
    // println!("twiddles[{}] - {:?}", twiddles.len(), twiddles);
    // (vec![], vec![])

    unsafe {
        fft_kernel::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(num_workgroups, 1, 1),
            CubeDim::new(WORKGROUP_SIZE, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&twiddles_handle, n as usize, 1),
            ArrayArg::from_raw_parts::<f32>(&real_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&imag_handle, n, 1),
            n as u32,
        )
    };

    let real_bytes = client.read_one(real_handle.binding());
    let real = f32::from_bytes(&real_bytes);

    let imag_bytes = client.read_one(imag_handle.binding());
    let imag = f32::from_bytes(&imag_bytes);

    println!(
        "real {:?}..{:?}",
        &real[0..10],
        &real[real.len() - 10..real.len() - 1]
    );
    println!(
        "imag {:?}..{:?}",
        &imag[0..10],
        &imag[imag.len() - 10..imag.len() - 1]
    );

    (real.into(), imag.into())
}
