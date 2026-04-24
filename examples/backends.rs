//! Demonstrates runtime backend selection.
//!
//! Run with:
//!   cargo run --example backends --features wgpu        # WGPU only
//!   cargo run --example backends --features wgpu,mlx    # WGPU + MLX

use gpu_fft::{available_backends, fft_with, ifft_with};

fn main() {
    let backends = available_backends();
    println!("Available backends: {backends:?}\n");

    let signal = vec![1.0f32, 0.0, 0.0, 0.0]; // impulse

    for &backend in &backends {
        let (real, imag) = fft_with(&signal, backend);
        let recovered = ifft_with(&real, &imag, backend);

        let max_err = signal
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("{backend:?}");
        println!("  FFT real: {:?}", &real[..4]);
        println!("  FFT imag: {:?}", &imag[..4]);
        println!("  Round-trip max error: {max_err:.2e}");
        println!();
    }
}
