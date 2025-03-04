use std::time::Instant;

use gpu_fft::{fft, ifft, psd, twiddles, utils};

type Runtime = cubecl::wgpu::WgpuRuntime;

pub fn main() {
    let device = Default::default();
    // let input = vec![1.0, 0.0, 3.0, 0.0, 0.0];
    let sample_rate = 100.0;
    let frequency = 5.0;
    let threshold = 0.0; //100.0;

    let input: Vec<f32> = utils::generate_sine_wave(frequency, sample_rate, 10.0); // 1 million samples

    println!("====================");
    println!("\tInput with frequency - {frequency} Hz");
    println!("====================");
    println!("{} {:?}..", input.len(), &input[0..10]);

    let start_time = Instant::now();
    // let (real, imag) = fft::<Runtime>(&device, input);
    let (real, imag) = twiddles::fft::<Runtime>(&device, input);
    let elapsed_time = start_time.elapsed();

    println!("====================");
    println!("\tFFT {elapsed_time:?}");
    println!("====================");
    for (i, (real, imag)) in real.iter().zip(imag.clone()).enumerate() {
        println!("Output[{}]:\tReal: {}, Imag: {}", i, real, imag);
    }

    let spectrum = psd(real.clone(), imag.clone());
    let frequencies = utils::calculate_frequencies(spectrum.len(), sample_rate);

    let dominant_frequencies = utils::find_dominant_frequencies(spectrum, frequencies, threshold);

    // Print dominant frequencies
    for (freq, power) in dominant_frequencies {
        println!("Frequency: {:.2} Hz, Power: {:.2}", freq, power);
    }

    let n = real.len();
    let start_time = Instant::now();
    let output = ifft::<Runtime>(&device, real, imag);
    let elapsed_time = start_time.elapsed();

    println!("====================");
    println!("\tIFFT {elapsed_time:?}");
    println!("====================");
    // for i in 0..n {
    //     let real = output[i];
    //     let imag = output[i + n]; // Assuming output is interleaved
    //     println!("Output[{}]: Real: {}, Imag: {}", i, real, imag);
    // }
}
