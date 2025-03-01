use std::f32::consts::PI;

type Runtime = cubecl::wgpu::WgpuRuntime;

fn generate_sine_wave(frequency: f32, sample_rate: f32, duration: f32) -> Vec<f32> {
    let num_samples = (sample_rate * duration) as usize;
    let mut sine_wave = Vec::with_capacity(num_samples);

    for n in 0..num_samples {
        let sample = (2.0 * PI * frequency * (n as f32 / sample_rate)).sin();
        sine_wave.push(sample);
    }

    sine_wave
}

pub fn main() {
    // let input = vec![1.0, 0.0, 3.0, 0.0, 0.0];
    let input: Vec<f32> = generate_sine_wave(30.0, 500.0, 2.0); // 3.0 is out of memory

    println!("====================");
    println!("\tInput");
    println!("====================");
    println!("{} {:?}..", input.len(), &input[0..10]);

    let (real, imag) = gpu_fft::fft::<Runtime>(&Default::default(), input);

    println!("====================");
    println!("\tFFT");
    println!("====================");
    // Print the FFT output
    for (i, (real, imag)) in real.iter().zip(imag.clone()).enumerate() {
        println!("Output[{}]: Real: {}, Imag: {}", i, real, imag);
    }

    let n = real.len();
    let output = gpu_fft::ifft::<Runtime>(&Default::default(), real, imag);

    println!("====================");
    println!("\tIFFT");
    println!("====================");
    // Print the IFFT output
    for i in 0..n {
        let real = output[i];
        let imag = output[i + n]; // Assuming output is interleaved
        println!("Output[{}]: Real: {}, Imag: {}", i, real, imag);
    }
}