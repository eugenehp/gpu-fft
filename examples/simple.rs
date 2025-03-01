pub fn main() {
    let input = vec![1.0, 0.0, 0.0, 0.0]; // Example input for FFT
    gpu_fft::launch::<cubecl::wgpu::WgpuRuntime>(&Default::default(), input);
}