use criterion::{black_box, criterion_group, criterion_main, Criterion};

type Runtime = cubecl::wgpu::WgpuRuntime;

fn benchmark_fft(c: &mut Criterion) {
    let input_size = 1_000;
    let input: Vec<f32> = (0..input_size).map(|x| (x as f32) * 0.1).collect();

    c.bench_function("fft", |b| {
        b.iter(|| {
            let device = Default::default();
            gpu_fft::fft::fft::<Runtime>(&device, black_box(input.clone()))
        })
    });
}

criterion_group!(benches, benchmark_fft);
criterion_main!(benches);
