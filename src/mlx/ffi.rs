extern "C" {
    pub fn mlx_fft_forward(
        input_real: *const f32,
        input_imag: *const f32,
        output_real: *mut f32,
        output_imag: *mut f32,
        n: u32,
    ) -> i32;

    pub fn mlx_fft_inverse(
        input_real: *const f32,
        input_imag: *const f32,
        output_real: *mut f32,
        output_imag: *mut f32,
        n: u32,
    ) -> i32;
}
