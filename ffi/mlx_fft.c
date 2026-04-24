#include "mlx_fft.h"
#include "mlx/c/array.h"
#include "mlx/c/fft.h"
#include "mlx/c/stream.h"

#include <complex.h>
#include <stdlib.h>
#include <string.h>

/// Shared helper: run a forward or inverse FFT via MLX.
static int mlx_fft_run(const float *input_real, const float *input_imag,
                       float *output_real, float *output_imag,
                       uint32_t n, int inverse) {
    // ── Interleave split-complex → packed complex64 ──────────────────────
    // complex64 = float _Complex = [re, im] pairs of float32.
    float *interleaved = (float *)malloc(2 * n * sizeof(float));
    if (!interleaved) return -1;
    for (uint32_t i = 0; i < n; i++) {
        interleaved[2 * i]     = input_real[i];
        interleaved[2 * i + 1] = input_imag[i];
    }

    // ── Create MLX array from interleaved data ───────────────────────────
    int shape[1] = { (int)n };
    mlx_array input_arr = mlx_array_new_data(
        interleaved, shape, 1, MLX_COMPLEX64);

    // ── Get GPU stream ───────────────────────────────────────────────────
    mlx_stream stream = mlx_default_gpu_stream_new();

    // ── Run FFT / IFFT ───────────────────────────────────────────────────
    mlx_array result_arr = mlx_array_new();
    int ret;
    if (inverse) {
        // BACKWARD norm = no scaling (matches our convention)
        ret = mlx_fft_ifft(&result_arr, input_arr, (int)n, 0,
                           MLX_FFT_NORM_BACKWARD, stream);
    } else {
        ret = mlx_fft_fft(&result_arr, input_arr, (int)n, 0,
                          MLX_FFT_NORM_BACKWARD, stream);
    }

    if (ret != 0) {
        mlx_array_free(input_arr);
        mlx_array_free(result_arr);
        mlx_stream_free(stream);
        free(interleaved);
        return -2;
    }

    // ── Evaluate (MLX is lazy) and synchronize ───────────────────────────
    mlx_array_eval(result_arr);
    mlx_synchronize(stream);

    // ── Extract complex output ───────────────────────────────────────────
    const float _Complex *out_ptr = mlx_array_data_complex64(result_arr);
    if (!out_ptr) {
        mlx_array_free(input_arr);
        mlx_array_free(result_arr);
        mlx_stream_free(stream);
        free(interleaved);
        return -3;
    }

    // Deinterleave complex → split
    const float *raw = (const float *)out_ptr;
    for (uint32_t i = 0; i < n; i++) {
        output_real[i] = raw[2 * i];
        output_imag[i] = raw[2 * i + 1];
    }

    // ── Cleanup ──────────────────────────────────────────────────────────
    mlx_array_free(input_arr);
    mlx_array_free(result_arr);
    mlx_stream_free(stream);
    free(interleaved);
    return 0;
}

int mlx_fft_forward(const float *input_real, const float *input_imag,
                    float *output_real, float *output_imag, uint32_t n) {
    return mlx_fft_run(input_real, input_imag, output_real, output_imag, n, 0);
}

int mlx_fft_inverse(const float *input_real, const float *input_imag,
                    float *output_real, float *output_imag, uint32_t n) {
    return mlx_fft_run(input_real, input_imag, output_real, output_imag, n, 1);
}
