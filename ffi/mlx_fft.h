#ifndef MLX_FFT_WRAPPER_H
#define MLX_FFT_WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Compute a forward complex FFT using MLX.
/// Input/output are split-complex (separate real and imag arrays of length n).
/// n must be a power of two >= 2.
/// Returns 0 on success, nonzero on error.
int mlx_fft_forward(const float *input_real, const float *input_imag,
                    float *output_real, float *output_imag, uint32_t n);

/// Compute an inverse complex FFT using MLX.
/// Same format as forward. Does NOT apply 1/N scaling (caller is responsible).
/// Returns 0 on success, nonzero on error.
int mlx_fft_inverse(const float *input_real, const float *input_imag,
                    float *output_real, float *output_imag, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* MLX_FFT_WRAPPER_H */
