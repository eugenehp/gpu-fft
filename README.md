# GPU-FFT

This project demonstrates the use of the `gpu-fft` library in Rust to perform Fast Fourier Transform (FFT) and Inverse Fast Fourier Transform (IFFT) on a generated sine wave signal. The application calculates the dominant frequencies in the signal and prints them along with their power.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Features

- Generate a sine wave signal.
- Perform FFT to analyze the frequency components of the signal.
- Calculate the Power Spectral Density (PSD).
- Identify and print the dominant frequencies in the signal.
- Perform IFFT to reconstruct the original signal.

## Roadmap

- [x] Add twiddles algorithm
- [ ] Add radix2 optimization
- [ ] update to new CubeCL version

## Requirements

- Rust (1.84.1 or later)
- `gpu_fft` crate

## Installation

```base
cargo add gpu_fft -F wgpu
```

## Usage

To run the application, use the following command:

```bash
cargo run --example simple -F wgpu
```

The program will generate a sine wave with a specified frequency and sample rate, perform FFT, and print the dominant frequencies along with their power.

### Example Output

```
Input
  samples    : 1000
  sample rate: 200 Hz
  frequency  : 15 Hz
  first 5    : [0.0000, 0.4540, 0.8090, 0.9877, 0.9511]

FFT  completed in 54.48ms
Dominant frequencies (0 … 100 Hz):
     15.00 Hz  power       250.00

IFFT completed in 3.32ms
Round-trip max error : 4.19e-4  (limit 5·N·ε = 5.96e-4)  ✓
```

## Benchmarks

Run all benchmarks and save a timestamped Markdown report to `bench-results/`:

```shell
./scripts/bench.sh
```

Or run a single group / variant directly:

```shell
cargo bench --bench fft_bench -- fft/1024
```

Results are saved to `bench-results/latest.md` and archived under
`bench-results/archive/`. See the latest results there.

## License

This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.

## Copyright

© 2025-2026, Eugene Hauptmann
