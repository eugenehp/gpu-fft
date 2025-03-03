# GPU-FFT

This project demonstrates the use of the `gpu_fft` library in Rust to perform Fast Fourier Transform (FFT) and Inverse Fast Fourier Transform (IFFT) on a generated sine wave signal. The application calculates the dominant frequencies in the signal and prints them along with their power.

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

- [ ] Add twiddles algorithm

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
====================
    Input with frequency - 10 Hz
====================
1000000 [0.0, 0.06279052, 0.12533323, 0.18738133, 0.2486899, 0.309017, 0.36812457, 0.4257793, 0.4817537, 0.5358268]..
====================
    FFT 3.7933425s
====================
Frequency: 10.00 Hz, Power: 249999.38
Frequency: 958.99 Hz, Power: 122.58
Frequency: 990.00 Hz, Power: 247388.88
====================
    IFFT 4.030771s
====================
```

## License

This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.

## Copyright

©️ 2025, Eugene Hauptmann