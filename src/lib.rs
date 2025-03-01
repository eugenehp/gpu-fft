mod fft;
mod ifft;
mod psd;
pub mod utils;

pub use fft::*;
pub use ifft::*;
pub use psd::*;

// The general advice for WebGPU is to choose a workgroup size of 64
// Common sizes are 32, 64, 128, 256, or 512 threads per workgroup.
// Apple Metal supports a maximum workgroup size of 1024 threads.
pub(crate) const WORKGROUP_SIZE: u32 = 1024;
