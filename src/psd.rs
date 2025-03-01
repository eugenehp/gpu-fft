pub fn psd(real: Vec<f32>, imag: Vec<f32>) -> Vec<f32> {
    let n = real.len();
    let mut psd = Vec::with_capacity(n);

    for i in 0..n {
        // Calculate the magnitude
        let magnitude = (real[i] * real[i] + imag[i] * imag[i]).sqrt();
        // Calculate the power
        let power = magnitude * magnitude;
        // Normalize the power (optional, depending on your needs)
        psd.push(power / n as f32); // Normalization by the number of points
    }

    psd
}
