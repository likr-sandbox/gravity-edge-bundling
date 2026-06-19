use rustfft::{FftPlanner, num_complex::Complex};

fn transpose(input: &[Complex<f32>], width: usize, height: usize) -> Vec<Complex<f32>> {
    let mut output = vec![Complex::new(0.0, 0.0); width * height];
    for y in 0..height {
        for x in 0..width {
            output[x * height + y] = input[y * width + x];
        }
    }
    output
}

pub fn fft2d(data: &mut [Complex<f32>], width: usize, height: usize) {
    let mut planner = FftPlanner::new();
    
    // 1. Perform 1D FFT on each row
    let fft_row = planner.plan_fft_forward(width);
    for y in 0..height {
        let row_start = y * width;
        fft_row.process(&mut data[row_start..row_start + width]);
    }

    // 2. Transpose (dimensions become height x width)
    let mut transposed = transpose(data, width, height);

    // 3. Perform 1D FFT on each column (now a row in the transposed matrix)
    let fft_col = planner.plan_fft_forward(height);
    for x in 0..width {
        let col_start = x * height;
        fft_col.process(&mut transposed[col_start..col_start + height]);
    }

    // 4. Transpose back (dimensions become width x height)
    let final_data = transpose(&transposed, height, width);
    data.copy_from_slice(&final_data);
}

pub fn ifft2d(data: &mut [Complex<f32>], width: usize, height: usize) {
    let mut planner = FftPlanner::new();
    
    // 1. Perform 1D IFFT on each row
    let fft_row = planner.plan_fft_inverse(width);
    for y in 0..height {
        let row_start = y * width;
        fft_row.process(&mut data[row_start..row_start + width]);
    }

    // 2. Transpose
    let mut transposed = transpose(data, width, height);

    // 3. Perform 1D IFFT on each column (now a row in the transposed matrix)
    let fft_col = planner.plan_fft_inverse(height);
    for x in 0..width {
        let col_start = x * height;
        fft_col.process(&mut transposed[col_start..col_start + height]);
    }

    // 4. Transpose back
    let final_data = transpose(&transposed, height, width);
    data.copy_from_slice(&final_data);

    // 5. Normalize
    let scale = (width * height) as f32;
    for val in data.iter_mut() {
        *val = *val / scale;
    }
}
