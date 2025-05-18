extern "C" __global__ void k1_stats_kernel(
    float* pixel_data,      // [num_pixels, num_frames, channels]
    float* means,           // [num_pixels, channels]
    float* variances,       // [num_pixels, channels]
    float* bics,            // [num_pixels]
    int num_pixels,
    int num_frames,
    int channels
) {
    // Get pixel index (each thread processes one pixel)
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= num_pixels) return;
    
    // Calculate mean for this pixel
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        // Sum across all frames for this pixel and channel
        for (int f = 0; f < num_frames; f++) {
            sum += pixel_data[(pixel_idx * num_frames + f) * channels + c];
        }
        
        // Store mean
        means[pixel_idx * channels + c] = sum / num_frames;
    }
    
    // Calculate variance for this pixel
    for (int c = 0; c < channels; c++) {
        float mean_val = means[pixel_idx * channels + c];
        float sum_sq_diff = 0.0f;
        
        // Sum squared differences from mean
        for (int f = 0; f < num_frames; f++) {
            float diff = pixel_data[(pixel_idx * num_frames + f) * channels + c] - mean_val;
            sum_sq_diff += diff * diff;
        }
        
        // Store variance with minimum threshold
        float var = sum_sq_diff / num_frames;
        variances[pixel_idx * channels + c] = (var > 0.05f) ? var : 0.05f;
    }
    
    // Calculate BIC score
    float mean_variance = 0.0f;
    for (int c = 0; c < channels; c++) {
        mean_variance += variances[pixel_idx * channels + c];
    }
    mean_variance /= channels;
    
    // Ensure minimum variance
    if (mean_variance <= 0.0f) {
        mean_variance = 1e-6f;
    }
    
    // Log-likelihood calculation
    float log_likelihood = -0.5f * num_frames * channels * logf(2.0f * 3.1415926535f * mean_variance);
    log_likelihood -= 0.5f * num_frames * channels;
    
    // BIC score calculation
    int n_params = channels + 1; // Mean + variance
    bics[pixel_idx] = log_likelihood - 0.5f * n_params * logf(num_frames);
}