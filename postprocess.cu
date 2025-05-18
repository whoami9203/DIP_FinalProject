extern "C" __global__ void postprocess_kernel(
    float* centers,         // [num_pixels, k, channels]
    int* labels,            // [num_pixels, num_frames]
    int* largest_ids,       // [num_pixels]
    float* pixel_data,      // [num_pixels, num_frames, channels]
    float* best_centers,    // [num_pixels, channels]
    float* covariances,     // [num_pixels, channels]
    int num_pixels, 
    int num_frames, 
    int channels,
    int k
) {
    // Each thread processes one pixel
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= num_pixels) return;
    
    // Get the largest cluster ID for this pixel
    int largest_id = largest_ids[pixel_idx];
    
    // Copy the best center (from largest cluster)
    for (int c = 0; c < channels; c++) {
        best_centers[pixel_idx * channels + c] = 
            centers[(pixel_idx * k + largest_id) * channels + c];
    }
    
    // Calculate covariance for the largest cluster
    // First, count points in the largest cluster
    int cluster_size = 0;
    for (int f = 0; f < num_frames; f++) {
        if (labels[pixel_idx * num_frames + f] == largest_id) {
            cluster_size++;
        }
    }
    
    // If cluster has points, calculate covariance
    if (cluster_size > 0) {
        // Calculate mean of points in this cluster
        float cluster_mean[3] = {0.0f, 0.0f, 0.0f}; // Assuming channels <= 3
        
        // First pass: calculate mean
        // The best center is the mean
        for (int c = 0; c < channels; c++) {
            cluster_mean[c] = centers[(pixel_idx * k + largest_id) * channels + c];
        }
        
        // Second pass: calculate variance
        float variance[3] = {0.0f, 0.0f, 0.0f}; // Assuming channels <= 3
        
        for (int f = 0; f < num_frames; f++) {
            if (labels[pixel_idx * num_frames + f] == largest_id) {
                for (int c = 0; c < channels; c++) {
                    float diff = pixel_data[pixel_idx * num_frames * channels + f * channels + c] - cluster_mean[c];
                    variance[c] += diff * diff;
                }
            }
        }
        
        // Store covariance with minimum value of 0.1
        for (int c = 0; c < channels; c++) {
            float var = variance[c] / cluster_size;
            covariances[pixel_idx * channels + c] = (var > 0.05f) ? var : 0.05f;
        }
    } else {
        // If no points in cluster, set default covariance of 0.1
        for (int c = 0; c < channels; c++) {
            covariances[pixel_idx * channels + c] = 0.05f;
        }
    }
}