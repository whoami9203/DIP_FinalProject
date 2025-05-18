// batch_kmeans.cu
extern "C" __global__ void batch_kmeans_kernel(
    float* pixel_data,      // [num_pixels, num_frames, channels]
    float* out_centers,     // [num_pixels, k, channels]
    int* out_labels,        // [num_pixels, num_frames]
    float* out_bics,        // [num_pixels]
    int* out_largest_ids,   // [num_pixels]
    int num_pixels,
    int num_frames, 
    int channels,
    int k,
    int max_iterations
) {
    // Get pixel index (one block per pixel)
    int pixel_idx = blockIdx.x;
    if (pixel_idx >= num_pixels) return;
    
    // Thread index within block
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Shared memory for centers and temporary calculations
    extern __shared__ float shared_mem[];
    float* centers = shared_mem;                             // [k, channels]
    float* new_centers = &centers[k * channels];            // [k, channels]
    int* cluster_sizes = (int*)&new_centers[k * channels];  // [k]
    float* distances = &shared_mem[k * channels * 2 + k];   // [num_threads]
    
    // Get data pointers for this pixel
    float* pixel_time_series = &pixel_data[pixel_idx * num_frames * channels];
    float* pixel_out_centers = &out_centers[pixel_idx * k * channels];
    int* pixel_labels = &out_labels[pixel_idx * num_frames];
    
    // Step 1: Initialize centers with k-means++ method
    // First center is chosen (using first frame for simplicity)
    if (tid < channels) {
        centers[tid] = pixel_time_series[tid];
    }
    __syncthreads();

    // Choose remaining centers
    for (int c = 1; c < k; c++) {
        // Calculate minimum squared distance to existing centers for each frame
        __shared__ float distances_sq[1024]; // Adjust size based on max threads
        __shared__ float distance_sum;
        
        if (tid == 0) distance_sum = 0.0f;
        
        // Initialize distances to max value
        if (tid < num_frames) distances_sq[tid] = 1e10f;
        __syncthreads();
        
        // Each thread calculates distances for some frames
        for (int frame = tid; frame < num_frames; frame += num_threads) {
            for (int existing_c = 0; existing_c < c; existing_c++) {
                float dist_sq = 0.0f;
                for (int d = 0; d < channels; d++) {
                    float diff = pixel_time_series[frame * channels + d] - 
                                centers[existing_c * channels + d];
                    dist_sq += diff * diff;
                }
                
                // Keep minimum distance to any existing center
                distances_sq[frame] = min(distances_sq[frame], dist_sq);
            }
            // Add to total sum (used for probability calculation)
            atomicAdd(&distance_sum, distances_sq[frame]);
        }
        __syncthreads();
        
        // We need to convert distances to cumulative probabilities
        __shared__ float cumulative_prob[1024]; // Adjust size as needed
        
        // Calculate cumulative probabilities
        if (tid < num_frames) {
            float prob = distances_sq[tid] / distance_sum;
            // Simple prefix sum for this thread's value
            cumulative_prob[tid] = prob;
            for (int offset = 1; offset < num_frames; offset <<= 1) {
                if (tid >= offset) {
                    cumulative_prob[tid] += cumulative_prob[tid - offset];
                }
                __syncthreads();
            }
        }
        __syncthreads();
        
        // Thread 0 selects the next center based on probabilities
        if (tid == 0) {
            // Generate a random value between 0 and 1
            // Simple pseudo-random number based on pixel_idx, c, and num_frames
            float r = (float)((pixel_idx * 263 + c * 71 + num_frames) % 1000) / 1000.0f;
            
            // Find the frame whose cumulative probability exceeds r
            int selected_frame = 0;
            for (int frame = 0; frame < num_frames; frame++) {
                if (cumulative_prob[frame] >= r) {
                    selected_frame = frame;
                    break;
                }
            }
            
            // Copy the selected frame to be the next center
            for (int d = 0; d < channels; d++) {
                centers[c * channels + d] = pixel_time_series[selected_frame * channels + d];
            }
        }
        __syncthreads();
    }
    __syncthreads();
    
    // Step 2: K-means iteration
    for (int iter = 0; iter < max_iterations; iter++) {
        // Reset cluster accumulations
        if (tid < k * channels) {
            new_centers[tid] = 0.0f;
        }
        if (tid < k) {
            cluster_sizes[tid] = 0;
        }
        __syncthreads();
        
        // Assign points to centers
        for (int frame = tid; frame < num_frames; frame += num_threads) {
            float min_dist = 1e10f;
            int best_cluster = 0;
            
            // Find closest center
            for (int c = 0; c < k; c++) {
                float dist = 0.0f;
                for (int d = 0; d < channels; d++) {
                    float diff = pixel_time_series[frame * channels + d] - 
                                 centers[c * channels + d];
                    dist += diff * diff;
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            
            // Assign label
            pixel_labels[frame] = best_cluster;
            
            // Atomically update cluster accumulators
            atomicAdd(&cluster_sizes[best_cluster], 1);
            for (int d = 0; d < channels; d++) {
                atomicAdd(&new_centers[best_cluster * channels + d], 
                         pixel_time_series[frame * channels + d]);
            }
        }
        __syncthreads();
        
        // Update centers
        if (tid < k * channels) {
            int c = tid / channels;
            int d = tid % channels;
            if (cluster_sizes[c] > 0) {
                centers[tid] = new_centers[tid] / cluster_sizes[c];
            }
        }
        __syncthreads();
    }
    
    // Step 3: Calculate BIC and find largest cluster
    // Each thread handles some BIC calculations
    __shared__ float log_likelihood;
    __shared__ int largest_cluster_id;
    __shared__ int largest_cluster_size;
    __shared__ float total_variance;
    
    if (tid == 0) {
        log_likelihood = 0.0f;
        largest_cluster_id = 0;
        largest_cluster_size = cluster_sizes[0];
        total_variance = 0.0f;
    }
    __syncthreads();
    
    // Find largest cluster
    if (tid < k) {
        atomicMax(&largest_cluster_size, cluster_sizes[tid]);
    }
    __syncthreads();
    
    if (tid < k) {
        if (cluster_sizes[tid] == largest_cluster_size) {
            largest_cluster_id = tid;
        }
    }
    __syncthreads();
    
    // Calculate variance for BIC    
    // Each thread computes part of the variance
    for (int frame = tid; frame < num_frames; frame += num_threads) {
        int c = pixel_labels[frame];
        float frame_variance = 0.0f;
        
        for (int d = 0; d < channels; d++) {
            float diff = pixel_time_series[frame * channels + d] - 
                        centers[c * channels + d];
            frame_variance += diff * diff;
        }
        
        atomicAdd(&total_variance, frame_variance);
    }
    __syncthreads();
    
    // Compute final BIC and set outputs
    if (tid == 0) {
        float avg_var = total_variance / (num_frames * channels);
        if (avg_var <= 0.0f) avg_var = 1e-6f;
        
        // Log likelihood calculation
        float log_like = -0.5f * num_frames * channels * logf(2.0f * 3.14159f * avg_var);
        log_like -= 0.5f * num_frames * channels;
        
        // Parameters count: k centers (k*channels) and k variances
        int n_params = k * channels + k;
        
        // BIC score
        float bic = log_like - 0.5f * n_params * logf(num_frames);
        out_bics[pixel_idx] = bic;
        out_largest_ids[pixel_idx] = largest_cluster_id;
        
        // Copy centers to output
        for (int i = 0; i < k * channels; i++) {
            pixel_out_centers[i] = centers[i];
        }
    }
}