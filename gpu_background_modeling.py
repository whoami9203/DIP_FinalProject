import cupy as cp

import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

def load_k1_stats_kernel():
    """Load and compile the CUDA kernel for k=1 statistics"""
    with open('k1_stats.cu', 'r') as f:
        cuda_source = f.read()
    
    # Create a RawKernel from the source code
    k1_kernel = cp.RawKernel(cuda_source, 'k1_stats_kernel')
    return k1_kernel

def load_kmeans_kernel():
    """Load and compile the CUDA kernel"""
    with open('batch_kmeans.cu', 'r') as f:
        cuda_source = f.read()
    
    # Create a RawKernel from the source code
    kmeans_kernel = cp.RawKernel(cuda_source, 'batch_kmeans_kernel')
    return kmeans_kernel

def load_postprocessing_kernel():
    """Load and compile the CUDA kernel for post-processing"""
    with open('postprocess.cu', 'r') as f:
        cuda_source = f.read()
    
    # Create a RawKernel from the source code
    postprocess_kernel = cp.RawKernel(cuda_source, 'postprocess_kernel')
    return postprocess_kernel

def gpu_k1_stats(k1_kernel, pixel_data_gpu):
    """Calculate k=1 statistics using GPU for all pixels"""
    num_pixels = pixel_data_gpu.shape[0]
    num_frames = pixel_data_gpu.shape[1]
    channels = pixel_data_gpu.shape[2]
    
    # Allocate output arrays on GPU
    means_gpu = cp.zeros((num_pixels, channels), dtype=cp.float32)
    variances_gpu = cp.zeros((num_pixels, channels), dtype=cp.float32)
    bics_gpu = cp.zeros(num_pixels, dtype=cp.float32)
    
    # Configure kernel launch
    threads_per_block = 256
    blocks_needed = (num_pixels + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    k1_kernel(
        (blocks_needed,),
        (threads_per_block,),
        (pixel_data_gpu, means_gpu, variances_gpu, bics_gpu,
         num_pixels, num_frames, channels)
    )
    
    return means_gpu, variances_gpu, bics_gpu

def gpu_batch_kmeans(kmeans_kernel, postprocess_kernel, batch_data_gpu, k, max_iterations=100):
    """
    Execute the custom CUDA kernel for batch K-means
    
    Args:
        batch_data_gpu: CuPy array with shape (num_pixels, num_frames, channels)
        k: Number of clusters
        max_iterations: Maximum K-means iterations
        
    Returns:
        tuple: (bics, centers, covariances) as CuPy arrays
    """
    # Get dimensions
    num_pixels = batch_data_gpu.shape[0]
    num_frames = batch_data_gpu.shape[1]
    channels = batch_data_gpu.shape[2]
    
    # Allocate output arrays on GPU
    centers_gpu = cp.zeros((num_pixels, k, channels), dtype=cp.float32)
    labels_gpu = cp.zeros((num_pixels, num_frames), dtype=cp.int32)
    bics_gpu = cp.zeros(num_pixels, dtype=cp.float32)
    largest_ids_gpu = cp.zeros(num_pixels, dtype=cp.int32)
    
    # Calculate shared memory size
    # centers + new_centers + cluster_sizes + distances + probability arrays
    shared_mem_size = (k * channels * 2 + k + 2 * num_frames) * 4  # bytes
    
    # Launch configuration
    threads_per_block = min(512, num_frames)  # Adjust based on your GPU
    
    # Launch kernel
    kmeans_kernel(
        (num_pixels,),       # Grid dimensions: one block per pixel
        (threads_per_block,), # Block dimensions: threads cooperate on one pixel
        (batch_data_gpu, centers_gpu, labels_gpu, bics_gpu, largest_ids_gpu,
        num_pixels, num_frames, channels, k, max_iterations),
        shared_mem=shared_mem_size
    )
    
    # Now compute covariances based on the assignments and largest cluster
    covariances_gpu = cp.zeros((num_pixels, channels), dtype=cp.float32)
    
    # Extract best centers (those of the largest cluster)
    best_centers = cp.zeros((num_pixels, channels), dtype=cp.float32)
    
    # This could be a separate kernel, but for simplicity we'll use CuPy ops
    # for i in range(num_pixels):
    #     largest_id = largest_ids_gpu[i]
    #     best_centers[i] = centers_gpu[i, largest_id]
        
    #     # Calculate covariance for largest cluster
    #     mask = (labels_gpu[i] == largest_id)
    #     if cp.any(mask):
    #         points = batch_data_gpu[i][mask]
    #         covariances_gpu[i] = cp.maximum(cp.var(points, axis=0), 0.05)
    #     else:
    #         covariances_gpu[i] = cp.ones(channels, dtype=cp.float32) * 0.05

    # Configure kernel launch
    threads_per_block_post = 256
    blocks_needed = (num_pixels + threads_per_block_post - 1) // threads_per_block_post
    
    # Launch post-processing kernel
    postprocess_kernel(
        (blocks_needed,),
        (threads_per_block_post,),
        (centers_gpu, labels_gpu, largest_ids_gpu, batch_data_gpu,
         best_centers, covariances_gpu, 
         num_pixels, num_frames, channels, k)
    )
    
    return bics_gpu, best_centers, covariances_gpu

def gpu_k_means_background_clustering(video_path: str, max_k=3, variance_multiplier=2.0) -> tuple:
    """
    Perform K-means clustering on the background of a video in temporal domain using GPU acceleration.
    For each pixel position, tests different K values and selects the best one using BIC.
    Returns both mean vectors and covariance matrices for each position.
    
    Args:
        video_path (str): Path to the input video file.
        max_k (int): Maximum number of clusters to consider (default: 3).
        variance_multiplier (float): Multiplier for median variance to set threshold (default: 2.0).
        
    Returns:
        tuple: (means, covariances) where:
               - means: Background model as a 3D array (height, width, 3) of RGB mean values
               - covariances: Background model as a 3D array (height, width, 3) of RGB variances
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {width}x{height}, {frame_count} frames")
    
    # Collect frames
    pixel_values = []
    
    print(f"Processing video frames from {video_path}...")
    with tqdm(total=frame_count, desc="Reading frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            pixel_values.append(frame)
            pbar.update(1)
    
    cap.release()
    
    # Convert list to numpy array for easier manipulation
    pixel_values = np.array(pixel_values)  # shape: (num_frames, height, width, 3)
    
    # Reshape to get temporal data for each pixel
    # From (num_frames, height, width, 3) to (height, width, num_frames, 3)
    pixel_values = np.transpose(pixel_values, (1, 2, 0, 3))
    
    # Initialize result arrays for background model
    background_means = np.zeros((height, width, 3), dtype=np.float32)
    background_covariances = np.zeros((height, width, 3), dtype=np.float32)
    
    # Store optimal K values for statistics
    optimal_k_values = np.zeros((height, width), dtype=np.int32)
    
    # Dictionary to track best BIC score for each pixel
    best_bics = {}
    
    # Convert pixel values to a flat list for batch processing
    pixel_list = []
    pixel_positions = []
    
    for y in range(height):
        for x in range(width):
            pixel_list.append(pixel_values[y, x])
            pixel_positions.append((y, x))
    
    total_pixels = height * width
    exception_count = 0
    exception_message = None
    
    # STEP 1: Process all pixels for k=1 case (no clustering needed)
    print("Computing k=1 case for all pixels...")
    
    # Load the k=1 kernel
    k1_kernel = load_k1_stats_kernel()

    # Process in batches to optimize GPU memory usage
    batch_size = min(20480, total_pixels)  # Adjust based on GPU memory
    k1_means = np.zeros((total_pixels, 3), dtype=np.float32)
    k1_variances = np.zeros((total_pixels, 3), dtype=np.float32)
    k1_bics = np.zeros(total_pixels, dtype=np.float32)

    for batch_start in tqdm(range(0, total_pixels, batch_size), desc="Computing k=1 statistics"):
        batch_end = min(batch_start + batch_size, total_pixels)
        batch_size_actual = batch_end - batch_start
        
        # Get this batch of pixels
        batch_pixels = pixel_list[batch_start:batch_end]
        
        try:
            # Transfer to GPU
            batch_data = np.array(batch_pixels, dtype=np.float32)
            batch_data_gpu = cp.asarray(batch_data)
            
            # Process on GPU
            batch_means_gpu, batch_variances_gpu, batch_bics_gpu = gpu_k1_stats(
                k1_kernel, batch_data_gpu
            )
            
            # Transfer results back to CPU
            k1_means[batch_start:batch_end] = cp.asnumpy(batch_means_gpu)
            k1_variances[batch_start:batch_end] = cp.asnumpy(batch_variances_gpu)
            k1_bics[batch_start:batch_end] = cp.asnumpy(batch_bics_gpu)
            
            # Free GPU memory
            del batch_data_gpu, batch_means_gpu, batch_variances_gpu, batch_bics_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
        except Exception as e:
            print(f"Error in GPU k=1 processing: {e}")
            
            # Fall back to CPU for this batch
            print("Falling back to CPU for this batch...")
            for i in range(batch_start, batch_end):
                temporal_data = pixel_list[i-batch_start]
                
                # Calculate mean and variance for k=1
                k1_means[i] = np.mean(temporal_data, axis=0)
                k1_variances[i] = np.maximum(np.var(temporal_data, axis=0), 0.05)
                
                # Calculate BIC score for k=1
                n_samples = len(temporal_data)
                n_dims = temporal_data.shape[1]
                var = np.mean(k1_variances[i])
                
                if var <= 0:
                    var = 1e-6
                
                log_likelihood = -0.5 * n_samples * n_dims * np.log(2 * np.pi * var)
                log_likelihood -= 0.5 * n_samples * n_dims
                n_params = n_dims + 1  # Mean + variance
                k1_bics[i] = log_likelihood - 0.5 * n_params * np.log(n_samples)

    # Store k=1 results for all pixels
    for i in range(total_pixels):
        y, x = pixel_positions[i]
        optimal_k_values[y, x] = 1
        background_means[y, x] = k1_means[i]
        background_covariances[y, x] = k1_variances[i]
        best_bics[(y, x)] = k1_bics[i]
    
    # STEP 2: Process k>1 cases using GPU acceleration and batching
    if max_k > 1:
        print("Computing k>1 cases using GPU acceleration...")
        
        # Calculate variance for each pixel position (using mean of RGB variances)
        pixel_variances = np.array([np.mean(k1_variances[i]) for i in range(total_pixels)])
        
        # Use median variance as threshold basis
        q1_variance = np.percentile(pixel_variances, 25)
        variance_threshold = q1_variance * variance_multiplier
        
        print(f"q1 variance: {q1_variance:.2f}")
        print(f"Using variance threshold: {variance_threshold:.2f} (q1 × {variance_multiplier})")
        
        # Identify high-variance pixels
        high_variance_indices = np.where(pixel_variances > variance_threshold)[0]
        
        print(f"Found {len(high_variance_indices)} high-variance pixels ({len(high_variance_indices)/total_pixels*100:.2f}%) that might benefit from k>1")
        
        if len(high_variance_indices) > 0:
            # Only process high-variance pixels for k>1
            high_var_positions = [pixel_positions[i] for i in high_variance_indices]
            high_var_pixels = [pixel_list[i] for i in high_variance_indices]
            
            # Process in batches to optimize GPU memory usage
            batch_size = min(20480, len(high_var_positions))  # Adjust based on GPU memory
            
            # Load the CUDA kernel
            kmeans_kernel = load_kmeans_kernel()
            postprocess_kernel = load_postprocessing_kernel()

            # Process each k value for all high-variance pixels
            for k in range(2, max_k + 1):
                print(f"Processing k={k} for {len(high_var_positions)} pixels")
                
                for batch_start in tqdm(range(0, len(high_var_positions), batch_size), desc=f"Processing k={k} batches"):
                    batch_end = min(batch_start + batch_size, len(high_var_positions))
                    batch_positions = high_var_positions[batch_start:batch_end]
                    batch_pixels = high_var_pixels[batch_start:batch_end]
                    
                    try:
                        # Transfer entire batch to GPU at once
                        batch_data = np.array(batch_pixels)
                        batch_data_gpu = cp.asarray(batch_data, dtype=cp.float32)
                        
                        # Process all pixels in parallel on GPU
                        batch_bics, batch_centers, batch_covariances = gpu_batch_kmeans(
                            kmeans_kernel, postprocess_kernel, batch_data_gpu, k, 200
                        )
                        
                        # Transfer results back to CPU
                        batch_bics_cpu = cp.asnumpy(batch_bics)
                        batch_centers_cpu = cp.asnumpy(batch_centers)
                        batch_covariances_cpu = cp.asnumpy(batch_covariances)
                                            
                        # After processing all pixels in the batch, update the best values
                        for i, (y, x) in enumerate(batch_positions):
                            # Only update if this clustering is better than the existing one
                            if batch_bics_cpu[i] < best_bics.get((y, x), float('-inf')):
                                best_bics[(y, x)] = batch_bics_cpu[i]
                                optimal_k_values[y, x] = k
                                background_means[y, x] = batch_centers_cpu[i]
                                background_covariances[y, x] = batch_covariances_cpu[i]
                        
                    except Exception as e:
                        print(f"Error processing batch with k={k}: {e}")
    
    # Report statistics on selected K values
    k_counts = np.bincount(optimal_k_values.flatten(), minlength=max_k+1)
    print("\nStatistics of optimal K values:")
    for k in range(1, max_k+1):
        print(f"  K={k}: {k_counts[k]} pixels ({k_counts[k]/total_pixels*100:.2f}%)")
    
    print(f"Exception count during processing: {exception_count}")
    if exception_count > 0:
        print(f"Last exception message: {exception_message}")
    print("\nGPU-accelerated background modeling complete.")
    
    return (background_means, background_covariances)
