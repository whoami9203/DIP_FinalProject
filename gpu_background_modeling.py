import cupy as cp
import cuml

import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# Import cuML K-means
from cuml.cluster import KMeans


def gpu_k_means_background_clustering(video_path: str, max_k=3) -> tuple:
    """
    Perform K-means clustering on the background of a video in temporal domain using GPU acceleration.
    For each pixel position, tests different K values and selects the best one using BIC.
    Returns both mean vectors and covariance matrices for each position.
    
    Args:
        video_path (str): Path to the input video file.
        max_k (int): Maximum number of clusters to consider (default: 3).
        
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
    background_covariances = np.zeros((height, width, 3), dtype=np.float32)  # Diagonal elements only
    
    # Store optimal K values for statistics
    optimal_k_values = np.zeros((height, width), dtype=np.int32)
    
    # Dictionary to track best BIC score for each pixel
    best_bics = {}
    
    # Main processing loop with batched GPU processing
    print("Performing GPU-accelerated K-means background clustering with batch processing...")
    
    # Process pixels in batches for better GPU utilization
    batch_size = 1000  # Adjust based on GPU memory
    total_pixels = height * width
    
    # Convert all pixel values to a list for batch processing
    pixel_list = []
    pixel_positions = []
    
    for y in range(height):
        for x in range(width):
            pixel_list.append(pixel_values[y, x])
            pixel_positions.append((y, x))
    
    # Process in batches
    for batch_start in tqdm(range(0, total_pixels, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, total_pixels)
        batch_positions = pixel_positions[batch_start:batch_end]
        batch_pixels = pixel_list[batch_start:batch_end]
        
        # Process each K value across all pixels in the batch
        for k in range(1, max_k + 1):
            # Special case for k=1
            if k == 1:
                # For k=1, just compute mean and variance directly
                for i, (y, x) in enumerate(batch_positions):
                    temporal_data = batch_pixels[i]
                    
                    # Only update if this is the first k we're processing for this pixel
                    if optimal_k_values[y, x] == 0:
                        background_means[y, x] = np.mean(temporal_data, axis=0)
                        background_covariances[y, x] = np.maximum(np.var(temporal_data, axis=0), 0.1)
                        optimal_k_values[y, x] = 1
                
                # All pixels automatically get a BIC score for k=1
                continue
            
            try:
                # Process in sub-batches if needed (for large K values with many pixels)
                sub_batch_size = max(1, min(200, len(batch_pixels)))
                
                for sub_batch_start in range(0, len(batch_pixels), sub_batch_size):
                    sub_batch_end = min(sub_batch_start + sub_batch_size, len(batch_pixels))
                    sub_batch_pixels = batch_pixels[sub_batch_start:sub_batch_end]
                    sub_batch_positions = batch_positions[sub_batch_start:sub_batch_end]
                    
                    # Process each pixel in sub-batch
                    for i, (y, x) in enumerate(sub_batch_positions):
                        temporal_data = sub_batch_pixels[i]
                        
                        # Skip if we've already found a better K for this pixel
                        if optimal_k_values[y, x] > k:
                            continue
                            
                        try:
                            # Transfer to GPU
                            temporal_data_gpu = cp.asarray(temporal_data)
                            
                            # Run K-means
                            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
                            kmeans.fit(temporal_data_gpu)
                            
                            # Get cluster assignments and centers
                            labels = kmeans.labels_
                            centers = kmeans.cluster_centers_
                            
                            # Calculate BIC using helper function
                            bic = calculate_bic_gpu(
                                temporal_data_gpu, 
                                labels, 
                                centers, 
                                k, 
                                len(temporal_data), 
                                temporal_data.shape[1]
                            )
                            
                            # Compare with existing best BIC for this pixel
                            current_best_k = optimal_k_values[y, x]
                            
                            if current_best_k == 0 or bic > best_bics.get((y, x), float('-inf')):
                                # This is a better clustering
                                best_bics[(y, x)] = bic
                                optimal_k_values[y, x] = k
                                
                                # Find largest cluster
                                labels_cpu = cp.asnumpy(labels)
                                centers_cpu = cp.asnumpy(centers)
                                
                                cluster_sizes = np.bincount(labels_cpu, minlength=k)
                                largest_cluster_id = np.argmax(cluster_sizes)
                                
                                # Get pixels belonging to largest cluster
                                largest_cluster_mask = (labels_cpu == largest_cluster_id)
                                cluster_points = temporal_data[largest_cluster_mask]
                                
                                # Calculate mean and covariance of largest cluster
                                background_means[y, x] = centers_cpu[largest_cluster_id]
                                background_covariances[y, x] = np.maximum(np.var(cluster_points, axis=0), 0.1)
                        
                        except Exception as e:
                            # If error occurs, fall back to k=1 if no valid clustering yet
                            if optimal_k_values[y, x] == 0:
                                optimal_k_values[y, x] = 1
                                background_means[y, x] = np.mean(temporal_data, axis=0)
                                background_covariances[y, x] = np.maximum(np.var(temporal_data, axis=0), 0.1)
                                
            except Exception as e:
                print(f"Error processing batch with k={k}: {e}")
    
    # Report statistics on selected K values
    k_counts = np.bincount(optimal_k_values.flatten(), minlength=max_k+1)
    print("\nStatistics of optimal K values:")
    for k in range(1, max_k+1):
        print(f"  K={k}: {k_counts[k]} pixels ({k_counts[k]/total_pixels*100:.2f}%)")
    
    print("\nGPU-accelerated background modeling complete.")
    
    return (background_means, background_covariances)


def calculate_bic_gpu(data_gpu, labels, centers, k, n_samples, n_dims):
    """Helper function to calculate BIC score for a clustering result."""
    # Calculate log-likelihood
    log_likelihood = 0
    
    # Calculate within-cluster variance
    variances = []
    for cluster_id in range(k):
        cluster_mask = (labels == cluster_id)
        if cp.any(cluster_mask):
            cluster_points = data_gpu[cluster_mask]
            cluster_center = centers[cluster_id]
            # Average squared Euclidean distance to center
            var = cp.mean(cp.sum((cluster_points - cluster_center) ** 2, axis=1))
            variances.append(var)
    
    if not variances:
        return float('-inf')
        
    avg_var = cp.mean(cp.array(variances))
    if avg_var <= 0:
        avg_var = 1e-6  # Avoid log(0)
    
    # Calculate log-likelihood (approximate, assuming spherical clusters)
    log_likelihood = -0.5 * n_samples * n_dims * cp.log(2 * cp.pi * avg_var)
    log_likelihood -= 0.5 * n_samples * n_dims  # Constant term from distance
    
    # Parameters: k centers (k*dims) and k variances
    n_params = k * n_dims + k
    
    # BIC = log(L) - 0.5 * num_params * log(n)
    bic = log_likelihood - 0.5 * n_params * cp.log(n_samples)
    
    return float(bic)