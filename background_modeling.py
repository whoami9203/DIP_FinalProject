import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from cyvlfeat.kmeans import kmeans
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

# Import cuML K-means



# Process each pixel position
def process_pixel_batch_with_progress(
        pixel_data, 
        start_y, 
        height, width, 
        batch_size=10, 
        process_id=0, 
    ):
    """
    Process a batch of rows of pixels with X-means clustering with progress bar.
    Computes both mean vectors and covariance matrices for the background model.
    """
    # Results will now contain both mean and covariance for each pixel
    batch_mean_result = np.zeros((min(batch_size, height-start_y), width, 3), dtype=np.float32)
    batch_cov_result = np.zeros((min(batch_size, height-start_y), width, 3), dtype=np.float32)  # Diagonal elements only
    
    # Calculate actual batch size for this batch
    actual_batch_size = min(batch_size, height-start_y)
    
    # Create progress bar for this specific process
    desc = f"Process {process_id}"
    pbar = tqdm(total=actual_batch_size*width, 
                desc=desc, 
                position=process_id,
                leave=True,
                ncols=80)
    
    # Track statistics for optimization reporting
    static_pixels = 0
    dynamic_pixels = 0
    
    for y_offset in range(actual_batch_size):
        y = start_y + y_offset
        for x in range(width):
            # Get temporal RGB data for this pixel
            temporal_data = pixel_data[y, x]  # shape: (num_frames, 3)
            
            # Calculate variance of RGB values across frames
            # Sum variances across R, G, B channels
            # rgb_variance = np.var(temporal_data, axis=0).sum()
            

            # Initialize centers with k-means++
            initial_centers = kmeans_plusplus_initializer(temporal_data, 1).initialize()
            
            # Create and run X-means
            xmeans_instance = xmeans(temporal_data, initial_centers, max_number_clusters=3)
            xmeans_instance.process()
            
            # Get clusters and centers
            clusters = xmeans_instance.get_clusters()
            centers = xmeans_instance.get_centers()
            
            # Find the largest cluster 
            largest_idx = np.argmax([len(cluster) for cluster in clusters])
            largest_cluster = clusters[largest_idx]
            mean_vector = centers[largest_idx]
            
            # Store the mean vector
            batch_mean_result[y_offset, x] = mean_vector
            
            # Extract the pixels in the largest cluster
            background_pixels = temporal_data[largest_cluster]
            
            # Compute covariance matrix (diagonal elements only - per channel variance)
            # If cluster has enough points
            if len(background_pixels) > 1:
                cov_vector = np.var(background_pixels, axis=0)
                # Ensure minimum variance to prevent numerical issues
                cov_vector = np.maximum(cov_vector, 1.0)
            else:
                # Fallback if cluster only has one point
                cov_vector = np.array([1.0, 1.0, 1.0])
                
            batch_cov_result[y_offset, x] = cov_vector
            dynamic_pixels += 1
            
            # Update progress bar
            pbar.update(1)
    
    # Close progress bar
    pbar.close()
    return start_y, batch_mean_result, batch_cov_result

def k_means_background_clustering(video_path: str) -> tuple:
    """
    Perform X-means clustering on the background of a video in temporal domain.
    For each pixel position, clusters the RGB values across all frames.
    Returns both mean vectors and covariance matrices for each position.
    
    Args:
        video_path (str): Path to the input video file.
        
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
    
    # Main processing loop with parallelization
    print("Performing Gaussian background modeling with parallel processing...")
    
    # Determine optimal batch size and number of processes
    num_processes = mp.cpu_count()
    batch_size = max(1, height // (num_processes))  # Simplified batch calculation
    print(f"Using {num_processes} processes with batch size {batch_size}.")
    
    # Create a pool of worker processes
    results = []
    with mp.Pool(processes=num_processes) as pool:
        # Generate tasks with process IDs
        tasks = []
        for i, start_y in enumerate(range(0, height, batch_size)):
            tasks.append((pixel_values, start_y, height, width, batch_size, i))
        
        # Map tasks to worker processes
        for result in pool.starmap(process_pixel_batch_with_progress, tasks):
            results.append(result)
    
    print("All processes completed.")

    # Assemble results into the final background model
    for start_y, batch_means, batch_covs in results:
        end_y = min(start_y + batch_size, height)
        background_means[start_y:end_y] = batch_means[:end_y-start_y]
        background_covariances[start_y:end_y] = batch_covs[:end_y-start_y]

    print("Gaussian background modeling complete.")
    
    return (background_means, background_covariances)
