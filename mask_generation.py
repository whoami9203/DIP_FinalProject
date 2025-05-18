import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

def masks_noise_removal(
        masks: np.ndarray,
        min_area: int = 60,
        min_hole_area: int = 15,  # New parameter for minimum hole size
        closing_kernel_size: int = 5
        ) -> np.ndarray:
    """
    Remove noise from foreground masks by:
    1. Eliminating small white connected components (foreground)
    2. Filling small black holes (background)
    3. Optional morphological closing to smooth boundaries
    
    Args:
        masks (np.ndarray): Binary foreground masks with shape (num_frames, height, width)
        min_area (int): Minimum area for a white component to be preserved
        min_hole_area (int): Minimum area for a black hole to be preserved
        closing_kernel_size (int): Size of the kernel for morphological closing
    
    Returns:
        np.ndarray: Cleaned binary masks with small components and holes removed
    """
    if len(masks.shape) != 3:
        raise ValueError("Masks must be a 3D array (num_frames, height, width)")
    
    cleaned_masks = np.zeros_like(masks)
    num_frames = masks.shape[0]
    
    # Create kernel for closing operation
    kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    
    print(f"Cleaning masks: removing components < {min_area} pixels and holes < {min_hole_area} pixels...")
    
    for frame_idx in tqdm(range(num_frames), desc="Cleaning masks"):
        # Get current frame mask
        mask = masks[frame_idx]
        
        # STEP 1: Remove small white connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=4, ltype=cv2.CV_32S
        )
        
        # Create a clean mask, starting with zeros
        clean_mask = np.zeros_like(mask)
        
        # Skip label 0 which is the background
        for label in range(1, num_labels):
            # Get area of the component
            area = stats[label, cv2.CC_STAT_AREA]
            
            # If area is large enough, keep this component
            if area >= min_area:
                clean_mask[labels == label] = 1
        
        # STEP 2: Remove small black holes (invert, find components, remove small ones, invert back)
        # Invert the mask to make holes into foreground objects
        inverted_mask = 1 - clean_mask
        
        # Find connected black components (holes)
        num_holes, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
            inverted_mask, connectivity=4, ltype=cv2.CV_32S
        )
        
        # Create a mask of holes to fill (small black areas)
        holes_to_fill = np.zeros_like(inverted_mask)
        
        # Skip label 0 which is now the "background" in the inverted image
        # (which was actually the foreground in the original)
        for label in range(1, num_holes):
            # Get area of the hole
            hole_area = hole_stats[label, cv2.CC_STAT_AREA]
            
            # If hole is small enough, mark it to be filled (converted to foreground)
            if hole_area < min_hole_area:
                holes_to_fill[hole_labels == label] = 1
        
        # Fill the small holes by adding them to the clean mask
        clean_mask = clean_mask | holes_to_fill
        
        # Apply optional morphological closing to smooth boundaries
        # Uncomment if needed
        # clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
        
        # Store the cleaned mask
        cleaned_masks[frame_idx] = clean_mask
    
    # Report statistics
    original_white_pixels = np.sum(masks)
    cleaned_white_pixels = np.sum(cleaned_masks)
    percentage_change = 100 * (original_white_pixels - cleaned_white_pixels) / max(1, original_white_pixels)
    
    if percentage_change > 0:
        print(f"Noise removal complete. Removed {percentage_change:.2f}% of foreground pixels.")
    
    return cleaned_masks

def foreground_segmentation(
        video_path: str,
        background_means: np.ndarray,
        background_covariances: np.ndarray,
        alpha: float = 9.0
    ) -> tuple:
    """
    Perform foreground segmentation using Gaussian background modeling.
    For each pixel, compare the current frame with the background model
    and classify it as foreground or background using Mahalanobis distance.
    Also computes statistics about Mahalanobis distances across frames for each position.
    
    Args:
        video_path (str): Path to the input video file.
        background_means (np.ndarray): Background model means (height, width, 3).
        background_covariances (np.ndarray): Background model covariances (height, width, 3).
        alpha (float): Scalar to determine size of the ellipsoid (default: 15.0).
        
    Returns:
        tuple: (foreground_masks, mahalanobis_statistics) where:
               - foreground_masks: Binary mask of the foreground (num_frames, height, width)
               - mahalanobis_statistics: Dictionary containing Mahalanobis distance statistics for each position
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Verify shapes match between video and background model
    if background_means.shape[:2] != (height, width):
        raise ValueError("Background model dimensions don't match video dimensions")
    
    # Initialize foreground masks
    foreground_masks = np.zeros((frame_count, height, width), dtype=np.uint8)
    
    # Initialize mahalanobis_distances as None, will create properly shaped array on first frame
    mahalanobis_distances = None
    
    # Process each frame
    print(f"Segmenting foreground using Mahalanobis distance (alpha={alpha})...")
    
    # For efficiency, precompute scaled inverse covariances
    # Since we assume RGB channels are independent, the covariance matrix is diagonal
    # and the inverse is just the reciprocal of the diagonal elements
    inv_scaled_covariances = 1.0 / (alpha * background_covariances)
    
    with tqdm(total=frame_count, desc="Processing frames") as pbar:
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Calculate Mahalanobis distance for each pixel
            # g(x,y) = [I(x,y) - μ(x,y)]^T [αΣ(x,y)]^-1 [I(x,y) - μ(x,y)]
            
            # Step 1: Calculate difference between current frame and background mean
            diff = frame.astype(np.float32) - background_means
            
            # Step 2: Square the difference and multiply by inverse scaled covariance
            # Since we're using diagonal covariance matrices, we can simplify the matrix operations
            mahalanobis_dist = np.sum((diff**2) * inv_scaled_covariances, axis=2)
            
            # Collect Mahalanobis distances for each pixel (for statistics) - vectorized version
            # Append to our collection
            if mahalanobis_distances is None:
                # First frame - initialize array
                mahalanobis_distances = mahalanobis_dist[np.newaxis, :, :]  # Shape: (1, height, width)
            else:
                # Append for subsequent frames - this is memory inefficient but simple
                mahalanobis_distances = np.append(mahalanobis_distances, mahalanobis_dist[np.newaxis, :, :], axis=0)
            
            # Step 3: Create binary mask (1 where g(x,y) > 1, 0 otherwise)
            foreground_masks[frame_idx] = (mahalanobis_dist > 1.0).astype(np.uint8)
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    print(f"Foreground segmentation complete. Generated {frame_idx} masks.")
    
    # Transpose mahalanobis_distances to get (height, width, frames) for easier statistics computation
    # This makes each pixel's time series contiguous in memory
    mahalanobis_distances = np.transpose(mahalanobis_distances, (1, 2, 0))  # Now shape is (height, width, num_frames)
    
    # Compute statistics for each pixel position
    print("Computing Mahalanobis distance statistics for each position...")
    
    # Compute statistics all at once (vectorized)
    mahalanobis_statistics = {
        'variance': np.var(mahalanobis_distances, axis=2),
        'mean': np.mean(mahalanobis_distances, axis=2),
        'q1': np.percentile(mahalanobis_distances, 25, axis=2),
        'q2': np.percentile(mahalanobis_distances, 50, axis=2),  # median
        'q3': np.percentile(mahalanobis_distances, 75, axis=2)
    }
    
    # Calculate IQR
    mahalanobis_statistics['iqr'] = mahalanobis_statistics['q3'] - mahalanobis_statistics['q1']
    
    print("Mahalanobis distance statistics computation complete.")
    
    # Add global statistics
    mahalanobis_statistics['global'] = {
        'variance_mean': np.mean(mahalanobis_statistics['variance']),
        'variance_median': np.median(mahalanobis_statistics['variance']),
        'mean_mean': np.mean(mahalanobis_statistics['mean']),
        'mean_median': np.median(mahalanobis_statistics['mean']),
        'q1_mean': np.mean(mahalanobis_statistics['q1']),
        'q2_mean': np.mean(mahalanobis_statistics['q2']),
        'q3_mean': np.mean(mahalanobis_statistics['q3']),
        'iqr_mean': np.mean(mahalanobis_statistics['iqr']),
        'q1_median': np.median(mahalanobis_statistics['q1']),
        'q2_median': np.median(mahalanobis_statistics['q2']),
        'q3_median': np.median(mahalanobis_statistics['q3']),
        'iqr_median': np.median(mahalanobis_statistics['iqr'])
    }

    # Apply noise removal to the masks
    foreground_masks = masks_noise_removal(foreground_masks)
    
    return foreground_masks, mahalanobis_statistics

def apply_mask_to_video(
        input_video_path: str, 
        output_video_path: str, 
        foreground_masks: np.ndarray, 
        darkness_factor: float = 0.3
    ) -> None:
    """
    Apply foreground masks to a video. Original pixels are preserved where the mask is 1,
    and pixels are darkened where the mask is 0.
    
    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output masked video.
        foreground_masks (np.ndarray): Binary foreground masks (num_frames, height, width).
        darkness_factor (float): Factor to darken background pixels (0-1, default: 0.3).
                                Lower value = darker background.
    """
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video file: {input_video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Verify mask dimensions match video dimensions
    if foreground_masks.shape[1:] != (height, width):
        raise ValueError("Mask dimensions don't match video dimensions")
    if foreground_masks.shape[0] < frame_count:
        print(f"Warning: Fewer masks ({foreground_masks.shape[0]}) than frames ({frame_count})")
        frame_count = foreground_masks.shape[0]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Could not open output video file: {output_video_path}")
    
    # Process each frame
    print(f"Applying masks to video...")
    
    with tqdm(total=frame_count, desc="Processing") as pbar:
        frame_idx = 0
        
        while frame_idx < frame_count:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get corresponding mask
            mask = foreground_masks[frame_idx]
            
            # Create a 3-channel mask for element-wise operations
            mask_3ch = np.stack((mask, mask, mask), axis=2)
            
            # Create darkened version of the frame
            darkened_frame = (frame * darkness_factor).astype(np.uint8)
            
            # Apply mask: original pixel where mask=1, darkened pixel where mask=0
            masked_frame = np.where(mask_3ch == 1, frame, darkened_frame)
            
            # Write the frame
            out.write(masked_frame)
            
            frame_idx += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    print(f"Masked video saved to {output_video_path}")