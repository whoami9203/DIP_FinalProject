import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from shadow_cancellation import shadow_cancellation

def masks_noise_removal(
        masks: np.ndarray,
        min_area: int = 60,
        closing_kernel_size: int = 5
        ) -> np.ndarray:
    """
    Remove noise from foreground masks by:
    1. Eliminating small white connected components (foreground)
    2. Filling ALL black holes inside foreground objects
       (keeping only the external background connected to image borders)
    """
    if len(masks.shape) != 3:
        raise ValueError("Masks must be a 3D array (num_frames, height, width)")
    
    cleaned_masks = np.zeros_like(masks)
    num_frames = masks.shape[0]
    
    # Create kernel for morphological operations
    kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    
    print(f"Cleaning masks: removing components < {min_area} pixels and filling ALL internal holes...")
    
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
        
        # STEP 2: Morphological closing to fill small holes
        # Apply closing to the cleaned mask
        clean_mask = cv2.morphologyEx(clean_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # STEP 3: Fill all internal holes (black areas not connected to border)
        ff_mask = clean_mask.copy()
        cv2.floodFill(ff_mask, None, (0, 0), 1)
        ff_mask = 1 - ff_mask

        # If clean_mask is 1 or ff_mask is 1, set to 1
        cleaned_mask = np.where((clean_mask == 1) | (ff_mask == 1), 1, 0).astype(np.uint8)

        # Append the cleaned mask to the cleaned_masks array
        cleaned_masks[frame_idx] = cleaned_mask
    
    # Report statistics
    original_white_pixels = np.sum(masks)
    cleaned_white_pixels = np.sum(cleaned_masks)
    
    if cleaned_white_pixels >= original_white_pixels:
        percentage_added = 100 * (cleaned_white_pixels - original_white_pixels) / max(1, original_white_pixels)
        print(f"Noise removal complete. Added {percentage_added:.2f}% to foreground pixels.")
    else:
        percentage_removed = 100 * (original_white_pixels - cleaned_white_pixels) / max(1, original_white_pixels)
        print(f"Noise removal complete. Removed {percentage_removed:.2f}% of foreground pixels.")
    
    return cleaned_masks

def foreground_segmentation(
        video_path: str,
        video_basename: str,
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
    # mahalanobis_distances = np.transpose(mahalanobis_distances, (1, 2, 0))  # Now shape is (height, width, num_frames)
    
    # # Compute statistics for each pixel position
    # print("Computing Mahalanobis distance statistics for each position...")
    
    # # Compute statistics all at once (vectorized)
    # mahalanobis_statistics = {
    #     'variance': np.var(mahalanobis_distances, axis=2),
    #     'mean': np.mean(mahalanobis_distances, axis=2),
    #     'q1': np.percentile(mahalanobis_distances, 25, axis=2),
    #     'q2': np.percentile(mahalanobis_distances, 50, axis=2),  # median
    #     'q3': np.percentile(mahalanobis_distances, 75, axis=2)
    # }
    
    # # Calculate IQR
    # mahalanobis_statistics['iqr'] = mahalanobis_statistics['q3'] - mahalanobis_statistics['q1']
    
    # print("Mahalanobis distance statistics computation complete.")
    
    # # Add global statistics
    # mahalanobis_statistics['global'] = {
    #     'variance_mean': np.mean(mahalanobis_statistics['variance']),
    #     'variance_median': np.median(mahalanobis_statistics['variance']),
    #     'mean_mean': np.mean(mahalanobis_statistics['mean']),
    #     'mean_median': np.median(mahalanobis_statistics['mean']),
    #     'q1_mean': np.mean(mahalanobis_statistics['q1']),
    #     'q2_mean': np.mean(mahalanobis_statistics['q2']),
    #     'q3_mean': np.mean(mahalanobis_statistics['q3']),
    #     'iqr_mean': np.mean(mahalanobis_statistics['iqr']),
    #     'q1_median': np.median(mahalanobis_statistics['q1']),
    #     'q2_median': np.median(mahalanobis_statistics['q2']),
    #     'q3_median': np.median(mahalanobis_statistics['q3']),
    #     'iqr_median': np.median(mahalanobis_statistics['iqr'])
    # }

    # Save the unclean foreground masks as a video
    output_unclean_masks_path = f"{video_basename}_unclean_masks.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_unclean = cv2.VideoWriter(output_unclean_masks_path, fourcc, 15.0, (width, height))
    # Write each frame to the unclean masks video
    print(f"Saving unclean masks to {output_unclean_masks_path}")
    for i in range(frame_idx):
        # Convert binary mask to 3-channel (required for mp4)
        mask_frame = np.stack([foreground_masks[i]*255]*3, axis=-1).astype(np.uint8)
        out_unclean.write(mask_frame)
    out_unclean.release()
    
    # Apply noise removal to the masks
    foreground_masks = masks_noise_removal(foreground_masks)
    
    # Save the cleaned foreground masks as a video
    output_clean_masks_path = f"{video_basename}_clean_masks.mp4"
    out_clean = cv2.VideoWriter(output_clean_masks_path, fourcc, 15.0, (width, height))
    # Write each frame to the clean masks video
    print(f"Saving clean masks to {output_clean_masks_path}")
    for i in range(frame_idx):
        # Convert binary mask to 3-channel (required for mp4)
        mask_frame = np.stack([foreground_masks[i]*255]*3, axis=-1).astype(np.uint8)
        out_clean.write(mask_frame)
    out_clean.release()

    # Apply shadow cancellation if needed
    print("Applying shadow cancellation...")
    shadowless_masks = shadow_cancellation(
        video_path=video_path,
        video_basename=video_basename,
        masks=foreground_masks,
        background_mean=background_means,
        window_size=3
    )

    # Save the shadowless masks as a video
    output_shadowless_masks_path = f"{video_basename}_shadowless_masks.mp4"
    out_shadowless = cv2.VideoWriter(output_shadowless_masks_path, fourcc, 15.0, (width, height))
    # Write each frame to the shadowless masks video
    print(f"Saving shadowless masks to {output_shadowless_masks_path}")
    for i in range(frame_idx):
        # Convert binary mask to 3-channel (required for mp4)
        mask_frame = np.stack([shadowless_masks[i]*255]*3, axis=-1).astype(np.uint8)
        out_shadowless.write(mask_frame)
    out_shadowless.release()

    # Apply AND operation to combine masks
    foreground_masks = np.logical_and(foreground_masks, shadowless_masks).astype(np.uint8)
    
    return foreground_masks

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