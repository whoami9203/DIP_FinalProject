import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

def shadow_cancellation(
        video_path: str,
        masks: np.ndarray,
        background_mean: np.ndarray,
        T_L: float = 50.0,     # Luminance threshold
        T_C1: float = 10.0,    # Lower chrominance threshold
        T_C2: float = 30.0,    # Upper chrominance threshold
        T_G1: float = 5.0,     # Lower gradient threshold
        T_G2: float = 20.0,    # Upper gradient threshold
        T_S: float = 0.5,      # Shadow confidence threshold
        window_size: int = 3   # Window size for gradient density
        ) -> np.ndarray:
    """
    Remove shadows from detected foreground masks using luminance, chrominance,
    and gradient analysis in YCrCb color space.
    
    Args:
        video_path (str): Path to the input video file
        masks (np.ndarray): Foreground binary masks with shape (num_frames, height, width)
        background_mean (np.ndarray): Background model with shape (height, width, 3) in RGB
        T_L (float): Luminance threshold for shadow determination
        T_C1 (float): Lower chrominance threshold
        T_C2 (float): Upper chrominance threshold
        T_G1 (float): Lower gradient threshold
        T_G2 (float): Upper gradient threshold
        T_S (float): Shadow confidence threshold
        window_size (int): Window size for gradient density computation
        
    Returns:
        np.ndarray: Refined foreground masks with shadows removed
                   Shape: (num_frames, height, width)
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Check dimensions match
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = masks.shape[0]
    
    if (frame_height, frame_width) != masks.shape[1:3]:
        raise ValueError("Video dimensions don't match mask dimensions")
    
    if (frame_height, frame_width, 3) != background_mean.shape:
        raise ValueError("Video dimensions don't match background model dimensions")
    
    # Convert background model from RGB to YCrCb
    background_ycrcb = cv2.cvtColor(background_mean.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    background_y = background_ycrcb[:, :, 0].astype(np.float32)
    background_cr = background_ycrcb[:, :, 1].astype(np.float32)
    background_cb = background_ycrcb[:, :, 2].astype(np.float32)
    
    # Compute background gradient for g(x,y)
    bg_grad_x = cv2.Sobel(background_y, cv2.CV_32F, 1, 0, ksize=3)
    bg_grad_y = cv2.Sobel(background_y, cv2.CV_32F, 0, 1, ksize=3)
    bg_grad_mag = np.sqrt(bg_grad_x**2 + bg_grad_y**2)
    
    # Compute background gradient density (average over window)
    bg_grad_density = cv2.boxFilter(bg_grad_mag, -1, (window_size, window_size), 
                                    normalize=True, borderType=cv2.BORDER_REFLECT)
    
    # Initialize array to store refined masks
    refined_masks = np.zeros_like(masks)
    
    # Process each frame
    print("Computing shadow cancellation...")
    frame_idx = 0
    
    with tqdm(total=num_frames, desc="Processing frames") as pbar:
        while frame_idx < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to YCrCb
            frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            frame_y = frame_ycrcb[:, :, 0].astype(np.float32)
            frame_cr = frame_ycrcb[:, :, 1].astype(np.float32)
            frame_cb = frame_ycrcb[:, :, 2].astype(np.float32)
            
            # Get current mask
            mask = masks[frame_idx]
            
            # Step 1: Compute S_L (luminance difference)
            L = np.zeros((frame_height, frame_width), dtype=np.float32)
            L[mask > 0] = frame_y[mask > 0] - background_y[mask > 0]
            
            S_L = np.zeros((frame_height, frame_width), dtype=np.float32)
            S_L[np.logical_and(mask > 0, L <= 0)] = 1.0
            
            middle_mask_l = np.logical_and(mask > 0, np.logical_and(L > 0, L < T_L))
            S_L[middle_mask_l] = (T_L - L[middle_mask_l]) / T_L
            
            # Step 2: Compute S_C (chrominance difference)
            C = np.zeros((frame_height, frame_width), dtype=np.float32)
            C[mask > 0] = (np.abs(frame_cr[mask > 0] - background_cr[mask > 0]) + 
                          np.abs(frame_cb[mask > 0] - background_cb[mask > 0]))
            
            S_C = np.zeros((frame_height, frame_width), dtype=np.float32)
            S_C[np.logical_and(mask > 0, C <= T_C1)] = 1.0
            
            middle_mask_c = np.logical_and(mask > 0, np.logical_and(C > T_C1, C < T_C2))
            S_C[middle_mask_c] = (T_C2 - C[middle_mask_c]) / (T_C2 - T_C1)
            
            # Step 3: Compute S_G (gradient difference)
            # Compute current frame gradient
            frame_grad_x = cv2.Sobel(frame_y, cv2.CV_32F, 1, 0, ksize=3)
            frame_grad_y = cv2.Sobel(frame_y, cv2.CV_32F, 0, 1, ksize=3)
            frame_grad_mag = np.sqrt(frame_grad_x**2 + frame_grad_y**2)
            
            # Compute gradient density
            frame_grad_density = cv2.boxFilter(frame_grad_mag, -1, (window_size, window_size), 
                                              normalize=True, borderType=cv2.BORDER_REFLECT)
            
            # Compute gradient difference G(x,y)
            G = np.zeros((frame_height, frame_width), dtype=np.float32)
            G[mask > 0] = frame_grad_density[mask > 0] - bg_grad_density[mask > 0]
            
            S_G = np.zeros((frame_height, frame_width), dtype=np.float32)
            S_G[np.logical_and(mask > 0, G <= T_G1)] = 1.0
            
            middle_mask_g = np.logical_and(mask > 0, np.logical_and(G > T_G1, G < T_G2))
            S_G[middle_mask_g] = (T_G2 - G[middle_mask_g]) / (T_G2 - T_G1)
            
            # Step 4: Compute final shadow confidence score
            S = S_L * S_C * S_G
            
            # Step 5 & 6: Edge detection and shadow filtering
            # For each connected component in the mask
            num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
            
            refined_mask = np.zeros_like(mask)
            
            # Process each connected component separately
            for label in range(1, num_labels):
                # Get component mask
                component_mask = (labels == label).astype(np.uint8)
                
                # Find edges using Canny edge detector
                edges = cv2.Canny(component_mask * 255, 50, 150)
                
                # Filter edges based on shadow confidence
                edge_points = np.where(edges > 0)
                valid_edge_points = []
                
                for y, x in zip(edge_points[0], edge_points[1]):
                    if S[y, x] < T_S:  # Not a shadow point
                        valid_edge_points.append((x, y))  # Note: (x,y) for convex hull
                
                # Step 7: Apply convex hull if we have enough points
                if len(valid_edge_points) >= 3:  # Need at least 3 points for convex hull
                    hull = cv2.convexHull(np.array(valid_edge_points))
                    
                    # Create a mask from convex hull
                    hull_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                    cv2.fillConvexPoly(hull_mask, hull, 1)
                    
                    # Add to refined mask
                    refined_mask = np.logical_or(refined_mask, hull_mask)
                else:
                    # Not enough points for convex hull, use original component
                    # but remove high confidence shadow pixels
                    component_mask[S > T_S] = 0
                    refined_mask = np.logical_or(refined_mask, component_mask)
            
            # Store the refined mask
            refined_masks[frame_idx] = refined_mask
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    
    # Report statistics
    original_fg = np.sum(masks > 0)
    refined_fg = np.sum(refined_masks > 0)
    reduction = 100.0 * (original_fg - refined_fg) / original_fg if original_fg > 0 else 0
    
    print(f"Shadow cancellation complete. Processed {frame_idx} frames.")
    print(f"Removed approximately {reduction:.2f}% of pixels from foreground masks.")
    
    return refined_masks.astype(np.uint8)
