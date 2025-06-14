import cv2
import numpy as np
# import cupy as cp



def vwme_filter(image, window_size=5, epsilon=1e-5):
    """
    Parameters:
    - image: Input RGB image (uint8)
    - window_size: Size of local neighborhood (odd integer)
    - epsilon: Small value to avoid division by zero
    
    Returns:
    - Smoothed RGB image (same shape, dtype float32)
    """
    assert image.ndim == 3 and image.shape[2] == 3, "Input must be RGB"
    pad = window_size // 2
    smoothed = np.zeros_like(image, dtype=np.float32)

    for c in range(3):  # R, G, B channels
        channel = image[:, :, c].astype(np.float32)
        
        # Compute local mean and local variance
        local_mean = cv2.blur(channel, (window_size, window_size))
        local_sq_mean = cv2.blur(channel**2, (window_size, window_size))
        local_var = local_sq_mean - local_mean**2
        local_var = np.maximum(local_var, epsilon)  # Avoid division by 0

        # Inverse variance weights
        weight = 1.0 / local_var

        # Weighted mean computation
        weighted_sum = cv2.blur(channel * weight, (window_size, window_size))
        weight_sum = cv2.blur(weight, (window_size, window_size))

        vwme_channel = weighted_sum / weight_sum
        smoothed[:, :, c] = vwme_channel

    return np.clip(smoothed, 0, 255).astype(np.uint8)

def vwme_filter_gpu(image, window_size=5, epsilon=1e-5):
    """
    VWME filter using GPU acceleration via CuPy.
    """
    smoothed = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        ch = cp.asarray(image[:, :, c], dtype=cp.float32)

        mean = cp.asarray(cv2.blur(cp.asnumpy(ch), (window_size, window_size)))
        sq_mean = cp.asarray(cv2.blur(cp.asnumpy(ch**2), (window_size, window_size)))
        var = cp.maximum(sq_mean - mean**2, epsilon)

        inv_var = 1.0 / var
        weighted_sum = cp.asarray(cv2.blur(cp.asnumpy(ch * inv_var), (window_size, window_size)))
        weight_sum = cp.asarray(cv2.blur(cp.asnumpy(inv_var), (window_size, window_size)))

        vwme = weighted_sum / weight_sum
        smoothed[:, :, c] = cp.asnumpy(vwme)

    return np.clip(smoothed, 0, 255).astype(np.uint8)

def vwme_filter_optimized(image, window_size=5, epsilon=1e-5):
    """
    Fast VWME filter using vectorized OpenCV operations (CPU).
    """
    image = image.astype(np.float64)
    smoothed = np.zeros_like(image)

    for c in range(3):  # R, G, B channels
        channel = image[:, :, c]

        mean = cv2.blur(channel, (window_size, window_size))
        sq_mean = cv2.blur(channel**2, (window_size, window_size))
        var = np.maximum(sq_mean - mean**2, epsilon)

        inv_var = 1.0 / var
        weighted_sum = cv2.blur(channel * inv_var, (window_size, window_size))
        weight_sum = cv2.blur(inv_var, (window_size, window_size))

        smoothed[:, :, c] = weighted_sum / weight_sum

    return np.clip(smoothed, 0, 255).astype(np.uint8)

# Step 2: RGB Gradient Filtering using max RGB distance
def rgb_gradient_filter(image, window_size=5):
    """
    Fast and high-quality RGB gradient filter using maximum RGB distance in a window.
    Parameters:
        image: RGB image (uint8)
        window_size: Size of the local window (odd, e.g., 3, 5)
    Returns:
        gradient: 8-bit gradient image representing max RGB distance
    """
    assert image.ndim == 3 and image.shape[2] == 3, "Input must be RGB"
    pad = window_size // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)
    h, w, _ = image.shape
    gradient = np.zeros((h, w), dtype=np.float32)

    # Center patch (sliding window view)
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = padded[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
            diff = np.linalg.norm(image.astype(np.float32) - shifted.astype(np.float32), axis=2, ord=1)
            gradient = np.maximum(gradient, diff)

    return cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def rgb_central_gradient_filter(image, window_size=3):
    """
    Computes RGB gradient by comparing center pixel to diagonally opposite pixel.
    Reduces dual-edge effect by focusing on direct transitions.
    """
    assert image.ndim == 3 and image.shape[2] == 3
    h, w = image.shape[:2]
    image_f = image.astype(np.float32)
    pad = window_size // 2
    padded = cv2.copyMakeBorder(image_f, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    gradient = np.zeros((h, w), dtype=np.float32)

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right

    for dy, dx in offsets:
        shifted = padded[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
        diff = np.linalg.norm(image_f - shifted, axis=2, ord=1)
        gradient = np.maximum(gradient, diff)

    return cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def sobel_rgb_gradient(image, ksize=5):
    image = image.astype(np.float32)
    sobel_total = np.zeros(image.shape[:2], dtype=np.float32)

    for c in range(3):  # R, G, B channels
        sobel_x = cv2.Sobel(image[:, :, c], cv2.CV_32F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(image[:, :, c], cv2.CV_32F, 0, 1, ksize=ksize)
        mag = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_total += mag

    return cv2.normalize(sobel_total, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

import cv2
import numpy as np

def rgb_canny(image, low_thresh=100, high_thresh=200, combine='max'):
    """
    RGB-aware Canny edge detector using cv2.Canny on each channel.

    Parameters:
        image: Input RGB image (uint8)
        low_thresh: Lower threshold for hysteresis
        high_thresh: Upper threshold for hysteresis
        combine: 'max' for union, 'mean' for average, 'or' for binary OR

    Returns:
        edges: Combined edge map (uint8)
    """
    assert image.ndim == 3 and image.shape[2] == 3, "Input must be an RGB image"

    edges = []
    for i in range(3):  # R, G, B channels
        ch = image[:, :, i]
        edge = cv2.Canny(ch, low_thresh, high_thresh)
        edges.append(edge.astype(np.uint8))

    

    if combine == 'max':
        combined = np.max(np.stack(edges, axis=0), axis=0)
    elif combine == 'mean':
        combined = np.mean(np.stack(edges, axis=0), axis=0).astype(np.uint8)
    elif combine == 'or':
        combined = np.bitwise_or.reduce(np.stack(edges, axis=0))
    elif combine == 'add':
        for i in range(3):
            edges[i] = edges[i].astype(np.uint32)
        combined = np.sum(np.stack(edges, axis=0), axis=0)
        combined[combined > 255] = 255
        combined = combined.astype(np.uint8)
    else:
        raise ValueError("combine must be one of: 'max', 'mean', 'or'")

    return combined



# Step 3: Eliminate small local minima (thresholding)
def eliminate_small_local_minima(gradient, threshold=10):
    return np.where(gradient > threshold, gradient, 0).astype(np.uint8)

# Step 4: Flooding and Invisible Dam Construction (Watershed)
def apply_watershed_circle(image, gradient):
    # Get sure foreground using threshold
    _, sure_fg = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # sure_fg = cv2.erode(sure_fg, np.ones((3, 3), np.uint8), iterations=1)
    opening = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Sure background from dilation
    sure_bg = cv2.dilate(sure_fg, np.ones((3, 3), np.uint8), iterations=3)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1  # Ensure background is not 0
    markers[unknown == 255] = 0  # Unknown is 0

    # Apply watershed
    image_ws = image.copy()
    cv2.watershed(image_ws, markers)

    return markers

def apply_watershed(image, gradient):
    # Invert gradient to flood from low-intensity areas
    inv_grad = 255 - gradient

    # Use distance transform for better markers
    _, inv_grad = cv2.threshold(inv_grad, 230, 255, cv2.THRESH_BINARY) #  + cv2.THRESH_OTSU
    # cv2.imwrite("gradient.png", inv_grad)

    # opening = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=2)
    # dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
    sure_fg = cv2.erode(inv_grad, np.ones((3, 3), np.uint8), iterations=1)
    sure_fg = sure_fg.astype(np.uint8)

    # Background
    sure_bg = cv2.dilate(inv_grad, np.ones((3, 3), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # cv2.imwrite("sure_fg.png", sure_fg)
    # cv2.imwrite("sure_bg.png", sure_bg)

    # Markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    image_ws = image.copy()
    cv2.watershed(image_ws, markers)

    return markers


def proposed_watershed_image(image):  #best:(3, 3, 0)

    # Step 1: VWME Preprocessing
    preprocessed = vwme_filter(image, window_size=3, epsilon=1e-5)
    cv2.imwrite("preprocessed.png", preprocessed)
    # preprocessed = image.copy()

    # Step 2: RGB Gradient Filtering
    # gradient = rgb_gradient_filter(preprocessed, window_size=3)
    # gradient = sobel_rgb_gradient(preprocessed, ksize=3)
    gradient = rgb_canny(preprocessed, 70, 100, 'add')
    cv2.imwrite("gradient.png", gradient)

    # Step 3: Eliminate small local minima
    filtered_gradient = eliminate_small_local_minima(gradient, threshold=0)

    # Step 4: Watershed Transform
    markers = apply_watershed(preprocessed, filtered_gradient)

    # Visualize boundaries
    result = image.copy()
    result[markers == -1] = [0, 0, 255]  # Mark watershed boundaries in red
    cv2.imwrite("watershed_result.png", result)

    return markers


# image = cv2.imread("background.png")  # RGB image



def region_coverage_testing(mask_s, watershed_markers, beta=0.5):
    """
    Perform region coverage testing.

    Parameters:
    - mask_s: Binary foreground mask (MaskS), shape (H, W)
    - watershed_markers: Labeled watershed regions (output of cv2.watershed), shape (H, W)
    - beta: Threshold for foreground coverage

    Returns:
    - mask_w: Refined binary mask (MaskW), shape (H, W)
    """
    mask_w = np.zeros_like(mask_s, dtype=np.uint8)

    # Get all region labels (exclude background 1 and boundary -1)
    region_labels = np.unique(watershed_markers)
    region_labels = region_labels[(region_labels > 1)]  # ignore background (1), and boundary (-1)

    for label in region_labels:
        region_mask = (watershed_markers == label)
        region_area = np.sum(region_mask)
        if region_area == 0:
            continue

        foreground_area = np.sum(mask_s[region_mask] > 0)
        coverage = foreground_area / region_area

        if coverage >= beta:
            mask_w[region_mask] = mask_s[region_mask]  # Mark region as foreground

    mask_w = cv2.dilate(mask_w, np.ones((3, 3), np.uint8), iterations=1)
    mask_w = cv2.erode(mask_w, np.ones((3, 3), np.uint8), iterations=1)

    return mask_w



from tqdm import tqdm
def process_video(maskS_path, original_rgb_path, output_path, beta=0.5):
    cap_mask = cv2.VideoCapture(maskS_path)
    cap_rgb = cv2.VideoCapture(original_rgb_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    width = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    frame_count = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=frame_count, desc="Processing video")

    while cap_rgb.isOpened():
        ret_rgb, frame_rgb = cap_rgb.read()
        ret_mask, frame_mask = cap_mask.read()

        if not ret_rgb or not ret_mask:
            break

        mask_gray = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
        _, mask_s = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)

        markers = proposed_watershed_image(frame_rgb)
        mask_w = region_coverage_testing(mask_s, markers, beta=beta)

        out.write(mask_w)
        cv2.imwrite("maskW.png", mask_w)
        pbar.update(1)

    cap_rgb.release()
    cap_mask.release()
    out.release()
    pbar.close()
    print("Saved output to:", output_path)

# Define this function at module level (outside any other functions)
def process_frames_batch(frame_indices, maskS, original_rgb_path, beta):
    """Process a batch of frames for watershed segmentation"""
    # Create a local video capture object for this process
    local_cap = cv2.VideoCapture(original_rgb_path)
    results = []
    
    for idx in frame_indices:
        if idx >= len(maskS):
            continue
            
        # Set frame position
        local_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_rgb = local_cap.read()
        
        if not ret:
            results.append((idx, np.zeros_like(maskS[0])))
            continue
        
        # Get corresponding mask
        mask_s = maskS[idx]
        
        # Process frame
        markers = proposed_watershed_image(frame_rgb)
        mask_w = region_coverage_testing(mask_s, markers, beta=beta)
        
        # Perform dilation to fill border lines
        mask_w = cv2.dilate(mask_w, np.ones((3, 3), np.uint8), iterations=1)
        
        # Store result with its index
        results.append((idx, mask_w))
    
    local_cap.release()
    return results

# Then modify the watershed_video function to use this external function
def watershed_video(maskS, original_rgb_path, beta=0.5, num_processes=None):
    """Parallel implementation of watershed video processing using multiprocessing."""
    import multiprocessing as mp
    from functools import partial
    from tqdm import tqdm
    
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Get video properties
    cap = cv2.VideoCapture(original_rgb_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Create output array
    maskW = np.zeros_like(maskS)
    
    # Split frames into batches
    num_frames = min(len(maskS), total_frames)
    frames_per_process = max(1, num_frames // num_processes)
    frame_batches = [
        list(range(i, min(i + frames_per_process, num_frames))) 
        for i in range(0, num_frames, frames_per_process)
    ]
    
    print(f"Processing {num_frames} frames using {num_processes} processes...")
    
    # Create a process pool and process batches in parallel
    with mp.Pool(processes=num_processes) as pool:
        # Create a partial function with fixed arguments
        process_func = partial(
            process_frames_batch,  # Now referencing the global function
            maskS=maskS, 
            original_rgb_path=original_rgb_path, 
            beta=beta
        )
        
        # Process batches and show progress
        all_results = []
        with tqdm(total=num_frames, desc="Processing frames") as pbar:
            for batch_results in pool.imap_unordered(process_func, frame_batches):
                all_results.extend(batch_results)
                pbar.update(len(batch_results))
    
    # Populate result array using frame indices
    for idx, mask_w in all_results:
        maskW[idx] = mask_w
    
    return maskW

# process_video(
#     maskS_path="Results/output2.mp4",            # Binary shadow-cancelled mask video
#     original_rgb_path="Results/preprocessed/preprocessed_test2.mp4",  # Original RGB video (needed for watershed gradient)
#     output_path="Results/maskW_output2.mp4",          # Final output with refined masks
#     beta=0.4                                 # Region coverage threshold
# )