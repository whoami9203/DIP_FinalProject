import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from preprocessing import preprocess_video
from gpu_background_modeling import gpu_k_means_background_clustering
from mask_generation import foreground_segmentation, apply_mask_to_video

def save_background_model(background_model, filepath):
    """
    Save the background model ndarray to a file preserving float32 precision.
    
    Args:
        background_model: numpy ndarray of shape (height, width, 3)
        filepath: Path to save the model (will add .npy if not present)
    """
    # Make sure the extension is correct
    if not filepath.endswith('.npy'):
        filepath += '.npy'
        
    # Save the array directly
    np.save(filepath, background_model)
    print(f"Background model saved to {filepath}")
    
    # Verify the shape and dtype of the saved file
    saved_model = np.load(filepath)
    print(f"Verification - Shape: {saved_model.shape}, dtype: {saved_model.dtype}")

def print_covariances_to_file(background_covariances, filepath="background_covariances.txt"):
    """
    Save the contents of the background covariances array to a text file.
    Includes quartile statistics for each channel.
    
    Args:
        background_covariances: numpy ndarray of shape (height, width, 3)
        filepath: Path to save the text file
    """
    height, width, channels = background_covariances.shape
    
    with open(filepath, 'w') as f:
        f.write(f"Background Covariances - Shape: {background_covariances.shape}\n")
        f.write(f"Min value: {np.min(background_covariances)}, Max value: {np.max(background_covariances)}\n")
        f.write(f"Mean value: {np.mean(background_covariances)}, Std: {np.std(background_covariances)}\n\n")
        
        # Print statistical summary for each channel
        for c in range(channels):
            channel_data = background_covariances[:, :, c]
            q1, q2, q3 = np.percentile(channel_data, [25, 50, 75])
            
            f.write(f"Channel {c} statistics:\n")
            f.write(f"  Min: {np.min(channel_data):.6f}, Max: {np.max(channel_data):.6f}\n")
            f.write(f"  Mean: {np.mean(channel_data):.6f}, Median: {np.median(channel_data):.6f}\n")
            f.write(f"  Std: {np.std(channel_data):.6f}\n")
            f.write(f"  Quartiles: Q1={q1:.6f}, Q2={q2:.6f}, Q3={q3:.6f}\n")
            f.write(f"  Interquartile Range (IQR): {(q3-q1):.6f}\n\n")
        
        # Print a sample of values (center region)
        sample_y_start = height // 2 - 2
        sample_y_end = height // 2 + 3
        sample_x_start = width // 2 - 2
        sample_x_end = width // 2 + 3
        
        f.write("Sample values from center region:\n")
        f.write("Format: [R variance, G variance, B variance]\n\n")
        
        for y in range(sample_y_start, sample_y_end):
            for x in range(sample_x_start, sample_x_end):
                f.write(f"Position ({y}, {x}): [{background_covariances[y, x, 0]:.6f}, "
                        f"{background_covariances[y, x, 1]:.6f}, {background_covariances[y, x, 2]:.6f}]\n")
        
        # Print histogram information
        f.write("\nHistogram data (10 bins):\n")
        for c in range(channels):
            channel_data = background_covariances[:, :, c].flatten()
            hist, bins = np.histogram(channel_data, bins=10)
            f.write(f"Channel {c} histogram:\n")
            for i in range(len(hist)):
                f.write(f"  Bin {i} ({bins[i]:.6f} to {bins[i+1]:.6f}): {hist[i]} values\n")
            f.write("\n")

    print(f"Covariances data written to {filepath}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Video background modeling and foreground segmentation")
    
    # Input/output paths
    parser.add_argument("--input", "-i", type=str, default="Video/test2.mp4",
                        help="Path to input video file")
    parser.add_argument("--output", "-o", type=str, default="Results/masked_video.mp4",
                        help="Path to output video file")
    parser.add_argument("--model-dir", type=str, default=".",
                        help="Directory to save/load background models")
                        
    # Processing options
    parser.add_argument("--preprocess", action="store_true",
                        help="Enable video preprocessing (default: off)")
    parser.add_argument("--resolution", type=str, default="854x480",
                        help="Target resolution for preprocessing (WxH)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Target FPS for preprocessing")
    parser.add_argument("--start-time", type=float, default=0,
                        help="Start time in seconds for preprocessing")
    parser.add_argument("--end-time", type=float, default=30,
                        help="End time in seconds for preprocessing")
                        
    # Algorithm parameters
    parser.add_argument("--alpha", type=float, default=37.0,
                        help="Alpha parameter for Mahalanobis distance threshold")
    parser.add_argument("--max-k", type=int, default=3,
                        help="Maximum number of clusters for K-means")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Force recomputation of background model even if saved model exists")
    parser.add_argument("--darkness-factor", type=float, default=0.0,
                        help="Darkness factor for background (0=black, 1=original)")
                        
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Parse resolution
    try:
        target_width, target_height = map(int, args.resolution.split('x'))
    except:
        print(f"Invalid resolution format: {args.resolution}, using default 854x480")
        target_width, target_height = 854, 480
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Define model paths
    means_model_path = os.path.join(args.model_dir, "background_means_model.npy")
    covariances_model_path = os.path.join(args.model_dir, "background_covariances_model.npy")
    
    # Preprocessing
    VIDEO_PATH = args.input
    if args.preprocess:
        preprocessed_dir = os.path.join(os.path.dirname(args.output), "preprocessed")
        os.makedirs(preprocessed_dir, exist_ok=True)
        
        input_filename = os.path.basename(args.input)
        PREPROCESSED_VIDEO_PATH = os.path.join(preprocessed_dir, f"preprocessed_{input_filename}")
        
        # Preprocess the video
        print(f"Preprocessing video to {target_width}x{target_height} at {args.fps} FPS...")
        preprocess_video(input_path=VIDEO_PATH, output_path=PREPROCESSED_VIDEO_PATH, 
                         horizontal=True, target_width=target_width, target_height=target_height, 
                         target_fps=args.fps, start_time=args.start_time, end_time=args.end_time)
        VIDEO_PATH = PREPROCESSED_VIDEO_PATH

    # Check if the model already exists
    if not args.force_recompute and os.path.exists(means_model_path) and os.path.exists(covariances_model_path):
        print("Background model already exists. Loading...")
        background_means = np.load(means_model_path)
        background_covariances = np.load(covariances_model_path)

        # Print covariances to file for inspection
        print_covariances_to_file(background_covariances, 
                                 os.path.join(args.model_dir, "background_covariances.txt"))
    else:
        # Perform k-means clustering on the video
        print("Performing k-means clustering on the video...")
        background_means, background_covariances = gpu_k_means_background_clustering(
            VIDEO_PATH, max_k=args.max_k)

        # Save the background array (float32) directly to a file
        save_background_model(background_means, means_model_path)
        save_background_model(background_covariances, covariances_model_path)

        # Save the background image
        background_image_path = os.path.join(args.model_dir, "background.png")
        cv2.imwrite(background_image_path, background_means)
        print(f"Background image saved to {background_image_path}")

    # Generate the foreground mask
    print(f"Segmenting foreground with alpha={args.alpha}...")
    foreground_masks, diff_stats = foreground_segmentation(
        VIDEO_PATH, background_means, background_covariances, alpha=args.alpha)
    
    # Print global statistics
    print("Difference magnitude statistics:")
    print(f"Mean variance: {diff_stats['global']['variance_mean']:.4f}")
    print(f"Median of all pixels: {diff_stats['global']['q2_mean']:.4f}")
    print(f"q1 median: {diff_stats['global']['q1_median']:.4f}")
    print(f"q2 median: {diff_stats['global']['q2_median']:.4f}")
    print(f"q3 median: {diff_stats['global']['q3_median']:.4f}")
    print(f"IQR median: {diff_stats['global']['iqr_median']:.4f}")

    # Apply the mask to the video
    print(f"Applying mask to video and saving result to {args.output}...")
    apply_mask_to_video(
        input_video_path=VIDEO_PATH,
        output_video_path=args.output, 
        foreground_masks=foreground_masks,
        darkness_factor=args.darkness_factor
    )
    
    print("Processing complete.")

if __name__ == "__main__":
    main()