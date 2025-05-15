import cv2
import numpy as np
import os
from tqdm import tqdm

def preprocess_video(
        input_path, 
        output_path=None, 
        horizontal=True, 
        target_width=426, 
        target_height=240, 
        target_fps=15,
        start_time=None,  # In seconds
        end_time=None,    # In seconds
        start_frame=None, # For backward compatibility
        end_frame=None    # For backward compatibility
    ):
    """
    Preprocess a video file by resizing to target resolution, converting to target fps,
    and optionally selecting a specific time segment.
    
    Args:
        input_path (str): Path to the input video file
        output_path (str, optional): Path to save the processed video. If None, 
                                    will use input filename with "_processed" suffix
        horizontal (bool): If True, maintain horizontal orientation; if False, swap dimensions
        target_width (int): Target width (default: 426 for 16:9 aspect ratio at 240p)
        target_height (int): Target height (default: 240)
        target_fps (int): Target frame rate (default: 15 fps)
        start_time (float, optional): Start time in seconds. If None, starts from beginning.
        end_time (float, optional): End time in seconds. If None, processes to end.
        start_frame (int, optional): Deprecated. First frame to include (0-indexed).
        end_frame (int, optional): Deprecated. Last frame to include (0-indexed, inclusive).
    
    Returns:
        str: Path to the processed video file
    """
    # If horizontal is False, swap width and height
    if not horizontal:
        target_width, target_height = target_height, target_width

    # Generate output path if not provided
    if output_path is None:
        filename, ext = os.path.splitext(input_path)
        output_path = f"{filename}_processed{ext}"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    # Get original video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frame_count / orig_fps if orig_fps > 0 else 0
    
    # Convert time parameters to frames if provided
    if start_time is not None:
        start_frame = int(start_time * orig_fps)
    if end_time is not None:
        end_frame = int(end_time * orig_fps)
    
    # Validate and adjust frame range
    if start_frame is None:
        start_frame = 0
    else:
        start_frame = max(0, min(start_frame, total_frame_count - 1))
    
    if end_frame is None:
        end_frame = total_frame_count - 1
    else:
        end_frame = max(start_frame, min(end_frame, total_frame_count - 1))
    
    # Convert frames back to times for display
    start_time = start_frame / orig_fps
    end_time = end_frame / orig_fps
    
    # Calculate effective frame count for processing
    frame_count = end_frame - start_frame + 1
    
    print(f"Input video: {orig_width}x{orig_height} @ {orig_fps:.2f}fps, {total_frame_count} frames ({duration:.2f}s)")
    print(f"Processing time segment: {start_time:.2f}s to {end_time:.2f}s")
    print(f"Processing frames {start_frame} to {end_frame} ({frame_count} frames)")
    print(f"Target: {target_width}x{target_height} @ {target_fps}fps")
    
    # Calculate frame sampling rate based on original and target fps
    frame_sample_interval = max(1, int(orig_fps / target_fps))
    estimated_output_frames = frame_count // frame_sample_interval
    
    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
    out = cv2.VideoWriter(
        output_path, 
        fourcc, 
        target_fps, 
        (target_width, target_height)
    )
    
    if not out.isOpened():
        raise ValueError(f"Could not create output video file: {output_path}")
    
    # Seek to start frame if needed
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Process the video
    frame_idx = start_frame
    processed_frames = 0
    
    print(f"Processing video...")
    with tqdm(total=frame_count, desc="Processing frames") as pbar:
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame based on target fps
            if (frame_idx - start_frame) % frame_sample_interval == 0:
                # Resize the frame
                resized_frame = cv2.resize(frame, (target_width, target_height), 
                                          interpolation=cv2.INTER_LINEAR)
                
                # Write the frame
                out.write(resized_frame)
                processed_frames += 1
            
            frame_idx += 1
            pbar.update(1)
            
            # If we've reached the end frame, stop processing
            if frame_idx > end_frame:
                break
    
    # Release resources
    cap.release()
    out.release()
    
    # Calculate output duration
    output_duration = processed_frames / target_fps
    
    print(f"\nVideo processing complete. Saved to {output_path}")
    print(f"Selected: {frame_count} frames ({(end_time-start_time):.2f}s), Processed: {processed_frames} frames ({output_duration:.2f}s)")
    
    return output_path