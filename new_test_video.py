import os
# If using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageChops
import cv2

def setup_device():
    """Set up and return the computational device based on availability."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        # Use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # Turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    
    return device

def show_mask(mask, image, show=False):
    """
    Overlay a colored mask on an image.
    
    Args:
        mask: Binary mask array
        image: PIL Image to overlay the mask on
        show: Whether to display the image immediately
        
    Returns:
        PIL Image with the mask overlaid
    """
    # Create a color for consistency (random color)
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    
    # Create ImageDraw object
    draw = ImageDraw.Draw(image)
    
    # Get mask dimensions
    h, w = mask.shape[-2:]
    
    # Create a colored mask image
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    # Convert mask_image to Pillow image
    mask_image = (mask_image * 255).astype(np.uint8)  # Pillow images need 0-255 integer values
    mask_pil_image = Image.fromarray(mask_image)
    
    # Convert to RGBA
    image = image.convert("RGBA")
    
    # Blend using alpha composition
    mask_with_image_background = Image.composite(mask_pil_image, image, mask_pil_image.split()[3])
    blend_image = ImageChops.multiply(image, mask_with_image_background)
    
    if show:
        blend_image.show()
        
    return blend_image

def show_points(coords, labels, image, marker_size=200, show=False):
    """
    Draw positive and negative points on an image.
    
    Args:
        coords: Numpy array of point coordinates
        labels: Numpy array of point labels (1 for positive, 0 for negative)
        image: PIL Image to draw points on
        marker_size: Size of the point markers
        show: Whether to display the image immediately
        
    Returns:
        PIL Image with points drawn on it
    """
    # Create a copy of the image
    copy_image = image.copy()
    draw = ImageDraw.Draw(copy_image)

    # Calculate radius from marker size
    radius = int(marker_size ** 0.5)
    
    # Extract positive and negative points based on labels
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    
    # Draw positive points (green circles with white outline)
    for point in pos_points:
        x, y = int(point[0]), int(point[1])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='green', outline='white')

    # Draw negative points (red circles with white outline)
    for point in neg_points:
        x, y = int(point[0]), int(point[1])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red', outline='white')

    if show:
        copy_image.show()
    
    return copy_image

def load_video_frames(video_dir):
    """
    Load all JPEG frames from a directory.
    
    Args:
        video_dir: Directory containing JPEG frames
        
    Returns:
        Sorted list of frame filenames
    """
    # Scan all JPEG frame names in the directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    # Sort frames by numerical index
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    return frame_names

def create_video_from_frames(frames, output_filename, fps=20):
    """
    Create a video from a list of PIL Image frames.
    
    Args:
        frames: List of PIL Image objects
        output_filename: Name of the output video file
        fps: Frames per second
    """
    frame_width, frame_height = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create VideoWriter object
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    # Write each frame to the video
    for frame in frames:
        frame_cv2 = np.array(frame)  # Convert PIL image to NumPy array
        frame_cv2 = cv2.cvtColor(frame_cv2, cv2.COLOR_RGBA2BGR)  # Convert to BGR color space
        out.write(frame_cv2)

    # Release VideoWriter object
    out.release()
    print(f"Video saved as {output_filename}")

def main():
    """Main function to run the SAM2 video segmentation pipeline."""
    # Setup device for computation
    device = setup_device()
    
    # Import SAM2 model builder (put import here to avoid loading before device setup)
    from sam2.build_sam import build_sam2_video_predictor
    
    # Load SAM2 model
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    print("Successfully built predictor")
    
    # Set input video directory
    video_dir = "./notebooks/videos/webcam"
    
    # Load video frames
    frame_names = load_video_frames(video_dir)
    
    # Initialize model state
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    
    # Set annotation parameters
    ann_frame_idx = 0  # Frame index we interact with
    ann_obj_id = 1  # Unique ID for each object we interact with
    
    # Define points of interest
    points = np.array([[300, 150]], dtype=np.float32)  # webcam
    labels = np.array([1], np.int32)  # 1 means positive click, 0 means negative click
    
    # Add points to the model
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    
    # Show initial results
    img = Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))
    show_points(points, labels, img, show=True)
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), img, show=True)
    
    # Run propagation throughout the video
    video_segments = {}  # Dictionary to store per-frame segmentation results
    frames = []
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    # Render segmentation results for each frame
    max_frame = min(200, len(frame_names))  # Limit to 200 frames or video length
    
    for out_frame_idx in range(0, max_frame):
        rend_image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            mask_overlay = show_mask(out_mask, rend_image)
            frames.append(mask_overlay)
    
    # Create and save output video
    create_video_from_frames(frames, "output_video.mp4")

if __name__ == "__main__":
    main()