import os
import numpy as np
import torch
import time
import cv2
from typing import Tuple, List, Optional, Dict, Any

class DeviceManager:
    """Manages device selection and configuration for computation."""
    
    @staticmethod
    def get_device() -> torch.device:
        """Select the appropriate computation device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        print(f"Using device: {device}")
        return device
    
    @staticmethod
    def configure_device(device: torch.device) -> None:
        """Configure the selected device for optimal performance."""
        if device.type == "cuda":
            # Use bfloat16 for better performance
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # Enable TF32 for Ampere GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )


class SAM2Tracker:
    """Manages SAM2 model initialization and object tracking."""
    
    def __init__(self, model_size: str = "large"):
        """
        Initialize the SAM2 tracker.
        
        Args:
            model_size: Either "small" (faster) or "large" (better quality)
        """
        # Set Apple MPS fallback for unsupported operations
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        # Initialize device
        self.device = DeviceManager.get_device()
        DeviceManager.configure_device(self.device)
        
        # Model configuration
        if model_size.lower() == "small":
            self.sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt"
            self.model_cfg = "configs/samurai/sam2.1_hiera_s.yaml"
            print("Using small model (faster, 6-9 FPS)")
        else:
            self.sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
            self.model_cfg = "configs/samurai/sam2.1_hiera_l.yaml"
            print("Using large model (better quality, ~5 FPS)")
        
        # Import and build predictor
        from sam2.build_sam import build_sam2_camera_predictor
        self.predictor = build_sam2_camera_predictor(
            self.model_cfg, 
            self.sam2_checkpoint, 
            device=self.device
        )
        print("Successfully built SAM2 predictor")
        
        # Tracking state
        self.initialized = False
        self.frame_idx = 0     # The frame index we interact with
        self.is_tracking = False  # Flag to indicate if tracking has started
        
        # Tracking Arrays
        self.tracking_obj = [] # ERROR: realtime model problem.
        
        # Fixed ID for all targets (SAM2 seems to only support tracking with a single ID)
        self.obj_id = 1
    
    def load_first_frame(self, frame: np.ndarray) -> None:
        """Load the first frame to initialize tracking."""
        self.predictor.load_first_frame(frame)
        print("First frame loaded")
        self.initialized = True
    
    def reset_state(self):
        """Reset the tracker state to allow adding new prompts."""
        if self.initialized:
            self.predictor.reset_state()
            self.is_tracking = False
            print("Tracker state reset. Ready for new objects.")
    
    def add_point_prompt(self, 
                         frame: np.ndarray, 
                         x: int, 
                         y: int, 
                         positive: bool = True) -> Tuple:
        """
        Add a point prompt to identify an object to track.
        
        Args:
            frame: The current video frame
            x, y: Point coordinates
            positive: True for positive click, False for negative
        
        Returns:
            Tuple of object IDs and mask logits
        """
        # If tracking has already started, reset the state
        if self.is_tracking:
            self.reset_state()
            self.load_first_frame(frame)
        elif not self.initialized:
            self.load_first_frame(frame)
        
        # Add the point prompt
        point = np.array([[x,y]], dtype=np.float32)
        label = np.array([1 if positive else 0], dtype=np.int32)
        
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
            frame_idx=self.frame_idx,
            obj_id=self.obj_id,
            points=point,
            labels=label
        )
        
        self.is_tracking = True
        action_type = "positive" if positive else "negative"
        print(f"Added {action_type} point at ({x}, {y}). Ready to track.")
        return out_obj_ids, out_mask_logits
    
    def add_bbox_prompt(self, 
                        frame: np.ndarray, 
                        x1: int, 
                        y1: int, 
                        x2: int, 
                        y2: int) -> Tuple:
        """
        Add a bounding box prompt to identify an object to track.
        
        Args:
            frame: The current video frame
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
        
        Returns:
            Tuple of object IDs and mask logits
        """
        # If tracking has already started, reset the state
        if self.is_tracking:
            self.reset_state()
            self.load_first_frame(frame)
        elif not self.initialized:
            self.load_first_frame(frame)
        
        # Add the bounding box prompt
        bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
            frame_idx=self.frame_idx,
            obj_id=self.obj_id,
            bbox=bbox
        )
        
        self.is_tracking = True
        print(f"Added bounding box at ({x1}, {y1}, {x2}, {y2}). Ready to track.")
        return out_obj_ids, out_mask_logits
    
    def add_mask_prompt(self, 
                        frame: np.ndarray, 
                        mask_path: str) -> Tuple:
        """
        Add a mask prompt to identify an object to track.
        
        Args:
            frame: The current video frame
            mask_path: Path to mask image
        
        Returns:
            Tuple of object IDs and mask logits
        """
        # If tracking has already started, reset the state
        if self.is_tracking:
            self.reset_state()
            self.load_first_frame(frame)
        elif not self.initialized:
            self.load_first_frame(frame)
        
        # Load and prepare the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask from {mask_path}")
        
        mask = mask / 255.0
        
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
            frame_idx=self.frame_idx,
            obj_id=self.obj_id,
            mask=mask
        )
        
        self.is_tracking = True
        print(f"Added mask from {mask_path}. Ready to track.")
        return out_obj_ids, out_mask_logits
    
    def track(self, frame: np.ndarray) -> Tuple:
        """
        Track objects in the current frame.
        
        Args:
            frame: The current video frame
        
        Returns:
            Tuple of object IDs and mask logits
        """
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Add a prompt first.")
        
        return self.predictor.track(frame)
    
    def visualize_masks(self, 
                        frame: np.ndarray, 
                        obj_ids: List, 
                        mask_logits: List, 
                        alpha: float = 0.5) -> np.ndarray:
        """
        Visualize the tracking masks on the frame.
        
        Args:
            frame: The current video frame
            obj_ids: List of object IDs
            mask_logits: List of mask logits
            alpha: Transparency of the mask overlay
        
        Returns:
            Frame with mask overlay
        """
        height, width = frame.shape[:2]
        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        
        for i in range(len(obj_ids)):
            out_mask = (mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
            all_mask = cv2.bitwise_or(all_mask, out_mask)
        
        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
        return cv2.addWeighted(frame, 1, all_mask, alpha, 0)


class CameraApp:
    """Main application for webcam-based object tracking with SAM2."""
    
    def __init__(self, model_size: str = "large", camera_id: int = 0):
        """
        Initialize the camera application.
        
        Args:
            model_size: Either "small" (faster) or "large" (better quality)
            camera_id: ID of the camera to use
        """
        self.tracker = SAM2Tracker(model_size=model_size)
        self.cap = cv2.VideoCapture(camera_id)
        self.window_name = "SAM2 Object Tracking"
        self.prev_frame_time = 0
        self.current_frame = None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for interaction."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click: Track a new object
            self.tracker.add_point_prompt(self.current_frame, x, y, positive=True)
            print(f"Tracking new object at ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click: Add negative point to refine tracking
            self.tracker.add_point_prompt(self.current_frame, x, y, positive=False)
            print(f"Added negative point at ({x}, {y}) to refine tracking")
    
    def calculate_fps(self) -> float:
        """Calculate the current frames per second."""
        current_time = cv2.getTickCount()
        time_diff = (current_time - self.prev_frame_time) / cv2.getTickFrequency()
        fps = 1 / time_diff if time_diff > 0 else 0
        self.prev_frame_time = current_time
        return fps
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process the current frame for display."""
        # Convert to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame = rgb_frame.copy()
        
        # Track objects if initialized
        if self.tracker.initialized:
            obj_ids, mask_logits = self.tracker.track(rgb_frame)
            rgb_frame = self.tracker.visualize_masks(rgb_frame, obj_ids, mask_logits)
        
        # Calculate and display FPS
        fps = self.calculate_fps()
        cv2.putText(
            rgb_frame, 
            f"FPS: {int(fps)}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Convert back to BGR for display
        return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    def run(self):
        """Run the main application loop."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("Starting camera feed.")
        print("Click on an object with left mouse button to track it.")
        print("If you want to track a different object, click on it with left mouse button.")
        print("Use right mouse button to refine the current tracking by marking areas that should NOT be tracked.")
        print("Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame from camera")
                    break
                
                processed_frame = self.process_frame(frame)
                cv2.imshow(self.window_name, processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Application closed")


if __name__ == "__main__":
    # You can choose "small" for faster processing or "large" for better quality
    app = CameraApp(model_size="large", camera_id=0)
    app.run()