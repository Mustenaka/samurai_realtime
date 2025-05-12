import os
import numpy as np
import torch
import time
import cv2
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque
import copy


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


@dataclass
class TrackedObject:
    """Class to store information about a tracked object."""
    object_id: int  # Unique ID for this object
    initial_position: Tuple[int, int]  # Initial click position (x, y)
    last_mask: Optional[np.ndarray] = None  # Most recent mask
    color: Tuple[int, int, int] = None  # Color for visualization
    
    def __post_init__(self):
        """Initialize random color if none provided."""
        if self.color is None:
            # Generate a random bright color
            self.color = (
                np.random.randint(100, 256),
                np.random.randint(100, 256),
                np.random.randint(100, 256)
            )


class MultiObjectTracker:
    """Manages tracking of multiple objects using SAM2."""
    
    def __init__(self, model_size: str = "large", max_objects: int = 5):
        """
        Initialize the multi-object tracker.
        
        Args:
            model_size: Either "small" (faster) or "large" (better quality)
            max_objects: Maximum number of objects to track simultaneously
        """
        # Set Apple MPS fallback for unsupported operations
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        # Initialize device
        self.device = DeviceManager.get_device()
        DeviceManager.configure_device(self.device)
        
        # Try to suppress warnings
        import warnings
        warnings.filterwarnings("ignore", message="cannot import name '_C'")
        
        # Import SAM2 here to ensure device is configured first
        from sam2.build_sam import build_sam2_camera_predictor
        self.build_predictor = build_sam2_camera_predictor
        
        # Model configuration
        if model_size.lower() == "small":
            self.sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt"
            self.model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
            print("Using small model (faster, 6-9 FPS)")
        else:
            self.sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
            self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            print("Using large model (better quality, ~5 FPS)")
            
        # Tracking state
        self.tracked_objects = {}  # Dict mapping object_id to TrackedObject
        self.next_object_id = 1
        self.max_objects = max_objects
        
        # Initialize separate predictor for each potential object
        self.predictors = {}
        self.first_frame = None
        
        # Debug mode
        self.debug = True
        
    def _create_new_predictor(self):
        """Create a new SAM2 predictor instance."""
        return self.build_predictor(
            self.model_cfg, 
            self.sam2_checkpoint, 
            device=self.device
        )
    
    def set_first_frame(self, frame: np.ndarray) -> None:
        """Store the first frame for initializing new trackers."""
        self.first_frame = frame.copy()
    
    def add_object(self, frame: np.ndarray, x: int, y: int) -> int:
        """
        Start tracking a new object at the specified position.
        
        Args:
            frame: Current video frame
            x, y: Point coordinates where user clicked
        
        Returns:
            ID of the newly tracked object
        """
        # Store first frame if not already stored
        if self.first_frame is None:
            self.set_first_frame(frame)
        
        # Check if we've reached the maximum number of objects
        if len(self.tracked_objects) >= self.max_objects:
            print(f"Maximum number of tracked objects ({self.max_objects}) reached.")
            return -1
            
        # Create a new object ID
        object_id = self.next_object_id
        self.next_object_id += 1
        
        try:
            # Create a new predictor for this object
            predictor = self._create_new_predictor()
            predictor.load_first_frame(self.first_frame)
            
            # Add the initial point prompt
            point = np.array([[x, y]], dtype=np.float32)
            label = np.array([1], dtype=np.int32)
            
            try:
                # Try to add bbox instead of point if having issues with point prompts
                # Calculate a small box around the clicked point
                box_size = min(frame.shape[0], frame.shape[1]) // 10
                x1 = max(0, x - box_size//2)
                y1 = max(0, y - box_size//2)
                x2 = min(frame.shape[1]-1, x + box_size//2)
                y2 = min(frame.shape[0]-1, y + box_size//2)
                bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                
                # Try both point and bbox prompts
                try:
                    _, _, mask_logits = predictor.add_new_prompt(
                        frame_idx=0,
                        obj_id=1,  # Always use ID 1 within each predictor
                        points=point,
                        labels=label
                    )
                except Exception as e:
                    print(f"Point prompt failed, trying bbox: {e}")
                    _, _, mask_logits = predictor.add_new_prompt(
                        frame_idx=0,
                        obj_id=1,
                        bbox=bbox
                    )
            except Exception as e:
                print(f"Failed to add prompt for object {object_id}: {e}")
                # Return -1 if we couldn't add the prompt
                return -1
            
            # Store the predictor
            self.predictors[object_id] = predictor
            
            # Get initial mask
            mask = None
            if mask_logits is not None and len(mask_logits) > 0:
                try:
                    mask_tensor = mask_logits[0] > 0.0
                    if mask_tensor.numel() > 0:
                        mask = mask_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
                        
                        # Check if mask is valid (not empty or full-screen)
                        mask_sum = np.sum(mask)
                        total_pixels = mask.shape[0] * mask.shape[1]
                        
                        if mask_sum == 0:
                            print(f"Warning: Empty mask for object {object_id}")
                            # Create a small circular mask around the clicked point
                            mask = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                            cv2.circle(mask, (x, y), box_size//2, 255, -1)
                        elif mask_sum >= 0.95 * total_pixels:
                            print(f"Warning: Full screen mask for object {object_id}")
                            # Create a smaller mask around the clicked point
                            mask = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                            cv2.circle(mask, (x, y), box_size, 255, -1)
                except Exception as e:
                    print(f"Error processing initial mask: {e}")
                    # Create a fallback mask if we couldn't process the one from the model
                    mask = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                    cv2.circle(mask, (x, y), box_size, 255, -1)
            
            # If we still couldn't get a mask, create a simple circular one
            if mask is None:
                mask = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
                cv2.circle(mask, (x, y), box_size, 255, -1)
            
            # Create and store tracked object
            tracked_object = TrackedObject(
                object_id=object_id,
                initial_position=(x, y),
                last_mask=mask
            )
            self.tracked_objects[object_id] = tracked_object
            
            print(f"Started tracking object {object_id} at position ({x}, {y})")
            return object_id
            
        except Exception as e:
            print(f"Failed to create tracking for object at ({x}, {y}): {e}")
            return -1
    
    def refine_object(self, object_id: int, x: int, y: int, positive: bool = False) -> bool:
        """
        Refine the tracking of an existing object by adding a positive or negative point.
        
        Args:
            object_id: ID of the object to refine
            x, y: Point coordinates
            positive: True for positive point, False for negative point
        
        Returns:
            True if refinement was successful, False otherwise
        """
        if object_id not in self.tracked_objects or object_id not in self.predictors:
            print(f"Object with ID {object_id} not found")
            return False
        
        predictor = self.predictors[object_id]
        
        # Add the refinement point
        point = np.array([[x, y]], dtype=np.float32)
        label = np.array([1 if positive else 0], dtype=np.int32)
        
        try:
            # Reset the predictor state
            predictor.reset_state()
            predictor.load_first_frame(self.first_frame)
            
            # Add the initial point (the one used to create this object)
            initial_x, initial_y = self.tracked_objects[object_id].initial_position
            initial_point = np.array([[initial_x, initial_y]], dtype=np.float32)
            initial_label = np.array([1], dtype=np.int32)
            
            predictor.add_new_prompt(
                frame_idx=0,
                obj_id=1,
                points=initial_point,
                labels=initial_label
            )
            
            # Add the refinement point
            predictor.add_new_prompt(
                frame_idx=0,
                obj_id=1,
                points=point,
                labels=label
            )
            
            print(f"Added {'positive' if positive else 'negative'} point at ({x}, {y}) to refine object {object_id}")
            return True
            
        except Exception as e:
            print(f"Error refining object {object_id}: {e}")
            return False
    
    def remove_object(self, object_id: int) -> bool:
        """
        Stop tracking an object.
        
        Args:
            object_id: ID of the object to remove
        
        Returns:
            True if removal was successful, False otherwise
        """
        if object_id not in self.tracked_objects:
            return False
        
        # Remove the object and its predictor
        del self.tracked_objects[object_id]
        if object_id in self.predictors:
            del self.predictors[object_id]
            
        print(f"Stopped tracking object {object_id}")
        return True
    
    def track_all_objects(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Track all objects in the current frame.
        
        Args:
            frame: Current video frame
        
        Returns:
            Dictionary mapping object IDs to their mask arrays
        """
        results = {}
        
        for object_id, tracked_object in list(self.tracked_objects.items()):
            if object_id not in self.predictors:
                continue
                
            predictor = self.predictors[object_id]
            
            try:
                # Track the object
                obj_ids, mask_logits = predictor.track(frame)
                
                # Check if tracking was successful
                if mask_logits is not None and len(mask_logits) > 0:
                    # More careful handling of mask creation
                    try:
                        # Get mask and ensure it has the right shape
                        mask = (mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy()
                        
                        # Validate mask shape and content
                        h, w = frame.shape[:2]
                        if mask.shape[:2] != (h, w):
                            print(f"Warning: Mask shape {mask.shape[:2]} doesn't match frame {(h, w)}")
                            # Resize mask if needed
                            mask_resized = np.zeros((h, w, 1), dtype=np.uint8)
                            min_h = min(mask.shape[0], h)
                            min_w = min(mask.shape[1], w)
                            mask_resized[:min_h, :min_w, :] = mask[:min_h, :min_w, :]
                            mask = mask_resized
                        
                        # Check if mask is empty or full
                        mask_sum = np.sum(mask)
                        total_pixels = h * w
                        if mask_sum == 0:
                            print(f"Warning: Empty mask for object {object_id}")
                            continue
                        elif mask_sum >= 0.95 * total_pixels:
                            print(f"Warning: Full screen mask for object {object_id}")
                            # Apply a simple filter - keep only center region
                            center_mask = np.zeros_like(mask)
                            ch, cw = h//2, w//2
                            size = min(h, w) // 4
                            center_mask[ch-size:ch+size, cw-size:cw+size, :] = 1
                            mask = mask * center_mask
                        
                        # Convert to final format
                        mask = mask.astype(np.uint8) * 255
                        tracked_object.last_mask = mask
                        results[object_id] = mask
                    except Exception as e:
                        print(f"Error processing mask for object {object_id}: {e}")
                    
            except Exception as e:
                print(f"Error tracking object {object_id}: {e}")
                # Continue tracking other objects even if one fails
        
        return results
        
    def get_combined_mask(self, frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Create a visualization with all tracked object masks.
        
        Args:
            frame: Current video frame
            alpha: Transparency of the overlay (0.0 to 1.0)
            
        Returns:
            Frame with colored mask overlays for all tracked objects
        """
        # Create a copy of the frame for overlay
        height, width = frame.shape[:2]
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add each object's mask with its color
        for object_id, tracked_object in self.tracked_objects.items():
            if tracked_object.last_mask is None:
                continue
            
            try:
                # Ensure mask has correct shape
                mask = tracked_object.last_mask
                if mask.shape[0] != height or mask.shape[1] != width:
                    if self.debug:
                        print(f"Resizing mask from {mask.shape[:2]} to {(height, width)}")
                    # Create a properly sized mask
                    resized_mask = np.zeros((height, width, mask.shape[2]), dtype=mask.dtype)
                    # Copy what we can
                    h = min(mask.shape[0], height)
                    w = min(mask.shape[1], width)
                    resized_mask[:h, :w] = mask[:h, :w]
                    mask = resized_mask
                
                # Ensure mask has the right number of channels
                if mask.shape[2] != 1:
                    if self.debug:
                        print(f"Converting mask from {mask.shape[2]} channels to 1 channel")
                    # Take the first channel or average if needed
                    if mask.shape[2] == 3:
                        mask = np.mean(mask, axis=2, keepdims=True).astype(mask.dtype)
                    else:
                        mask = mask[:, :, :1]
                
                # Create a colored mask for this object
                color_mask = np.zeros((height, width, 3), dtype=np.uint8)
                for i in range(3):  # Apply color to each channel
                    color_mask[:, :, i] = mask[:, :, 0] * tracked_object.color[i] // 255
                
                # Add to the overlay
                overlay = cv2.addWeighted(overlay, 1.0, color_mask, 1.0, 0)
            
            except Exception as e:
                if self.debug:
                    print(f"Error creating colored mask for object {object_id}: {e}")
        
        # Combine with the original frame
        return cv2.addWeighted(frame, 1.0, overlay, alpha, 0)


class CameraApp:
    """Main application for webcam-based multi-object tracking with SAM2."""
    
    def __init__(self, model_size: str = "large", camera_id: int = 0, max_objects: int = 5):
        """
        Initialize the camera application.
        
        Args:
            model_size: Either "small" (faster) or "large" (better quality)
            camera_id: ID of the camera to use
            max_objects: Maximum number of objects to track simultaneously
        """
        self.tracker = MultiObjectTracker(model_size=model_size, max_objects=max_objects)
        self.cap = cv2.VideoCapture(camera_id)
        self.window_name = "SAM2 Multi-Object Tracking"
        self.prev_frame_time = 0
        self.current_frame = None
        
        # Track which object the user is interacting with
        self.selected_object_id = None
        
        # Mode can be "add" (left click adds new object) or "refine" (left click refines existing object)
        self.interaction_mode = "add"
        
        # Store click positions for visualization
        self.click_positions = []
        self.max_stored_positions = 10  # Maximum number of click positions to remember
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for interaction."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click: Add new object or refine current selection
            if self.interaction_mode == "add":
                # Add new object
                object_id = self.tracker.add_object(self.current_frame, x, y)
                if object_id > 0:
                    self.selected_object_id = object_id
                    # Store the click position
                    self.click_positions.append((x, y, object_id))
                    if len(self.click_positions) > self.max_stored_positions:
                        self.click_positions.pop(0)
            else:
                # Refine current selection with positive point
                if self.selected_object_id is not None:
                    self.tracker.refine_object(self.selected_object_id, x, y, positive=True)
                    # Store the click position
                    self.click_positions.append((x, y, self.selected_object_id))
                    if len(self.click_positions) > self.max_stored_positions:
                        self.click_positions.pop(0)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click: Add negative point to refine tracking or remove object
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                # With CTRL: Remove the nearest object to the click
                nearest_id = self.find_nearest_object(x, y)
                if nearest_id is not None:
                    self.tracker.remove_object(nearest_id)
                    # Remove click positions for this object
                    self.click_positions = [p for p in self.click_positions if p[2] != nearest_id]
                    if self.selected_object_id == nearest_id:
                        self.selected_object_id = None
            else:
                # Without CTRL: Add negative point
                if self.selected_object_id is not None:
                    self.tracker.refine_object(self.selected_object_id, x, y, positive=False)
                    # Store the click position (with negative marker)
                    self.click_positions.append((x, y, -self.selected_object_id))  # Negative ID indicates negative point
                    if len(self.click_positions) > self.max_stored_positions:
                        self.click_positions.pop(0)
    
    def find_nearest_object(self, x: int, y: int) -> Optional[int]:
        """Find the ID of the object closest to the given coordinates."""
        min_distance = float('inf')
        nearest_id = None
        
        for object_id, tracked_object in self.tracker.tracked_objects.items():
            initial_x, initial_y = tracked_object.initial_position
            distance = ((x - initial_x) ** 2 + (y - initial_y) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                nearest_id = object_id
        
        return nearest_id
    
    def calculate_fps(self) -> float:
        """Calculate the current frames per second."""
        current_time = cv2.getTickCount()
        time_diff = (current_time - self.prev_frame_time) / cv2.getTickFrequency()
        fps = 1 / time_diff if time_diff > 0 else 0
        self.prev_frame_time = current_time
        return fps
    
    def draw_click_positions(self, frame: np.ndarray) -> np.ndarray:
        """Draw the stored click positions on the frame."""
        result = frame.copy()
        
        for x, y, obj_id in self.click_positions:
            # Determine color and shape based on point type
            if obj_id < 0:  # Negative point
                color = (0, 0, 255)  # Red for negative points
                radius = 5
                obj_id = -obj_id  # Get the actual object ID
            else:  # Positive point
                # Use the object's assigned color
                if obj_id in self.tracker.tracked_objects:
                    color = self.tracker.tracked_objects[obj_id].color
                else:
                    color = (0, 255, 0)  # Green fallback
                radius = 5
            
            # Draw the circle
            cv2.circle(result, (x, y), radius, color, -1)
            
            # Add object ID text
            cv2.putText(result, str(obj_id), (x + 7, y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 1, cv2.LINE_AA)
        
        return result
    
    def draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add information overlay to the frame."""
        result = frame.copy()
        
        # Add FPS counter
        fps = self.calculate_fps()
        cv2.putText(result, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add object count
        num_objects = len(self.tracker.tracked_objects)
        cv2.putText(result, f"Objects: {num_objects}/{self.tracker.max_objects}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add selected object indicator
        if self.selected_object_id is not None:
            cv2.putText(result, f"Selected: {self.selected_object_id}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add mode indicator
        mode_text = "Mode: ADD" if self.interaction_mode == "add" else "Mode: REFINE"
        cv2.putText(result, mode_text, (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process the current frame for display."""
        try:
            # Convert to RGB for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = rgb_frame.copy()
            
            # Set first frame if not already set
            if self.tracker.first_frame is None:
                self.tracker.set_first_frame(rgb_frame)
            
            # Track all objects
            self.tracker.track_all_objects(rgb_frame)
            
            # Get combined visualization
            result = self.tracker.get_combined_mask(rgb_frame)
            
            # Draw click positions
            result = self.draw_click_positions(result)
            
            # Add information overlay
            result = self.draw_info_overlay(result)
            
            # Convert back to BGR for display
            return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error in process_frame: {e}")
            # Return original frame if processing fails
            return frame
    
    def toggle_mode(self):
        """Toggle between 'add' and 'refine' interaction modes."""
        if self.interaction_mode == "add":
            self.interaction_mode = "refine"
        else:
            self.interaction_mode = "add"
        print(f"Switched to {self.interaction_mode.upper()} mode")
    
    def run(self):
        """Run the main application loop."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("Starting multi-object tracking application.")
        print("\nControls:")
        print("  Left-click: Add new object or add positive point in refine mode")
        print("  Right-click: Add negative point to refine current selection")
        print("  CTRL + Right-click: Remove the nearest object")
        print("  Press 'm': Toggle between ADD and REFINE modes")
        print("  Press 'q': Quit the application")
        print("\nCurrently in ADD mode - left click to start tracking objects")
        
        self.prev_frame_time = cv2.getTickCount()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame from camera")
                    break
                
                processed_frame = self.process_frame(frame)
                cv2.imshow(self.window_name, processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.toggle_mode()
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Application closed")


if __name__ == "__main__":
    # You can choose "small" for faster processing or "large" for better quality
    app = CameraApp(model_size="small", camera_id=0, max_objects=5)
    app.run()