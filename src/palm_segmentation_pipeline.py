"""
PalmSegmentationPipeline orchestrator for coordinating palm detection and segmentation.

This module provides the main pipeline for processing hand images to detect and segment
the palm region. It includes functionality for:
- Processing images through the complete detection pipeline
- Saving output images in PNG format with proper color channels
- Creating side-by-side visualizations of original and processed images
- Displaying output using OpenCV windows
- Visualizing hand landmarks and palm regions for debugging
- Performance timing measurements for each pipeline stage
"""
from typing import Optional, Dict
import numpy as np
import cv2
import os
import time

from src.image_loader import ImageLoader
from src.hand_detector import HandDetector
from src.palm_extractor import PalmExtractor
from src.mask_generator import MaskGenerator
from src.data_models import ProcessingResult


class PalmSegmentationPipeline:
    """Orchestrates the complete palm detection and segmentation pipeline."""
    
    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        blur_kernel_size: int = 5,
        blur_sigma: float = 1.0,
        palm_scale: float = 0.6
    ):
        """
        Initialize the pipeline with all required components.
        
        Args:
            min_detection_confidence: Minimum confidence for hand detection (0.0-1.0)
            blur_kernel_size: Kernel size for mask edge smoothing (must be odd)
            blur_sigma: Standard deviation for Gaussian blur
            palm_scale: Scale factor for central palm region (0.1-1.0). Lower = smaller central area
        """
        self.image_loader = ImageLoader()
        self.hand_detector = HandDetector(min_detection_confidence=min_detection_confidence)
        self.palm_extractor = PalmExtractor(palm_scale=palm_scale)
        self.mask_generator = MaskGenerator(
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma
        )
    
    def process_image(
        self,
        image_path: str,
        output_mode: str = 'RGB',
        background_color: tuple = (0, 0, 0),
        verbose: bool = False,
        measure_performance: bool = False
    ) -> ProcessingResult:
        """
        Process an image through the complete pipeline: 
        ImageLoader → HandDetector → PalmExtractor → MaskGenerator
        
        Args:
            image_path: Path to input image file
            output_mode: Output format - 'RGB' or 'RGBA'
            background_color: RGB tuple for background (ignored in RGBA mode)
            verbose: If True, store intermediate processing steps for debugging
            measure_performance: If True, measure and report timing for each stage
            
        Returns:
            ProcessingResult with success status, output image, palm region, error message,
            and optionally timing information
        """
        # Store intermediate results if verbose mode is enabled
        intermediate_steps = {} if verbose else None
        timing_info = {} if measure_performance else None
        pipeline_start_time = time.perf_counter() if measure_performance else None
        
        try:
            # Stage 1: Load and validate image
            stage_start = time.perf_counter() if measure_performance else None
            try:
                image = self.image_loader.load_image(image_path, resize_large=True)
                if verbose:
                    intermediate_steps['original_image'] = image.copy()
                    print(f"[VERBOSE] Stage 1: Image loaded - Shape: {image.shape}, Size: {image.shape[1]}x{image.shape[0]}")
                
                if measure_performance:
                    timing_info['image_loading'] = time.perf_counter() - stage_start
                    
            except FileNotFoundError as e:
                return ProcessingResult(
                    success=False,
                    output_image=None,
                    palm_region=None,
                    error_message=f"File not found: {str(e)}"
                )
            except ValueError as e:
                return ProcessingResult(
                    success=False,
                    output_image=None,
                    palm_region=None,
                    error_message=f"Invalid image: {str(e)}"
                )
            
            # Stage 2: Detect hand landmarks
            stage_start = time.perf_counter() if measure_performance else None
            try:
                hand_landmarks = self.hand_detector.detect_hand(image)
                
                if hand_landmarks is None:
                    if verbose:
                        print("[VERBOSE] Stage 2: No hand detected in image")
                    return ProcessingResult(
                        success=False,
                        output_image=None,
                        palm_region=None,
                        error_message=(
                            "No hand detected in the image. "
                            "Please ensure the hand is clearly visible with good lighting."
                        )
                    )
                
                if verbose:
                    intermediate_steps['hand_landmarks'] = hand_landmarks
                    print(f"[VERBOSE] Stage 2: Hand detected - {len(hand_landmarks.landmarks)} landmarks, Handedness: {hand_landmarks.handedness}")
                    # Create visualization with landmarks
                    landmarks_vis = self.visualize_landmarks(image, hand_landmarks, palm_region=None, show_palm_center=False)
                    intermediate_steps['landmarks_visualization'] = landmarks_vis
                
                if measure_performance:
                    timing_info['hand_detection'] = time.perf_counter() - stage_start
                    
            except Exception as e:
                return ProcessingResult(
                    success=False,
                    output_image=None,
                    palm_region=None,
                    error_message=f"Hand detection failed: {str(e)}"
                )
            
            # Stage 3: Extract palm region
            stage_start = time.perf_counter() if measure_performance else None
            try:
                palm_region = self.palm_extractor.extract_palm_region(
                    hand_landmarks.landmarks,
                    (image.shape[0], image.shape[1])
                )
                
                if verbose:
                    intermediate_steps['palm_region'] = palm_region
                    print(f"[VERBOSE] Stage 3: Palm region extracted - Center: {palm_region.center}, Area: {palm_region.area:.2f} pixels")
                    # Create visualization with palm contour
                    palm_vis = self.visualize_landmarks(image, hand_landmarks, palm_region=palm_region, show_palm_center=True)
                    intermediate_steps['palm_visualization'] = palm_vis
                
                if measure_performance:
                    timing_info['palm_extraction'] = time.perf_counter() - stage_start
                    
            except Exception as e:
                return ProcessingResult(
                    success=False,
                    output_image=None,
                    palm_region=None,
                    error_message=f"Palm extraction failed: {str(e)}"
                )
            
            # Stage 4: Generate mask and apply to image
            stage_start = time.perf_counter() if measure_performance else None
            try:
                mask = self.mask_generator.create_palm_mask(
                    (image.shape[0], image.shape[1]),
                    palm_region.contour
                )
                
                if verbose:
                    intermediate_steps['mask'] = mask
                    print(f"[VERBOSE] Stage 4: Mask created - Shape: {mask.shape}, Non-zero pixels: {np.count_nonzero(mask)}")
                
                mask_creation_time = time.perf_counter() - stage_start if measure_performance else None
                
                output_image = self.mask_generator.apply_mask(
                    image,
                    mask,
                    background_color=background_color,
                    output_mode=output_mode
                )
                
                if verbose:
                    intermediate_steps['output_image'] = output_image
                    print(f"[VERBOSE] Stage 4: Mask applied - Output mode: {output_mode}, Shape: {output_image.shape}")
                
                if measure_performance:
                    timing_info['mask_generation'] = mask_creation_time
                    timing_info['mask_application'] = time.perf_counter() - stage_start - mask_creation_time
                    
            except Exception as e:
                return ProcessingResult(
                    success=False,
                    output_image=None,
                    palm_region=None,
                    error_message=f"Mask generation failed: {str(e)}"
                )
            
            # Calculate total processing time
            if measure_performance:
                timing_info['total_time'] = time.perf_counter() - pipeline_start_time
                
                # Print timing report
                print("\n" + "=" * 60)
                print("PERFORMANCE TIMING REPORT")
                print("=" * 60)
                print(f"Image Loading:      {timing_info['image_loading']*1000:>8.2f} ms")
                print(f"Hand Detection:     {timing_info['hand_detection']*1000:>8.2f} ms")
                print(f"Palm Extraction:    {timing_info['palm_extraction']*1000:>8.2f} ms")
                print(f"Mask Generation:    {timing_info['mask_generation']*1000:>8.2f} ms")
                print(f"Mask Application:   {timing_info['mask_application']*1000:>8.2f} ms")
                print("-" * 60)
                print(f"TOTAL TIME:         {timing_info['total_time']*1000:>8.2f} ms ({timing_info['total_time']:.3f} seconds)")
                print("=" * 60 + "\n")
            
            # Success!
            if verbose:
                print("[VERBOSE] Pipeline completed successfully")
                # Store all intermediate steps in the result
                intermediate_steps['success'] = True
            
            result = ProcessingResult(
                success=True,
                output_image=output_image,
                palm_region=palm_region,
                error_message=None
            )
            
            # Attach intermediate steps if verbose mode was enabled
            if verbose:
                result.intermediate_steps = intermediate_steps
            
            # Attach timing information if performance measurement was enabled
            if measure_performance:
                result.timing_info = timing_info
            
            return result
            
        except Exception as e:
            # Catch any unexpected errors
            return ProcessingResult(
                success=False,
                output_image=None,
                palm_region=None,
                error_message=f"Unexpected error in pipeline: {str(e)}"
            )
    
    def visualize_landmarks(
        self,
        image: np.ndarray,
        hand_landmarks,
        palm_region=None,
        show_palm_center: bool = True
    ) -> np.ndarray:
        """
        Create visualization with hand landmarks and palm region for debugging.
        
        Args:
            image: Input image in RGB format
            hand_landmarks: HandLandmarks object with detected landmarks
            palm_region: Optional PalmRegion object to visualize palm contour
            show_palm_center: Whether to mark the palm center point
            
        Returns:
            Annotated image with landmarks and palm region drawn
        """
        # Create a copy to avoid modifying original
        vis_image = image.copy()
        
        # Convert RGB to BGR for OpenCV drawing
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        
        # Draw all hand landmarks
        if hand_landmarks is not None:
            for idx, (x, y) in enumerate(hand_landmarks.landmarks):
                # Draw landmark point
                cv2.circle(vis_image, (x, y), 5, (0, 255, 0), -1)
                # Draw landmark index
                cv2.putText(
                    vis_image,
                    str(idx),
                    (x + 7, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1
                )
            
            # Draw connections between landmarks
            connections = [
                # Thumb
                (0, 1), (1, 2), (2, 3), (3, 4),
                # Index finger
                (0, 5), (5, 6), (6, 7), (7, 8),
                # Middle finger
                (0, 9), (9, 10), (10, 11), (11, 12),
                # Ring finger
                (0, 13), (13, 14), (14, 15), (15, 16),
                # Pinky
                (0, 17), (17, 18), (18, 19), (19, 20),
                # Palm
                (5, 9), (9, 13), (13, 17)
            ]
            
            for start_idx, end_idx in connections:
                start_point = hand_landmarks.landmarks[start_idx]
                end_point = hand_landmarks.landmarks[end_idx]
                cv2.line(vis_image, start_point, end_point, (255, 0, 0), 2)
        
        # Draw palm region contour
        if palm_region is not None:
            # Draw palm contour
            cv2.drawContours(vis_image, [palm_region.contour], -1, (0, 0, 255), 2)
            
            # Draw palm center
            if show_palm_center:
                center_x, center_y = palm_region.center
                cv2.circle(vis_image, (center_x, center_y), 8, (255, 0, 255), -1)
                cv2.putText(
                    vis_image,
                    "Palm Center",
                    (center_x + 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2
                )
        
        # Convert back to RGB
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        
        return vis_image
    
    def display_intermediate_steps(
        self,
        intermediate_steps: dict,
        window_prefix: str = "Step"
    ) -> None:
        """
        Display all intermediate processing steps in separate windows for debugging.
        
        Args:
            intermediate_steps: Dictionary containing intermediate processing results
            window_prefix: Prefix for window names
        """
        if not intermediate_steps:
            print("No intermediate steps to display")
            return
        
        print("\n" + "=" * 60)
        print("Displaying Intermediate Processing Steps")
        print("=" * 60)
        print("Press any key in any window to proceed to next step")
        print("=" * 60 + "\n")
        
        step_order = [
            ('original_image', 'Original Image'),
            ('landmarks_visualization', 'Hand Landmarks Detected'),
            ('palm_visualization', 'Palm Region Identified'),
            ('mask', 'Binary Mask'),
            ('output_image', 'Final Output')
        ]
        
        for step_key, step_name in step_order:
            if step_key in intermediate_steps:
                image = intermediate_steps[step_key]
                
                # Handle different image types
                if step_key == 'mask':
                    # Mask is grayscale, convert to BGR for display
                    display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif len(image.shape) == 3:
                    if image.shape[2] == 3:  # RGB
                        display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    elif image.shape[2] == 4:  # RGBA
                        display_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
                    else:
                        display_image = image
                else:
                    display_image = image
                
                window_name = f"{window_prefix}: {step_name}"
                print(f"Showing: {step_name}")
                cv2.imshow(window_name, display_image)
                cv2.waitKey(0)
                cv2.destroyWindow(window_name)
        
        print("\nAll intermediate steps displayed")
        cv2.destroyAllWindows()
    
    def save_intermediate_steps(
        self,
        intermediate_steps: dict,
        output_dir: str,
        base_filename: str = "step"
    ) -> bool:
        """
        Save all intermediate processing steps to disk for debugging.
        
        Args:
            intermediate_steps: Dictionary containing intermediate processing results
            output_dir: Directory where intermediate images should be saved
            base_filename: Base name for output files
            
        Returns:
            True if all saves were successful, False otherwise
        """
        if not intermediate_steps:
            print("No intermediate steps to save")
            return False
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            step_order = [
                ('original_image', '1_original'),
                ('landmarks_visualization', '2_landmarks'),
                ('palm_visualization', '3_palm_region'),
                ('mask', '4_mask'),
                ('output_image', '5_output')
            ]
            
            saved_count = 0
            
            for step_key, step_suffix in step_order:
                if step_key in intermediate_steps:
                    image = intermediate_steps[step_key]
                    output_path = os.path.join(output_dir, f"{base_filename}_{step_suffix}.png")
                    
                    # Convert to BGR/BGRA for saving
                    if step_key == 'mask':
                        # Mask is already grayscale
                        image_to_save = image
                    elif len(image.shape) == 3:
                        if image.shape[2] == 3:  # RGB
                            image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        elif image.shape[2] == 4:  # RGBA
                            image_to_save = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
                        else:
                            image_to_save = image
                    else:
                        image_to_save = image
                    
                    success = cv2.imwrite(output_path, image_to_save)
                    
                    if success:
                        print(f"Saved: {output_path}")
                        saved_count += 1
                    else:
                        print(f"Failed to save: {output_path}")
            
            print(f"\nSaved {saved_count} intermediate step images to {output_dir}")
            return saved_count > 0
            
        except Exception as e:
            print(f"Error saving intermediate steps: {str(e)}")
            return False
    
    def save_output(
        self,
        image: np.ndarray,
        output_path: str,
        create_dirs: bool = True
    ) -> bool:
        """
        Save processed image to disk in PNG format with proper color channels.
        
        Args:
            image: Image array to save (RGB or RGBA format)
            output_path: Path where the image should be saved
            create_dirs: Whether to create parent directories if they don't exist
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Create parent directories if needed
            if create_dirs:
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
            
            # Ensure output path has .png extension
            if not output_path.lower().endswith('.png'):
                output_path = output_path + '.png'
            
            # Convert RGB to BGR for OpenCV saving
            if image.shape[2] == 3:  # RGB
                image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image.shape[2] == 4:  # RGBA
                image_to_save = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
            else:
                raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
            
            # Save image
            success = cv2.imwrite(output_path, image_to_save)
            
            if not success:
                raise IOError(f"Failed to write image to {output_path}")
            
            return True
            
        except Exception as e:
            print(f"Error saving output: {str(e)}")
            return False
    
    def create_side_by_side(
        self,
        original_image: np.ndarray,
        masked_image: np.ndarray,
        add_labels: bool = True
    ) -> np.ndarray:
        """
        Create side-by-side visualization of original and masked output.
        
        Args:
            original_image: Original input image (RGB)
            masked_image: Processed image with palm mask (RGB or RGBA)
            add_labels: Whether to add text labels to each image
            
        Returns:
            Combined side-by-side image in RGB format
        """
        # Convert RGBA to RGB if needed for masked image
        if masked_image.shape[2] == 4:
            # Create white background
            background = np.ones((masked_image.shape[0], masked_image.shape[1], 3), dtype=np.uint8) * 255
            # Extract alpha channel
            alpha = masked_image[:, :, 3:4] / 255.0
            # Blend with background
            masked_rgb = (masked_image[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)
        else:
            masked_rgb = masked_image
        
        # Ensure both images have the same height
        if original_image.shape[0] != masked_rgb.shape[0]:
            # Resize to match heights
            target_height = max(original_image.shape[0], masked_rgb.shape[0])
            if original_image.shape[0] != target_height:
                aspect_ratio = original_image.shape[1] / original_image.shape[0]
                new_width = int(target_height * aspect_ratio)
                original_image = cv2.resize(original_image, (new_width, target_height))
            if masked_rgb.shape[0] != target_height:
                aspect_ratio = masked_rgb.shape[1] / masked_rgb.shape[0]
                new_width = int(target_height * aspect_ratio)
                masked_rgb = cv2.resize(masked_rgb, (new_width, target_height))
        
        # Add labels if requested
        if add_labels:
            original_labeled = original_image.copy()
            masked_labeled = masked_rgb.copy()
            
            # Convert to BGR for text rendering
            original_bgr = cv2.cvtColor(original_labeled, cv2.COLOR_RGB2BGR)
            masked_bgr = cv2.cvtColor(masked_labeled, cv2.COLOR_RGB2BGR)
            
            # Add text labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            color = (255, 255, 255)
            bg_color = (0, 0, 0)
            
            # Original image label
            text = "Original"
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (original_bgr.shape[1] - text_size[0]) // 2
            text_y = 40
            cv2.rectangle(original_bgr, (text_x - 10, text_y - text_size[1] - 10), 
                         (text_x + text_size[0] + 10, text_y + 10), bg_color, -1)
            cv2.putText(original_bgr, text, (text_x, text_y), font, font_scale, color, thickness)
            
            # Masked image label
            text = "Palm Segmented"
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (masked_bgr.shape[1] - text_size[0]) // 2
            text_y = 40
            cv2.rectangle(masked_bgr, (text_x - 10, text_y - text_size[1] - 10), 
                         (text_x + text_size[0] + 10, text_y + 10), bg_color, -1)
            cv2.putText(masked_bgr, text, (text_x, text_y), font, font_scale, color, thickness)
            
            # Convert back to RGB
            original_labeled = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
            masked_labeled = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
            
            # Concatenate horizontally
            side_by_side = np.hstack([original_labeled, masked_labeled])
        else:
            # Concatenate without labels
            side_by_side = np.hstack([original_image, masked_rgb])
        
        return side_by_side
    
    def display_output(
        self,
        image: np.ndarray,
        window_name: str = "Palm Segmentation Output",
        wait_key: int = 0
    ) -> None:
        """
        Display output image using OpenCV window.
        
        Args:
            image: Image to display (RGB or RGBA format)
            window_name: Name of the display window
            wait_key: Time to wait in milliseconds (0 = wait for key press)
        """
        # Convert RGB/RGBA to BGR/BGRA for OpenCV display
        if image.shape[2] == 3:  # RGB
            display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.shape[2] == 4:  # RGBA
            display_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        else:
            raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
        
        # Create window and display
        cv2.imshow(window_name, display_image)
        cv2.waitKey(wait_key)
        cv2.destroyAllWindows()
