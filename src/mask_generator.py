"""
MaskGenerator module for creating and applying palm region masks.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class MaskGenerator:
    """
    Handles creation of binary masks for palm regions and applies them to images.
    Supports both RGB (black background) and RGBA (transparent background) output modes.
    """

    def __init__(self, blur_kernel_size: int = 5, blur_sigma: float = 1.0):
        """
        Initialize MaskGenerator with edge smoothing parameters.

        Args:
            blur_kernel_size: Size of Gaussian blur kernel for edge smoothing (must be odd)
            blur_sigma: Standard deviation for Gaussian blur
        """
        if blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be an odd number")
        
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma

    def create_palm_mask(self, image_shape: Tuple[int, int], palm_contour: np.ndarray) -> np.ndarray:
        """
        Generate a binary mask for the palm region using contour filling.

        Args:
            image_shape: Tuple of (height, width) for the mask dimensions
            palm_contour: Array of contour points defining the palm boundary, shape (N, 2)

        Returns:
            Binary mask as numpy array with shape (height, width), 
            where 255 represents palm region and 0 represents background

        Raises:
            ValueError: If image_shape or palm_contour is invalid
        """
        if len(image_shape) != 2:
            raise ValueError("image_shape must be a tuple of (height, width)")
        
        if palm_contour is None or len(palm_contour) < 3:
            raise ValueError("palm_contour must contain at least 3 points")
        
        height, width = image_shape
        
        # Create empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Ensure contour is in the correct format for cv2.fillPoly
        # Convert to integer coordinates and reshape if needed
        contour_points = palm_contour.astype(np.int32)
        if contour_points.ndim == 2 and contour_points.shape[1] == 2:
            contour_points = contour_points.reshape((-1, 1, 2))
        
        # Fill the palm region with white (255)
        cv2.fillPoly(mask, [contour_points], 255)
        
        # Apply Gaussian blur for edge smoothing (anti-aliasing)
        mask = cv2.GaussianBlur(
            mask, 
            (self.blur_kernel_size, self.blur_kernel_size), 
            self.blur_sigma
        )
        
        return mask

    def apply_mask(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        background_color: Optional[Tuple[int, int, int]] = None,
        output_mode: str = 'RGB'
    ) -> np.ndarray:
        """
        Apply mask to image, compositing palm region onto specified background.
        Uses optimized NumPy vectorized operations for performance.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            mask: Binary mask as numpy array (H, W) with values 0-255
            background_color: RGB tuple for background color. 
                            Defaults to (0, 0, 0) for black background.
                            Ignored if output_mode is 'RGBA'.
            output_mode: Output format - 'RGB' for black background or 'RGBA' for transparent background

        Returns:
            Masked image as numpy array:
            - RGB mode: (H, W, 3) with palm region visible and background in specified color
            - RGBA mode: (H, W, 4) with palm region visible and transparent background

        Raises:
            ValueError: If image and mask dimensions don't match or output_mode is invalid
        """
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image shape {image.shape[:2]} doesn't match mask shape {mask.shape[:2]}"
            )
        
        if output_mode not in ['RGB', 'RGBA']:
            raise ValueError("output_mode must be 'RGB' or 'RGBA'")
        
        if background_color is None:
            background_color = (0, 0, 0)
        
        if output_mode == 'RGBA':
            # Create RGBA output with transparent background using vectorized operations
            # Allocate output array
            output = np.empty((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            
            # Copy RGB channels from original image (vectorized)
            output[:, :, :3] = image
            
            # Set alpha channel from mask (vectorized)
            output[:, :, 3] = mask
            
        else:  # RGB mode - optimized with vectorized operations
            # Normalize mask to 0-1 range for blending (vectorized)
            mask_normalized = mask.astype(np.float32) * (1.0 / 255.0)
            
            # Expand mask to 3 channels for broadcasting (view, not copy)
            mask_3channel = mask_normalized[:, :, np.newaxis]
            
            # Create inverse mask for background blending
            inv_mask = 1.0 - mask_3channel
            
            # Vectorized blending: output = image * mask + background * (1 - mask)
            # Use in-place operations where possible
            output = np.empty_like(image, dtype=np.uint8)
            
            # Perform blending with vectorized operations
            if background_color == (0, 0, 0):
                # Optimized path for black background - no need to add background
                np.multiply(image, mask_3channel, out=output, casting='unsafe')
            else:
                # General case with colored background
                background_array = np.array(background_color, dtype=np.float32)
                output = (image.astype(np.float32) * mask_3channel + 
                         background_array * inv_mask).astype(np.uint8)
        
        return output

