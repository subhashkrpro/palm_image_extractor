"""
ImageLoader component for loading and validating input images.
"""
import os
from typing import Optional, Tuple
import cv2
import numpy as np


class ImageLoader:
    """Handles image input and validation for the palm segmentation pipeline."""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png'}
    MAX_WIDTH = 1920
    MAX_HEIGHT = 1080
    
    def load_image(self, image_path: str, resize_large: bool = True) -> np.ndarray:
        """
        Load an image from the specified file path.
        
        Args:
            image_path: Path to the image file
            resize_large: If True, resize images larger than MAX_WIDTH x MAX_HEIGHT
            
        Returns:
            numpy.ndarray: Image in RGB format
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If the image format is not supported or image is invalid
        """
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check file extension
        _, ext = os.path.splitext(image_path)
        if ext.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {ext}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        # Load image using OpenCV
        image = cv2.imread(image_path)
        
        # Check if image was loaded successfully
        if image is None:
            raise ValueError(f"Failed to load image. The file may be corrupted: {image_path}")
        
        # Convert from BGR to RGB (OpenCV loads in BGR by default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Validate the loaded image
        if not self.validate_image(image_rgb):
            raise ValueError(f"Invalid image data: {image_path}")
        
        # Resize large images for performance optimization
        if resize_large:
            image_rgb = self._resize_if_large(image_rgb)
        
        return image_rgb
    
    def _resize_if_large(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image if it exceeds maximum dimensions while maintaining aspect ratio.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Resized image if necessary, otherwise original image
        """
        height, width = image.shape[:2]
        
        # Check if resizing is needed
        if width <= self.MAX_WIDTH and height <= self.MAX_HEIGHT:
            return image
        
        # Calculate scaling factor to fit within max dimensions
        width_scale = self.MAX_WIDTH / width
        height_scale = self.MAX_HEIGHT / height
        scale = min(width_scale, height_scale)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize using high-quality interpolation
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate that the image contains readable data.
        
        Args:
            image: Image as numpy array
            
        Returns:
            bool: True if image is valid, False otherwise
        """
        if image is None:
            return False
        
        # Check if it's a numpy array
        if not isinstance(image, np.ndarray):
            return False
        
        # Check if image has valid dimensions
        if image.ndim not in [2, 3]:
            return False
        
        # Check if image has valid shape
        if image.shape[0] == 0 or image.shape[1] == 0:
            return False
        
        # Check if image has valid data type
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            return False
        
        return True
