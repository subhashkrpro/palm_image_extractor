"""Data models for palm detection and segmentation pipeline."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class HandLandmarks:
    """Represents detected hand landmarks with metadata.
    
    Attributes:
        landmarks: List of 21 (x, y) coordinate tuples for hand landmarks
        image_width: Width of the source image in pixels
        image_height: Height of the source image in pixels
        handedness: Hand type, either "Left" or "Right"
    """
    landmarks: List[Tuple[int, int]]
    image_width: int
    image_height: int
    handedness: str


@dataclass
class PalmRegion:
    """Represents the extracted palm region with geometric properties.
    
    Attributes:
        contour: NumPy array of polygon points defining palm boundary
        center: (x, y) coordinate tuple of palm center point
        area: Area of palm region in pixels
    """
    contour: np.ndarray
    center: Tuple[int, int]
    area: float


@dataclass
class ProcessingResult:
    """Represents the result of the palm segmentation pipeline.
    
    Attributes:
        success: Boolean indicating if processing completed successfully
        output_image: Processed image with masked palm region, or None if failed
        palm_region: Extracted palm region data, or None if failed
        error_message: Error description if processing failed, or None if successful
        intermediate_steps: Optional dictionary containing intermediate processing steps for debugging
        timing_info: Optional dictionary containing timing measurements for each pipeline stage
    """
    success: bool
    output_image: Optional[np.ndarray]
    palm_region: Optional[PalmRegion]
    error_message: Optional[str]
    intermediate_steps: Optional[Dict] = None
    timing_info: Optional[Dict[str, float]] = None
