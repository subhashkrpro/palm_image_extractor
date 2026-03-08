# API Documentation

Complete API reference for the Palm Detection and Segmentation System.

## Table of Contents

- [Overview](#overview)
- [Core Components](#core-components)
  - [PalmSegmentationPipeline](#palmsegmentationpipeline)
  - [ImageLoader](#imageloader)
  - [HandDetector](#handdetector)
  - [PalmExtractor](#palmextractor)
  - [MaskGenerator](#maskgenerator)
- [Data Models](#data-models)
- [Usage Patterns](#usage-patterns)
- [Error Handling](#error-handling)

## Overview

The system is organized into modular components that can be used independently or through the main pipeline orchestrator. The typical flow is:

```
ImageLoader → HandDetector → PalmExtractor → MaskGenerator → Output
```

## Core Components

### PalmSegmentationPipeline

Main orchestrator that coordinates all components in the processing pipeline.

**Module:** `src.palm_segmentation_pipeline`

#### Class Definition

```python
class PalmSegmentationPipeline:
    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        blur_kernel_size: int = 5,
        blur_sigma: float = 1.0,
        palm_scale: float = 0.6
    )
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_detection_confidence` | float | 0.7 | Minimum confidence threshold for hand detection (0.0-1.0) |
| `blur_kernel_size` | int | 5 | Kernel size for Gaussian blur on mask edges (must be odd) |
| `blur_sigma` | float | 1.0 | Standard deviation for Gaussian blur |
| `palm_scale` | float | 0.6 | Scale factor for central palm region (0.1-1.0). Lower values create smaller, more centered palm regions |

**Example:**
```python
from src.palm_segmentation_pipeline import PalmSegmentationPipeline

# Default parameters
pipeline = PalmSegmentationPipeline()

# Custom parameters
pipeline = PalmSegmentationPipeline(
    min_detection_confidence=0.8,
    blur_kernel_size=7,
    blur_sigma=2.0
)
```

#### Methods

##### process_image()

Process an image through the complete detection and segmentation pipeline.

```python
def process_image(
    self,
    image_path: str,
    output_mode: str = 'RGB',
    background_color: tuple = (0, 0, 0),
    verbose: bool = False,
    measure_performance: bool = False
) -> ProcessingResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | str | Required | Path to input image file |
| `output_mode` | str | 'RGB' | Output format: 'RGB' or 'RGBA' |
| `background_color` | tuple | (0, 0, 0) | RGB background color (ignored in RGBA mode) |
| `verbose` | bool | False | Store intermediate steps for debugging |
| `measure_performance` | bool | False | Measure and report timing for each stage |

**Returns:** `ProcessingResult` object

**Example:**
```python
# Basic usage
result = pipeline.process_image("input.jpg")

# With transparent background
result = pipeline.process_image(
    "input.jpg",
    output_mode='RGBA'
)

# With custom background
result = pipeline.process_image(
    "input.jpg",
    output_mode='RGB',
    background_color=(50, 100, 200)
)

# With debugging
result = pipeline.process_image(
    "input.jpg",
    verbose=True,
    measure_performance=True
)
```

##### save_output()

Save processed image to disk in PNG format.

```python
def save_output(
    self,
    image: np.ndarray,
    output_path: str,
    create_dirs: bool = True
) -> bool
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | np.ndarray | Required | Image array to save (RGB or RGBA) |
| `output_path` | str | Required | Path where image should be saved |
| `create_dirs` | bool | True | Create parent directories if needed |

**Returns:** `bool` - True if successful, False otherwise

**Example:**
```python
success = pipeline.save_output(result.output_image, "output.png")
if success:
    print("Image saved successfully")
```

##### visualize_landmarks()

Create visualization with hand landmarks and palm region for debugging.

```python
def visualize_landmarks(
    self,
    image: np.ndarray,
    hand_landmarks: HandLandmarks,
    palm_region: PalmRegion = None,
    show_palm_center: bool = True
) -> np.ndarray
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | np.ndarray | Required | Input image in RGB format |
| `hand_landmarks` | HandLandmarks | Required | Detected hand landmarks |
| `palm_region` | PalmRegion | None | Optional palm region to visualize |
| `show_palm_center` | bool | True | Whether to mark palm center point |

**Returns:** `np.ndarray` - Annotated image with landmarks and palm region

**Example:**
```python
from src.image_loader import ImageLoader

loader = ImageLoader()
original = loader.load_image("input.jpg")

result = pipeline.process_image("input.jpg")
hand_landmarks = pipeline.hand_detector.detect_hand(original)

vis_image = pipeline.visualize_landmarks(
    original,
    hand_landmarks,
    palm_region=result.palm_region,
    show_palm_center=True
)

pipeline.save_output(vis_image, "visualization.png")
```

##### create_side_by_side()

Create side-by-side comparison of original and processed images.

```python
def create_side_by_side(
    self,
    original_image: np.ndarray,
    masked_image: np.ndarray,
    add_labels: bool = True
) -> np.ndarray
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `original_image` | np.ndarray | Required | Original input image (RGB) |
| `masked_image` | np.ndarray | Required | Processed image (RGB or RGBA) |
| `add_labels` | bool | True | Add text labels to each image |

**Returns:** `np.ndarray` - Combined side-by-side image in RGB format

**Example:**
```python
comparison = pipeline.create_side_by_side(
    original_image,
    result.output_image,
    add_labels=True
)
pipeline.save_output(comparison, "comparison.png")
```

##### display_output()

Display output image using OpenCV window.

```python
def display_output(
    self,
    image: np.ndarray,
    window_name: str = "Palm Segmentation Output",
    wait_key: int = 0
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | np.ndarray | Required | Image to display (RGB or RGBA) |
| `window_name` | str | "Palm Segmentation Output" | Name of display window |
| `wait_key` | int | 0 | Time to wait in ms (0 = wait for key press) |

**Example:**
```python
pipeline.display_output(result.output_image)
```

##### display_intermediate_steps()

Display all intermediate processing steps in separate windows.

```python
def display_intermediate_steps(
    self,
    intermediate_steps: dict,
    window_prefix: str = "Step"
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `intermediate_steps` | dict | Required | Dictionary from ProcessingResult |
| `window_prefix` | str | "Step" | Prefix for window names |

**Example:**
```python
result = pipeline.process_image("input.jpg", verbose=True)
if result.intermediate_steps:
    pipeline.display_intermediate_steps(result.intermediate_steps)
```

##### save_intermediate_steps()

Save all intermediate processing steps to disk.

```python
def save_intermediate_steps(
    self,
    intermediate_steps: dict,
    output_dir: str,
    base_filename: str = "step"
) -> bool
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `intermediate_steps` | dict | Required | Dictionary from ProcessingResult |
| `output_dir` | str | Required | Directory for output images |
| `base_filename` | str | "step" | Base name for output files |

**Returns:** `bool` - True if successful, False otherwise

**Example:**
```python
result = pipeline.process_image("input.jpg", verbose=True)
if result.intermediate_steps:
    pipeline.save_intermediate_steps(
        result.intermediate_steps,
        "debug_output",
        base_filename="debug"
    )
```

---

### ImageLoader

Handles image loading, validation, and preprocessing.

**Module:** `src.image_loader`

#### Class Definition

```python
class ImageLoader:
    def __init__(self)
```

#### Methods

##### load_image()

Load and validate an image from file.

```python
def load_image(
    self,
    image_path: str,
    resize_large: bool = False,
    max_dimension: int = 1920
) -> np.ndarray
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | str | Required | Path to image file |
| `resize_large` | bool | False | Resize large images automatically |
| `max_dimension` | int | 1920 | Maximum width or height if resizing |

**Returns:** `np.ndarray` - Image in RGB format

**Raises:**
- `FileNotFoundError` - If image file doesn't exist
- `ValueError` - If image is invalid or corrupted

**Example:**
```python
from src.image_loader import ImageLoader

loader = ImageLoader()

# Basic loading
image = loader.load_image("input.jpg")

# With automatic resizing
image = loader.load_image("large_image.jpg", resize_large=True)
```

---

### HandDetector

Detects hands and extracts landmarks using MediaPipe HandLandmarker (tasks API).

**Module:** `src.hand_detector`

**Note:** This component uses MediaPipe 0.10+ with the new `tasks` API. On first initialization, it will automatically download the hand landmarker model (~10MB) if not already present.

#### Class Definition

```python
class HandDetector:
    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5
    )
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_detection_confidence` | float | 0.7 | Minimum confidence for detection (0.0-1.0) |
| `min_tracking_confidence` | float | 0.5 | Minimum confidence for tracking (0.0-1.0) |

#### Methods

##### detect_hand()

Detect hand and extract landmarks from image.

```python
def detect_hand(
    self,
    image: np.ndarray
) -> Optional[HandLandmarks]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | np.ndarray | Required | Input image in RGB format |

**Returns:** `HandLandmarks` object or `None` if no hand detected

**Example:**
```python
from src.hand_detector import HandDetector
from src.image_loader import ImageLoader

loader = ImageLoader()
detector = HandDetector(min_detection_confidence=0.8)

image = loader.load_image("input.jpg")
hand_landmarks = detector.detect_hand(image)

if hand_landmarks:
    print(f"Detected {hand_landmarks.handedness} hand")
    print(f"Landmarks: {len(hand_landmarks.landmarks)}")
else:
    print("No hand detected")
```

---

### PalmExtractor

Extracts central palm region from hand landmarks.

**Module:** `src.palm_extractor`

#### Class Definition

```python
class PalmExtractor:
    def __init__(self, palm_scale: float = 0.6)
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `palm_scale` | float | 0.6 | Scale factor for central palm region (0.1-1.0). Lower values create smaller, more centered palm regions |

#### Methods

##### extract_palm_region()

Extract palm region from hand landmarks.

```python
def extract_palm_region(
    self,
    landmarks: List[Tuple[int, int]],
    image_shape: Tuple[int, int]
) -> PalmRegion
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `landmarks` | List[Tuple[int, int]] | Required | List of 21 (x, y) landmark coordinates |
| `image_shape` | Tuple[int, int] | Required | Image dimensions (height, width) |

**Returns:** `PalmRegion` object

**Example:**
```python
from src.palm_extractor import PalmExtractor

# Default central palm extraction
extractor = PalmExtractor()

# Smaller central palm region
extractor = PalmExtractor(palm_scale=0.5)

# Larger palm region
extractor = PalmExtractor(palm_scale=0.8)

palm_region = extractor.extract_palm_region(
    hand_landmarks.landmarks,
    (image.shape[0], image.shape[1])
)

print(f"Palm center: {palm_region.center}")
print(f"Palm area: {palm_region.area} pixels")
```

##### get_palm_landmarks()

Extract palm-specific landmarks (wrist and finger bases).

```python
def get_palm_landmarks(
    self,
    landmarks: List[Tuple[int, int]]
) -> List[Tuple[int, int]]
```

**Returns:** List of 5 palm landmark coordinates (indices 0, 5, 9, 13, 17)

##### calculate_palm_center()

Calculate centroid of palm region.

```python
def calculate_palm_center(
    self,
    palm_landmarks: List[Tuple[int, int]]
) -> Tuple[int, int]
```

**Returns:** (x, y) coordinate of palm center

---

### MaskGenerator

Creates and applies palm masks.

**Module:** `src.mask_generator`

#### Class Definition

```python
class MaskGenerator:
    def __init__(
        self,
        blur_kernel_size: int = 5,
        blur_sigma: float = 1.0
    )
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `blur_kernel_size` | int | 5 | Kernel size for Gaussian blur (must be odd) |
| `blur_sigma` | float | 1.0 | Standard deviation for Gaussian blur |

#### Methods

##### create_palm_mask()

Create binary mask for palm region.

```python
def create_palm_mask(
    self,
    image_shape: Tuple[int, int],
    palm_contour: np.ndarray
) -> np.ndarray
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_shape` | Tuple[int, int] | Required | Image dimensions (height, width) |
| `palm_contour` | np.ndarray | Required | Polygon points defining palm boundary |

**Returns:** `np.ndarray` - Binary mask (255 for palm, 0 for background)

##### apply_mask()

Apply mask to image with specified background.

```python
def apply_mask(
    self,
    image: np.ndarray,
    mask: np.ndarray,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    output_mode: str = 'RGB'
) -> np.ndarray
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | np.ndarray | Required | Input image in RGB format |
| `mask` | np.ndarray | Required | Binary mask |
| `background_color` | Tuple[int, int, int] | (0, 0, 0) | RGB background color |
| `output_mode` | str | 'RGB' | Output format: 'RGB' or 'RGBA' |

**Returns:** `np.ndarray` - Masked image

**Example:**
```python
from src.mask_generator import MaskGenerator

generator = MaskGenerator(blur_kernel_size=7, blur_sigma=2.0)

mask = generator.create_palm_mask(
    (image.shape[0], image.shape[1]),
    palm_region.contour
)

# RGB with black background
output_rgb = generator.apply_mask(image, mask)

# RGB with custom background
output_custom = generator.apply_mask(
    image, mask,
    background_color=(50, 100, 200)
)

# RGBA with transparency
output_rgba = generator.apply_mask(
    image, mask,
    output_mode='RGBA'
)
```

---

## Data Models

### HandLandmarks

Represents detected hand landmarks with metadata.

**Module:** `src.data_models`

```python
@dataclass
class HandLandmarks:
    landmarks: List[Tuple[int, int]]
    image_width: int
    image_height: int
    handedness: str
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `landmarks` | List[Tuple[int, int]] | 21 (x, y) coordinate tuples |
| `image_width` | int | Width of source image in pixels |
| `image_height` | int | Height of source image in pixels |
| `handedness` | str | "Left" or "Right" |

**Landmark Indices:**
- 0: Wrist
- 1-4: Thumb
- 5-8: Index finger
- 9-12: Middle finger
- 13-16: Ring finger
- 17-20: Pinky

### PalmRegion

Represents extracted palm region with geometric properties.

**Module:** `src.data_models`

```python
@dataclass
class PalmRegion:
    contour: np.ndarray
    center: Tuple[int, int]
    area: float
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `contour` | np.ndarray | Polygon points defining palm boundary |
| `center` | Tuple[int, int] | (x, y) coordinate of palm center |
| `area` | float | Area of palm region in pixels |

### ProcessingResult

Represents the result of pipeline processing.

**Module:** `src.data_models`

```python
@dataclass
class ProcessingResult:
    success: bool
    output_image: Optional[np.ndarray]
    palm_region: Optional[PalmRegion]
    error_message: Optional[str]
    intermediate_steps: Optional[Dict] = None
    timing_info: Optional[Dict[str, float]] = None
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `success` | bool | True if processing completed successfully |
| `output_image` | Optional[np.ndarray] | Processed image or None if failed |
| `palm_region` | Optional[PalmRegion] | Palm region data or None if failed |
| `error_message` | Optional[str] | Error description or None if successful |
| `intermediate_steps` | Optional[Dict] | Intermediate images (if verbose=True) |
| `timing_info` | Optional[Dict[str, float]] | Timing measurements (if measure_performance=True) |

---

## Usage Patterns

### Pattern 1: Simple Processing

```python
from src.palm_segmentation_pipeline import PalmSegmentationPipeline

pipeline = PalmSegmentationPipeline()
result = pipeline.process_image("input.jpg")

if result.success:
    pipeline.save_output(result.output_image, "output.png")
else:
    print(f"Error: {result.error_message}")
```

### Pattern 2: Batch Processing

```python
from src.palm_segmentation_pipeline import PalmSegmentationPipeline
import os

pipeline = PalmSegmentationPipeline()
image_files = ["img1.jpg", "img2.jpg", "img3.jpg"]

for image_file in image_files:
    result = pipeline.process_image(image_file)
    if result.success:
        output_name = f"output_{os.path.basename(image_file)}"
        pipeline.save_output(result.output_image, output_name)
```

### Pattern 3: Custom Processing with Visualization

```python
from src.palm_segmentation_pipeline import PalmSegmentationPipeline
from src.image_loader import ImageLoader

pipeline = PalmSegmentationPipeline(
    min_detection_confidence=0.8,
    blur_kernel_size=7
)

loader = ImageLoader()
original = loader.load_image("input.jpg")

result = pipeline.process_image("input.jpg", output_mode='RGBA')

if result.success:
    # Create visualization
    hand_landmarks = pipeline.hand_detector.detect_hand(original)
    vis = pipeline.visualize_landmarks(
        original, hand_landmarks,
        palm_region=result.palm_region
    )
    
    # Create comparison
    comparison = pipeline.create_side_by_side(original, result.output_image)
    
    # Save all outputs
    pipeline.save_output(result.output_image, "output.png")
    pipeline.save_output(vis, "visualization.png")
    pipeline.save_output(comparison, "comparison.png")
```

### Pattern 4: Debugging with Intermediate Steps

```python
from src.palm_segmentation_pipeline import PalmSegmentationPipeline

pipeline = PalmSegmentationPipeline()

result = pipeline.process_image(
    "input.jpg",
    verbose=True,
    measure_performance=True
)

if result.success:
    # Save intermediate steps
    pipeline.save_intermediate_steps(
        result.intermediate_steps,
        "debug_output"
    )
    
    # Access timing information
    if result.timing_info:
        print(f"Total time: {result.timing_info['total_time']:.3f}s")
```

---

## Error Handling

### Common Errors

#### FileNotFoundError

Raised when input image file doesn't exist.

```python
try:
    result = pipeline.process_image("nonexistent.jpg")
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

#### ValueError

Raised when image is invalid or corrupted.

```python
try:
    loader = ImageLoader()
    image = loader.load_image("corrupted.jpg")
except ValueError as e:
    print(f"Invalid image: {e}")
```

### ProcessingResult Error Messages

When `result.success` is False, check `result.error_message`:

| Error Message | Cause | Solution |
|---------------|-------|----------|
| "File not found: ..." | Image file doesn't exist | Check file path |
| "Invalid image: ..." | Corrupted or unsupported format | Use valid JPG/PNG |
| "No hand detected in the image..." | No hand found | Improve lighting, positioning |
| "Hand detection failed: ..." | MediaPipe error | Check image quality |
| "Palm extraction failed: ..." | Landmark processing error | Ensure clear hand visibility |
| "Mask generation failed: ..." | Mask creation error | Check contour validity |

### Best Practices

1. **Always check result.success** before accessing output_image or palm_region
2. **Use try-except** for file operations
3. **Validate inputs** before processing
4. **Use verbose mode** for debugging
5. **Check intermediate_steps** to identify failure points

```python
result = pipeline.process_image("input.jpg", verbose=True)

if not result.success:
    print(f"Processing failed: {result.error_message}")
    
    # Check where it failed
    if result.intermediate_steps:
        if 'hand_landmarks' not in result.intermediate_steps:
            print("Failed at hand detection stage")
        elif 'palm_region' not in result.intermediate_steps:
            print("Failed at palm extraction stage")
    
    sys.exit(1)

# Safe to use result.output_image here
pipeline.save_output(result.output_image, "output.png")
```
