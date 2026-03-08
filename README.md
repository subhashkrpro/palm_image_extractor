# Palm Detection and Segmentation System

A Python-based computer vision system that automatically detects hands in images and segments the palm region using MediaPipe and OpenCV.

## Features

- **Automatic Hand Detection**: Uses MediaPipe to detect hands and extract 21 landmark points
- **Palm Region Extraction**: Identifies and isolates the palm area from the full hand
- **High-Quality Masking**: Generates smooth, anti-aliased masks for clean segmentation
- **Flexible Output**: Supports RGB (with custom background) or RGBA (transparent) output
- **Visualization Tools**: Debug and visualize landmarks, palm contours, and processing steps
- **Batch Processing**: Process multiple images efficiently
- **Performance Monitoring**: Measure and report timing for each pipeline stage
- **Command-Line Interface**: Easy-to-use CLI with extensive options

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface)
  - [Python API](#python-api)
- [API Documentation](#api-documentation)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Requirements](#requirements)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Repository

```bash
git clone <repository-url>
cd palm-detection-segmentation
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

The required packages are:
- `opencv-python` - Image processing and computer vision
- `mediapipe` (>=0.10.0) - Hand detection and landmark extraction (uses new tasks API)
- `numpy` - Numerical operations
- `pillow` - Additional image I/O support

**Important:** This project uses MediaPipe 0.10+ with the new `tasks` API. If you have an older version of MediaPipe installed, please upgrade:
```bash
pip install --upgrade mediapipe
```

### Step 3: Verify Installation

```bash
python main.py --help
```

You should see the help message with all available options.

**Note:** On first run, the system will automatically download the MediaPipe hand landmarker model (~10MB). This is a one-time download and will be cached in the `models/` directory.

## Quick Start

### Process a Single Image

```bash
python main.py input.jpg -o output.png
```

### Process with Visualization

```bash
python main.py input.jpg -o output.png --visualize
```

### Batch Process Multiple Images

```bash
python main.py image1.jpg image2.jpg image3.jpg -o output_dir/
```

### Use Transparent Background

```bash
python main.py input.jpg -o output.png --transparent
```

## Usage

### Command-Line Interface

The system provides a comprehensive CLI with many options:

#### Basic Usage

```bash
python main.py <input_image> [options]
```

#### Common Options

| Option | Description |
|--------|-------------|
| `-o, --output PATH` | Output file or directory path |
| `--visualize` | Show landmarks and palm region |
| `--show-landmarks` | Display hand landmarks on output |
| `--show-contour` | Display palm contour on output |
| `--side-by-side` | Create comparison view |
| `--display` | Show output in window |
| `--transparent` | Use transparent background (RGBA) |
| `--background R G B` | Custom background color (default: 0 0 0) |
| `--confidence FLOAT` | Detection confidence threshold (0.0-1.0) |
| `--blur INT` | Blur kernel size for smoothing (odd number) |
| `--palm-scale FLOAT` | Central palm region scale (0.1-1.0, default: 0.6) |
| `-v, --verbose` | Enable detailed output |
| `--measure-performance` | Show timing for each stage |

#### Examples

**Process with custom background color:**
```bash
python main.py input.jpg -o output.png --background 50 100 200
```

**Show landmarks and palm contour:**
```bash
python main.py input.jpg -o output.png --show-landmarks --show-contour
```

**Verbose mode with intermediate steps:**
```bash
python main.py input.jpg -o output.png -v --show-steps
```

**Save intermediate processing steps:**
```bash
python main.py input.jpg -o output.png -v --save-steps debug_output/
```

**Measure performance:**
```bash
python main.py input.jpg -o output.png --measure-performance
```

**Adjust detection confidence:**
```bash
python main.py input.jpg -o output.png --confidence 0.8
```

**Adjust central palm region size:**
```bash
# Smaller central palm area (more focused on center)
python main.py input.jpg -o output.png --palm-scale 0.5

# Larger palm area (closer to full palm)
python main.py input.jpg -o output.png --palm-scale 0.8
```

### Python API

You can also use the system programmatically in your Python code.

#### Basic Usage

```python
from src.palm_segmentation_pipeline import PalmSegmentationPipeline

# Initialize pipeline
pipeline = PalmSegmentationPipeline()

# Process image
result = pipeline.process_image("input.jpg")

if result.success:
    # Save output
    pipeline.save_output(result.output_image, "output.png")
    print(f"Palm center: {result.palm_region.center}")
    print(f"Palm area: {result.palm_region.area} pixels")
else:
    print(f"Error: {result.error_message}")
```

#### Advanced Usage

```python
from src.palm_segmentation_pipeline import PalmSegmentationPipeline
from src.image_loader import ImageLoader

# Initialize with custom parameters
pipeline = PalmSegmentationPipeline(
    min_detection_confidence=0.8,
    blur_kernel_size=7,
    blur_sigma=2.0,
    palm_scale=0.6  # Adjust central palm region size
)

# Process with transparent background
result = pipeline.process_image(
    "input.jpg",
    output_mode='RGBA',
    verbose=True,
    measure_performance=True
)

if result.success:
    # Load original for visualization
    image_loader = ImageLoader()
    original = image_loader.load_image("input.jpg")
    
    # Create visualization
    hand_landmarks = pipeline.hand_detector.detect_hand(original)
    vis_image = pipeline.visualize_landmarks(
        original,
        hand_landmarks,
        palm_region=result.palm_region,
        show_palm_center=True
    )
    
    # Create side-by-side comparison
    comparison = pipeline.create_side_by_side(
        original,
        result.output_image,
        add_labels=True
    )
    
    # Save outputs
    pipeline.save_output(result.output_image, "output.png")
    pipeline.save_output(vis_image, "visualization.png")
    pipeline.save_output(comparison, "comparison.png")
```

## API Documentation

### PalmSegmentationPipeline

Main orchestrator for the palm segmentation pipeline.

#### Constructor

```python
PalmSegmentationPipeline(
    min_detection_confidence: float = 0.7,
    blur_kernel_size: int = 5,
    blur_sigma: float = 1.0
)
```

**Parameters:**
- `min_detection_confidence`: Minimum confidence for hand detection (0.0-1.0)
- `blur_kernel_size`: Kernel size for mask edge smoothing (must be odd)
- `blur_sigma`: Standard deviation for Gaussian blur

#### Methods

##### process_image()

```python
process_image(
    image_path: str,
    output_mode: str = 'RGB',
    background_color: tuple = (0, 0, 0),
    verbose: bool = False,
    measure_performance: bool = False
) -> ProcessingResult
```

Process an image through the complete pipeline.

**Parameters:**
- `image_path`: Path to input image file
- `output_mode`: Output format - 'RGB' or 'RGBA'
- `background_color`: RGB tuple for background (ignored in RGBA mode)
- `verbose`: Store intermediate processing steps for debugging
- `measure_performance`: Measure and report timing for each stage

**Returns:** `ProcessingResult` object with:
- `success`: Boolean indicating success
- `output_image`: Processed image array or None
- `palm_region`: PalmRegion object or None
- `error_message`: Error description or None
- `intermediate_steps`: Dict of intermediate images (if verbose=True)
- `timing_info`: Dict of timing measurements (if measure_performance=True)

##### save_output()

```python
save_output(
    image: np.ndarray,
    output_path: str,
    create_dirs: bool = True
) -> bool
```

Save processed image to disk in PNG format.

**Parameters:**
- `image`: Image array to save (RGB or RGBA)
- `output_path`: Path where image should be saved
- `create_dirs`: Create parent directories if needed

**Returns:** True if successful, False otherwise

##### visualize_landmarks()

```python
visualize_landmarks(
    image: np.ndarray,
    hand_landmarks: HandLandmarks,
    palm_region: PalmRegion = None,
    show_palm_center: bool = True
) -> np.ndarray
```

Create visualization with hand landmarks and palm region.

**Parameters:**
- `image`: Input image in RGB format
- `hand_landmarks`: HandLandmarks object
- `palm_region`: Optional PalmRegion to visualize
- `show_palm_center`: Whether to mark palm center

**Returns:** Annotated image with landmarks and palm region

##### create_side_by_side()

```python
create_side_by_side(
    original_image: np.ndarray,
    masked_image: np.ndarray,
    add_labels: bool = True
) -> np.ndarray
```

Create side-by-side comparison of original and masked output.

**Parameters:**
- `original_image`: Original input image (RGB)
- `masked_image`: Processed image (RGB or RGBA)
- `add_labels`: Add text labels to each image

**Returns:** Combined side-by-side image in RGB format

##### display_output()

```python
display_output(
    image: np.ndarray,
    window_name: str = "Palm Segmentation Output",
    wait_key: int = 0
) -> None
```

Display output image using OpenCV window.

**Parameters:**
- `image`: Image to display (RGB or RGBA)
- `window_name`: Name of display window
- `wait_key`: Time to wait in ms (0 = wait for key press)

### ImageLoader

Handles image loading and validation.

```python
from src.image_loader import ImageLoader

loader = ImageLoader()
image = loader.load_image("input.jpg", resize_large=True)
```

### HandDetector

Detects hands and extracts landmarks using MediaPipe.

```python
from src.hand_detector import HandDetector

detector = HandDetector(min_detection_confidence=0.7)
hand_landmarks = detector.detect_hand(image)
```

### PalmExtractor

Extracts palm region from hand landmarks.

```python
from src.palm_extractor import PalmExtractor

extractor = PalmExtractor()
palm_region = extractor.extract_palm_region(landmarks, image_shape)
```

### MaskGenerator

Creates and applies palm masks.

```python
from src.mask_generator import MaskGenerator

generator = MaskGenerator(blur_kernel_size=5, blur_sigma=1.0)
mask = generator.create_palm_mask(image_shape, palm_contour)
output = generator.apply_mask(image, mask, background_color=(0, 0, 0))
```

### Data Models

#### HandLandmarks

```python
@dataclass
class HandLandmarks:
    landmarks: List[Tuple[int, int]]  # 21 (x, y) coordinates
    image_width: int
    image_height: int
    handedness: str  # "Left" or "Right"
```

#### PalmRegion

```python
@dataclass
class PalmRegion:
    contour: np.ndarray  # Polygon points defining palm boundary
    center: Tuple[int, int]  # Center point of palm
    area: float  # Area in pixels
```

#### ProcessingResult

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

## Examples

See `example.py` for comprehensive usage examples:

```bash
python example.py
```

The example script demonstrates:
1. Basic palm segmentation
2. Transparent background output
3. Custom background colors
4. Landmark visualization
5. Side-by-side comparisons
6. Batch processing
7. Custom detection parameters
8. Verbose mode with debugging
9. Performance measurement
10. Error handling

## Troubleshooting

### Common Issues

#### 1. No Hand Detected

**Problem:** Error message "No hand detected in the image"

**Solutions:**
- Ensure the hand is clearly visible in the image
- Check that lighting is adequate
- Position the hand in the center of the frame
- Try lowering the detection confidence: `--confidence 0.5`
- Ensure the hand is not too small in the frame

#### 2. Import Errors

**Problem:** `ModuleNotFoundError` or import errors, or "module 'mediapipe' has no attribute 'solutions'"

**Solutions:**
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (requires 3.8+)
- Ensure you're running from the project root directory
- Try reinstalling dependencies: `pip install --upgrade -r requirements.txt`
- If you see MediaPipe errors, ensure you have version 0.10.0 or higher: `pip install --upgrade mediapipe>=0.10.0`
- The system uses MediaPipe's new `tasks` API (0.10+), not the legacy `solutions` API

#### 3. Poor Segmentation Quality

**Problem:** Jagged edges or inaccurate palm boundaries

**Solutions:**
- Increase blur kernel size: `--blur 7` or `--blur 9`
- Use higher quality input images
- Ensure good lighting and contrast
- Try adjusting detection confidence: `--confidence 0.8`

#### 4. Slow Processing

**Problem:** Processing takes too long

**Solutions:**
- Large images are automatically resized to max 1920x1080
- Process images at lower resolution before input
- Use batch processing for multiple images
- Check performance with: `--measure-performance`

#### 5. Multiple Hands Detected

**Problem:** Unexpected results with multiple hands in image

**Solution:**
- The system processes the most prominent hand (highest confidence)
- For best results, use images with a single hand
- Crop images to focus on one hand

#### 6. File Not Found Errors

**Problem:** Cannot find input or output files

**Solutions:**
- Use absolute paths: `/full/path/to/image.jpg`
- Check current working directory
- Verify file extensions are correct (.jpg, .png, .jpeg)
- Ensure output directories exist or use `-o` to create them

#### 7. OpenCV Display Issues

**Problem:** Windows don't display or crash

**Solutions:**
- On headless systems, avoid `--display` and `--show-steps`
- Use `--save-steps` instead to save intermediate images
- Check OpenCV installation: `python -c "import cv2; print(cv2.__version__)"`

#### 8. Model Download Issues

**Problem:** "Failed to download hand landmarker model" error

**Solutions:**
- Ensure you have internet connectivity on first run
- Check firewall settings that might block downloads from storage.googleapis.com
- Manually download the model from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
- Place the downloaded file in `models/hand_landmarker.task` (create the `models/` directory if needed)
- The model file is approximately 10MB

### Getting Help

If you encounter issues not covered here:

1. Run with verbose mode to see detailed processing steps:
   ```bash
   python main.py input.jpg -o output.png -v
   ```

2. Check intermediate steps to identify where processing fails:
   ```bash
   python main.py input.jpg -o output.png -v --save-steps debug/
   ```

3. Measure performance to identify bottlenecks:
   ```bash
   python main.py input.jpg -o output.png --measure-performance
   ```

4. Test with the example script:
   ```bash
   python example.py
   ```

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: Minimum 2GB, recommended 4GB+
- **Disk Space**: ~500MB for dependencies
- **Camera**: Not required (processes static images)

### Performance Expectations

Typical processing times on modern hardware:
- Image loading: 10-50ms
- Hand detection: 100-300ms
- Palm extraction: 5-20ms
- Mask generation: 10-30ms
- **Total**: 150-400ms per image

For images larger than 1920x1080, automatic resizing adds 20-50ms.

## Requirements

See `requirements.txt` for the complete list of dependencies:

```
opencv-python>=4.5.0
mediapipe>=0.10.0
numpy>=1.19.0
pillow>=8.0.0
```

## Project Structure

```
palm-detection-segmentation/
├── src/
│   ├── __init__.py
│   ├── data_models.py              # Data structures
│   ├── image_loader.py             # Image loading and validation
│   ├── hand_detector.py            # Hand detection with MediaPipe
│   ├── palm_extractor.py           # Palm region extraction
│   ├── mask_generator.py           # Mask creation and application
│   └── palm_segmentation_pipeline.py  # Main pipeline orchestrator
├── tests/
│   ├── test_image_loader.py
│   ├── test_hand_detector.py
│   ├── test_palm_extractor.py
│   ├── test_mask_generator.py
│   └── test_pipeline_integration.py
├── data/
│   └── samples/                    # Sample images (add your own)
├── main.py                         # Command-line interface
├── example.py                      # Usage examples
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

- **MediaPipe**: Google's MediaPipe framework for hand detection
- **OpenCV**: Computer vision and image processing
- **NumPy**: Numerical computing

## Contact

[Add contact information here]
