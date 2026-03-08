# Visualization and Debugging Guide

This guide explains the visualization and debugging features available in the Palm Detection and Segmentation System.

## Overview

The system provides several visualization options to help you understand and debug the palm detection process:

1. **Landmark Visualization** - Display all 21 hand landmarks with connections
2. **Palm Contour Drawing** - Show the extracted palm region boundary
3. **Palm Center Marker** - Mark the calculated center point of the palm
4. **Verbose Mode** - Display detailed processing information and intermediate steps
5. **Side-by-Side Comparison** - Compare original and processed images

## Command-Line Usage

### Basic Visualization

Show hand landmarks on the output image:
```bash
python main.py input.jpg -o output.png --show-landmarks
```

Show palm contour on the output image:
```bash
python main.py input.jpg -o output.png --show-contour
```

Show both landmarks and palm region:
```bash
python main.py input.jpg -o output.png --visualize
```

### Verbose Mode

Enable verbose output with detailed processing information:
```bash
python main.py input.jpg -o output.png -v
```

This will print detailed information about each processing stage:
- Image dimensions and format
- Number of landmarks detected
- Hand handedness (left/right)
- Palm region center and area
- Mask statistics
- Output format details

### Display Intermediate Steps

Display each processing step in separate windows:
```bash
python main.py input.jpg -o output.png -v --show-steps
```

This will show:
1. Original image
2. Hand landmarks detected
3. Palm region identified
4. Binary mask
5. Final output

Press any key in each window to proceed to the next step.

### Save Intermediate Steps

Save all intermediate processing steps to disk:
```bash
python main.py input.jpg -o output.png -v --save-steps debug_output/
```

This creates a directory with the following files:
- `step_1_original.png` - Original input image
- `step_2_landmarks.png` - Image with hand landmarks overlay
- `step_3_palm_region.png` - Image with palm contour and center
- `step_4_mask.png` - Binary mask
- `step_5_output.png` - Final processed output

### Side-by-Side Comparison

Create a side-by-side comparison of original and processed images:
```bash
python main.py input.jpg -o output.png --side-by-side
```

### Combined Options

You can combine multiple visualization options:
```bash
python main.py input.jpg -o output.png -v --show-steps --save-steps debug/ --side-by-side
```

## Programmatic Usage

### Basic Landmark Visualization

```python
from src.palm_segmentation_pipeline import PalmSegmentationPipeline
from src.image_loader import ImageLoader

# Initialize pipeline
pipeline = PalmSegmentationPipeline()

# Load image
image_loader = ImageLoader()
image = image_loader.load_image("input.jpg")

# Detect hand
hand_landmarks = pipeline.hand_detector.detect_hand(image)

# Extract palm region
palm_region = pipeline.palm_extractor.extract_palm_region(
    hand_landmarks.landmarks,
    (image.shape[0], image.shape[1])
)

# Create visualization
vis_image = pipeline.visualize_landmarks(
    image,
    hand_landmarks,
    palm_region=palm_region,
    show_palm_center=True
)

# Display or save
pipeline.display_output(vis_image)
pipeline.save_output(vis_image, "visualization.png")
```

### Verbose Mode with Intermediate Steps

```python
# Process with verbose mode enabled
result = pipeline.process_image(
    "input.jpg",
    verbose=True
)

# Access intermediate steps
if result.intermediate_steps:
    # Display all steps
    pipeline.display_intermediate_steps(result.intermediate_steps)
    
    # Or save to disk
    pipeline.save_intermediate_steps(
        result.intermediate_steps,
        output_dir="debug_output",
        base_filename="step"
    )
    
    # Or access individual steps
    original = result.intermediate_steps['original_image']
    landmarks_vis = result.intermediate_steps['landmarks_visualization']
    palm_vis = result.intermediate_steps['palm_visualization']
    mask = result.intermediate_steps['mask']
    output = result.intermediate_steps['output_image']
```

### Side-by-Side Comparison

```python
# Load original image
original_image = image_loader.load_image("input.jpg")

# Process image
result = pipeline.process_image("input.jpg")

# Create side-by-side
side_by_side = pipeline.create_side_by_side(
    original_image,
    result.output_image,
    add_labels=True
)

# Save or display
pipeline.save_output(side_by_side, "comparison.png")
```

## Visualization Elements

### Hand Landmarks

The system detects 21 hand landmarks:
- **0**: Wrist
- **1-4**: Thumb (base to tip)
- **5-8**: Index finger (base to tip)
- **9-12**: Middle finger (base to tip)
- **13-16**: Ring finger (base to tip)
- **17-20**: Pinky (base to tip)

Landmarks are displayed as:
- Green circles for landmark points
- White numbers for landmark indices
- Blue lines connecting related landmarks

### Palm Region

The palm region is defined by landmarks 0, 5, 9, 13, and 17:
- Red contour showing the palm boundary
- Magenta circle marking the palm center
- Label "Palm Center" next to the center point

### Binary Mask

The binary mask shows:
- White pixels (255) for palm region
- Black pixels (0) for background
- Smooth edges from Gaussian blur

## Debugging Tips

### No Hand Detected

If no hand is detected:
1. Use `--visualize` to see if any landmarks are detected
2. Try adjusting `--confidence` threshold (lower values are more permissive)
3. Ensure good lighting and clear hand visibility
4. Check that the hand is not too small or too large in the frame

### Incorrect Palm Region

If the palm region looks wrong:
1. Use `-v --show-steps` to see each processing stage
2. Check the landmarks visualization to ensure correct detection
3. Verify that all 5 palm landmarks (0, 5, 9, 13, 17) are correctly positioned
4. Try different hand poses or camera angles

### Poor Mask Quality

If the mask has jagged edges:
1. Adjust `--blur` parameter (higher values = smoother edges)
2. Use `-v --save-steps` to examine the mask in detail
3. Check that the palm contour is smooth in the visualization

### Performance Issues

If processing is slow:
1. Use `-v` to see timing information for each stage
2. Consider resizing large images before processing
3. Check that MediaPipe is properly installed and using GPU if available

## Example Scripts

See `example_visualization.py` for complete working examples of all visualization features.

Run the examples:
```bash
python example_visualization.py
```

## Requirements

All visualization features require:
- OpenCV (cv2) for image display and manipulation
- NumPy for array operations
- MediaPipe for hand detection

No additional dependencies are needed for visualization features.
