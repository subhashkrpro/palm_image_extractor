# Palm Detection CLI Usage Guide

## Overview

The `main.py` script provides a comprehensive command-line interface for the Palm Detection and Segmentation System. It supports single image processing, batch processing, various visualization options, and flexible output configurations.

## Basic Usage

### Process a Single Image

```bash
python main.py input.jpg -o output.png
```

### Display Help

```bash
python main.py --help
```

## Command-Line Arguments

### Required Arguments

- `input`: One or more input image paths (JPG, PNG, JPEG formats)

### Optional Arguments

#### Output Options

- `-o, --output OUTPUT`: Output path
  - For single image: specify file path
  - For batch processing: specify directory path
  - If omitted: saves next to input with `_palm` suffix

#### Visualization Options

- `--visualize`: Enable full visualization mode with landmarks and palm region
- `--show-landmarks`: Show hand landmarks on output image
- `--show-contour`: Show palm contour on output image
- `--side-by-side`: Create side-by-side comparison of original and processed image
- `--display`: Display output in window (press any key to close)

#### Output Format Options

- `--transparent`: Use transparent background (RGBA) instead of black background
- `--background R G B`: Custom background color as RGB values (0-255)
  - Default: `0 0 0` (black)
  - Example: `--background 255 255 255` (white)

#### Detection Parameters

- `--confidence CONFIDENCE`: Minimum detection confidence (0.0-1.0)
  - Default: `0.7`
  - Higher values = stricter detection
  
- `--blur BLUR`: Blur kernel size for mask smoothing (must be odd number)
  - Default: `5`
  - Higher values = smoother edges

#### Progress and Verbosity

- `-v, --verbose`: Enable verbose output with detailed progress information
- `--quiet`: Suppress all output except errors

## Usage Examples

### 1. Basic Processing

Process a single image with default settings:

```bash
python main.py hand_image.jpg -o output.png
```

### 2. Batch Processing

Process multiple images at once:

```bash
python main.py image1.jpg image2.jpg image3.jpg -o output_directory/
```

Process all JPG files in a directory (using shell expansion):

```bash
python main.py data/*.jpg -o processed/
```

### 3. Visualization Options

Show hand landmarks on the output:

```bash
python main.py input.jpg -o output.png --show-landmarks
```

Show palm contour:

```bash
python main.py input.jpg -o output.png --show-contour
```

Full visualization mode (landmarks + contour + center):

```bash
python main.py input.jpg -o output.png --visualize
```

Create side-by-side comparison:

```bash
python main.py input.jpg -o comparison.png --side-by-side
```

### 4. Output Format Options

Create output with transparent background:

```bash
python main.py input.jpg -o output.png --transparent
```

Use custom background color (white):

```bash
python main.py input.jpg -o output.png --background 255 255 255
```

Use custom background color (blue):

```bash
python main.py input.jpg -o output.png --background 0 0 255
```

### 5. Display Without Saving

Display the result in a window without saving:

```bash
python main.py input.jpg --display
```

### 6. Adjust Detection Parameters

Use higher confidence threshold for stricter detection:

```bash
python main.py input.jpg -o output.png --confidence 0.9
```

Use larger blur kernel for smoother mask edges:

```bash
python main.py input.jpg -o output.png --blur 9
```

### 7. Verbose and Quiet Modes

Verbose mode with detailed progress:

```bash
python main.py input.jpg -o output.png --verbose
```

Quiet mode (only errors):

```bash
python main.py input.jpg -o output.png --quiet
```

### 8. Combined Options

Process with multiple options:

```bash
python main.py input.jpg -o output.png \
  --transparent \
  --show-landmarks \
  --confidence 0.8 \
  --blur 7 \
  --verbose
```

Batch process with visualization:

```bash
python main.py *.jpg -o processed/ \
  --side-by-side \
  --verbose
```

## Error Handling

The CLI provides clear error messages for common issues:

### File Not Found

```bash
$ python main.py nonexistent.jpg -o output.png
[ERROR] Error: Input file not found: nonexistent.jpg
```

### Invalid Confidence Value

```bash
$ python main.py input.jpg --confidence 1.5
[ERROR] Error: Confidence must be between 0.0 and 1.0
```

### Invalid Blur Kernel

```bash
$ python main.py input.jpg --blur 4
[ERROR] Error: Blur kernel size must be a positive odd number
```

### No Hand Detected

```bash
$ python main.py no_hand.jpg -o output.png
[ERROR] Failed to process no_hand.jpg: No hand detected in the image. 
Please ensure the hand is clearly visible with good lighting.
```

## Output Summary

After processing, the CLI displays a summary:

```
============================================================
Processing Complete
============================================================
Total images: 3
Successful: 3
============================================================
```

If any images fail:

```
============================================================
Processing Complete
============================================================
Total images: 5
Successful: 4
Failed: 1
============================================================
```

## Exit Codes

- `0`: Success (all images processed successfully)
- `1`: Failure (one or more images failed or invalid arguments)

## Tips

1. **Batch Processing**: Use shell wildcards for easy batch processing
   ```bash
   python main.py data/*.jpg -o output/
   ```

2. **Testing**: Use `--display` to quickly preview results without saving
   ```bash
   python main.py test.jpg --display
   ```

3. **Debugging**: Use `--verbose` and `--visualize` together for detailed feedback
   ```bash
   python main.py input.jpg -o output.png --verbose --visualize
   ```

4. **Quality Control**: Adjust `--confidence` based on your image quality
   - Low quality images: use `--confidence 0.5`
   - High quality images: use `--confidence 0.8` or higher

5. **Smooth Masks**: Increase `--blur` for smoother mask edges
   ```bash
   python main.py input.jpg -o output.png --blur 11
   ```
