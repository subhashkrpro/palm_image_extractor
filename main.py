#!/usr/bin/env python3
"""
Palm Detection and Segmentation System
Main entry point for the application
"""
import argparse
import sys
import os
from pathlib import Path
from typing import List
import cv2

from src.palm_segmentation_pipeline import PalmSegmentationPipeline
from src.image_loader import ImageLoader


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Palm Detection and Segmentation System - Extract palm regions from hand images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  python main.py input.jpg -o output.png
  
  # Process with visualization
  python main.py input.jpg -o output.png --visualize
  
  # Batch process multiple images
  python main.py image1.jpg image2.jpg image3.jpg -o output_dir/
  
  # Process with transparent background
  python main.py input.jpg -o output.png --transparent
  
  # Show landmarks and palm contour
  python main.py input.jpg -o output.png --show-landmarks --show-contour
  
  # Create side-by-side comparison
  python main.py input.jpg -o output.png --side-by-side
  
  # Display output without saving
  python main.py input.jpg --display
  
  # Verbose mode with intermediate steps display
  python main.py input.jpg -o output.png -v --show-steps
  
  # Verbose mode with intermediate steps saved to disk
  python main.py input.jpg -o output.png -v --save-steps debug_output/
        """
    )
    
    # Input arguments
    parser.add_argument(
        'input',
        nargs='+',
        help='Input image path(s). Can be single file or multiple files for batch processing.'
    )
    
    # Output arguments
    parser.add_argument(
        '-o', '--output',
        help='Output path. For single image: file path. For batch: directory path.'
    )
    
    # Visualization options
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable visualization mode with landmarks and palm region'
    )
    
    parser.add_argument(
        '--show-landmarks',
        action='store_true',
        help='Show hand landmarks on output image'
    )
    
    parser.add_argument(
        '--show-contour',
        action='store_true',
        help='Show palm contour on output image'
    )
    
    parser.add_argument(
        '--side-by-side',
        action='store_true',
        help='Create side-by-side comparison of original and processed image'
    )
    
    parser.add_argument(
        '--display',
        action='store_true',
        help='Display output in window (press any key to close)'
    )
    
    # Output format options
    parser.add_argument(
        '--transparent',
        action='store_true',
        help='Use transparent background (RGBA) instead of black background'
    )
    
    parser.add_argument(
        '--background',
        type=int,
        nargs=3,
        metavar=('R', 'G', 'B'),
        default=[0, 0, 0],
        help='Background color as RGB values (0-255). Default: 0 0 0 (black)'
    )
    
    # Detection parameters
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.7,
        help='Minimum detection confidence (0.0-1.0). Default: 0.7'
    )
    
    parser.add_argument(
        '--blur',
        type=int,
        default=5,
        help='Blur kernel size for mask smoothing (odd number). Default: 5'
    )
    
    parser.add_argument(
        '--palm-scale',
        type=float,
        default=0.6,
        help='Scale factor for central palm region (0.1-1.0). Lower = smaller central area. Default: 0.6'
    )
    
    # Progress and verbosity
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output with detailed progress information and intermediate steps'
    )
    
    parser.add_argument(
        '--show-steps',
        action='store_true',
        help='Display intermediate processing steps in separate windows (requires --verbose)'
    )
    
    parser.add_argument(
        '--save-steps',
        type=str,
        metavar='DIR',
        help='Save intermediate processing steps to specified directory (requires --verbose)'
    )
    
    parser.add_argument(
        '--measure-performance',
        action='store_true',
        help='Measure and report timing for each pipeline stage'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command-line arguments and return error message if invalid."""
    # Validate confidence threshold
    if not 0.0 <= args.confidence <= 1.0:
        return "Error: Confidence must be between 0.0 and 1.0"
    
    # Validate blur kernel size
    if args.blur < 1 or args.blur % 2 == 0:
        return "Error: Blur kernel size must be a positive odd number"
    
    # Validate palm scale
    if not 0.1 <= args.palm_scale <= 1.0:
        return "Error: Palm scale must be between 0.1 and 1.0"
    
    # Validate background color
    if not all(0 <= c <= 255 for c in args.background):
        return "Error: Background color values must be between 0 and 255"
    
    # Check if input files exist
    for input_path in args.input:
        if not os.path.exists(input_path):
            return f"Error: Input file not found: {input_path}"
        if not os.path.isfile(input_path):
            return f"Error: Input path is not a file: {input_path}"
    
    # Validate output path for batch processing
    if len(args.input) > 1 and args.output:
        # For batch processing, output should be a directory
        if os.path.exists(args.output) and not os.path.isdir(args.output):
            return "Error: For batch processing, output must be a directory path"
    
    return None


def get_output_path(input_path: str, output_arg: str, is_batch: bool) -> str:
    """Generate output path based on input and arguments."""
    if not output_arg:
        # Default: save next to input with _palm suffix
        input_path_obj = Path(input_path)
        return str(input_path_obj.parent / f"{input_path_obj.stem}_palm.png")
    
    if is_batch:
        # Batch mode: output_arg is directory
        input_filename = Path(input_path).stem
        return os.path.join(output_arg, f"{input_filename}_palm.png")
    else:
        # Single file mode: output_arg is file path
        return output_arg


def print_progress(message: str, verbose: bool, quiet: bool):
    """Print progress message if not in quiet mode."""
    if not quiet:
        if verbose:
            print(f"[INFO] {message}")
        else:
            print(message)


def print_error(message: str, quiet: bool):
    """Print error message unless in quiet mode."""
    if not quiet:
        print(f"[ERROR] {message}", file=sys.stderr)


def process_single_image(
    pipeline: PalmSegmentationPipeline,
    input_path: str,
    output_path: str,
    args,
    image_num: int = 1,
    total_images: int = 1
) -> bool:
    """
    Process a single image through the pipeline.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Progress message
        if total_images > 1:
            print_progress(
                f"Processing image {image_num}/{total_images}: {input_path}",
                args.verbose,
                args.quiet
            )
        else:
            print_progress(f"Processing: {input_path}", args.verbose, args.quiet)
        
        # Determine output mode
        output_mode = 'RGBA' if args.transparent else 'RGB'
        background_color = tuple(args.background)
        
        # Process image
        result = pipeline.process_image(
            input_path,
            output_mode=output_mode,
            background_color=background_color,
            verbose=args.verbose,
            measure_performance=args.measure_performance
        )
        
        # Check for errors
        if not result.success:
            print_error(f"Failed to process {input_path}: {result.error_message}", args.quiet)
            return False
        
        # Load original image for visualization if needed
        original_image = None
        if args.visualize or args.show_landmarks or args.show_contour or args.side_by_side:
            image_loader = ImageLoader()
            original_image = image_loader.load_image(input_path)
        
        # Create visualization if requested
        output_image = result.output_image
        
        if args.visualize or args.show_landmarks or args.show_contour:
            # Load hand landmarks for visualization
            hand_landmarks = pipeline.hand_detector.detect_hand(original_image)
            
            # Create visualization
            vis_image = pipeline.visualize_landmarks(
                original_image,
                hand_landmarks,
                palm_region=result.palm_region if (args.visualize or args.show_contour) else None,
                show_palm_center=(args.visualize or args.show_contour)
            )
            output_image = vis_image
        
        # Create side-by-side if requested
        if args.side_by_side:
            output_image = pipeline.create_side_by_side(
                original_image,
                result.output_image,
                add_labels=True
            )
        
        # Save output if path provided
        if output_path:
            print_progress(f"Saving to: {output_path}", args.verbose, args.quiet)
            success = pipeline.save_output(output_image, output_path)
            
            if not success:
                print_error(f"Failed to save output to {output_path}", args.quiet)
                return False
            
            if args.verbose:
                print_progress(f"Successfully saved: {output_path}", args.verbose, args.quiet)
        
        # Display output if requested
        if args.display:
            print_progress("Displaying output (press any key to close)", args.verbose, args.quiet)
            pipeline.display_output(output_image, window_name=f"Palm Segmentation - {Path(input_path).name}")
        
        # Display intermediate steps if requested
        if args.verbose and args.show_steps and result.intermediate_steps:
            print_progress("Displaying intermediate processing steps", args.verbose, args.quiet)
            pipeline.display_intermediate_steps(
                result.intermediate_steps,
                window_prefix=f"{Path(input_path).stem}"
            )
        
        # Save intermediate steps if requested
        if args.verbose and args.save_steps and result.intermediate_steps:
            steps_dir = os.path.join(args.save_steps, Path(input_path).stem)
            print_progress(f"Saving intermediate steps to: {steps_dir}", args.verbose, args.quiet)
            pipeline.save_intermediate_steps(
                result.intermediate_steps,
                steps_dir,
                base_filename="step"
            )
        
        return True
        
    except Exception as e:
        print_error(f"Unexpected error processing {input_path}: {str(e)}", args.quiet)
        return False


def main():
    """Main entry point for palm detection and segmentation"""
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    error_msg = validate_arguments(args)
    if error_msg:
        print_error(error_msg, args.quiet)
        sys.exit(1)
    
    # Print header
    if not args.quiet:
        print("=" * 60)
        print("Palm Detection and Segmentation System")
        print("=" * 60)
    
    # Initialize pipeline
    try:
        print_progress("Initializing pipeline...", args.verbose, args.quiet)
        pipeline = PalmSegmentationPipeline(
            min_detection_confidence=args.confidence,
            blur_kernel_size=args.blur,
            palm_scale=args.palm_scale
        )
        print_progress("Pipeline initialized successfully", args.verbose, args.quiet)
    except Exception as e:
        print_error(f"Failed to initialize pipeline: {str(e)}", args.quiet)
        sys.exit(1)
    
    # Determine if batch processing
    is_batch = len(args.input) > 1
    
    # Create output directory for batch processing
    if is_batch and args.output:
        os.makedirs(args.output, exist_ok=True)
        print_progress(f"Output directory: {args.output}", args.verbose, args.quiet)
    
    # Process images
    total_images = len(args.input)
    successful = 0
    failed = 0
    
    if not args.quiet:
        print()
    
    for idx, input_path in enumerate(args.input, 1):
        output_path = get_output_path(input_path, args.output, is_batch) if args.output or not args.display else None
        
        success = process_single_image(
            pipeline,
            input_path,
            output_path,
            args,
            image_num=idx,
            total_images=total_images
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Add separator between images in batch mode
        if is_batch and idx < total_images and not args.quiet:
            print()
    
    # Print summary
    if not args.quiet:
        print()
        print("=" * 60)
        print("Processing Complete")
        print("=" * 60)
        print(f"Total images: {total_images}")
        print(f"Successful: {successful}")
        if failed > 0:
            print(f"Failed: {failed}")
        print("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
