#!/usr/bin/env python3
"""
Example script demonstrating visualization and debugging features.

This script shows how to use the various visualization options available
in the palm segmentation pipeline.
"""
import sys
from pathlib import Path

from src.palm_segmentation_pipeline import PalmSegmentationPipeline
from src.image_loader import ImageLoader


def example_basic_visualization():
    """Example 1: Basic visualization with landmarks and palm contour."""
    print("=" * 60)
    print("Example 1: Basic Visualization")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = PalmSegmentationPipeline()
    
    # Process image
    image_path = "data/sample_hand.jpg"  # Replace with your image path
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        print("Please provide a valid hand image path")
        return
    
    # Load original image
    image_loader = ImageLoader()
    original_image = image_loader.load_image(image_path)
    
    # Detect hand
    hand_landmarks = pipeline.hand_detector.detect_hand(original_image)
    
    if hand_landmarks is None:
        print("No hand detected in image")
        return
    
    # Extract palm region
    palm_region = pipeline.palm_extractor.extract_palm_region(
        hand_landmarks.landmarks,
        (original_image.shape[0], original_image.shape[1])
    )
    
    # Create visualization
    vis_image = pipeline.visualize_landmarks(
        original_image,
        hand_landmarks,
        palm_region=palm_region,
        show_palm_center=True
    )
    
    # Display
    pipeline.display_output(vis_image, window_name="Landmarks and Palm Region")
    
    print("Visualization displayed successfully")


def example_verbose_mode():
    """Example 2: Verbose mode with intermediate steps."""
    print("\n" + "=" * 60)
    print("Example 2: Verbose Mode with Intermediate Steps")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = PalmSegmentationPipeline()
    
    # Process image with verbose mode
    image_path = "data/sample_hand.jpg"  # Replace with your image path
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        print("Please provide a valid hand image path")
        return
    
    print("\nProcessing with verbose mode enabled...")
    print("-" * 60)
    
    result = pipeline.process_image(
        image_path,
        verbose=True
    )
    
    if not result.success:
        print(f"Processing failed: {result.error_message}")
        return
    
    print("-" * 60)
    print("\nProcessing completed successfully!")
    
    # Display intermediate steps
    if result.intermediate_steps:
        print("\nDisplaying intermediate processing steps...")
        pipeline.display_intermediate_steps(result.intermediate_steps)


def example_save_intermediate_steps():
    """Example 3: Save intermediate steps to disk."""
    print("\n" + "=" * 60)
    print("Example 3: Save Intermediate Steps")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = PalmSegmentationPipeline()
    
    # Process image with verbose mode
    image_path = "data/sample_hand.jpg"  # Replace with your image path
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        print("Please provide a valid hand image path")
        return
    
    result = pipeline.process_image(
        image_path,
        verbose=True
    )
    
    if not result.success:
        print(f"Processing failed: {result.error_message}")
        return
    
    # Save intermediate steps
    if result.intermediate_steps:
        output_dir = "debug_output"
        print(f"\nSaving intermediate steps to {output_dir}/...")
        success = pipeline.save_intermediate_steps(
            result.intermediate_steps,
            output_dir,
            base_filename="example"
        )
        
        if success:
            print("\nIntermediate steps saved successfully!")
            print(f"Check the '{output_dir}' directory for the images")


def example_side_by_side():
    """Example 4: Create side-by-side comparison."""
    print("\n" + "=" * 60)
    print("Example 4: Side-by-Side Comparison")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = PalmSegmentationPipeline()
    
    # Process image
    image_path = "data/sample_hand.jpg"  # Replace with your image path
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        print("Please provide a valid hand image path")
        return
    
    # Load original image
    image_loader = ImageLoader()
    original_image = image_loader.load_image(image_path)
    
    # Process image
    result = pipeline.process_image(image_path)
    
    if not result.success:
        print(f"Processing failed: {result.error_message}")
        return
    
    # Create side-by-side visualization
    side_by_side = pipeline.create_side_by_side(
        original_image,
        result.output_image,
        add_labels=True
    )
    
    # Display
    pipeline.display_output(side_by_side, window_name="Side-by-Side Comparison")
    
    # Save
    pipeline.save_output(side_by_side, "output/side_by_side_comparison.png")
    print("\nSide-by-side comparison saved to output/side_by_side_comparison.png")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Palm Segmentation Visualization Examples")
    print("=" * 60)
    print("\nNote: Replace 'data/sample_hand.jpg' with your own hand image")
    print("Press any key in the display windows to continue\n")
    
    try:
        # Run examples
        example_basic_visualization()
        example_verbose_mode()
        example_save_intermediate_steps()
        example_side_by_side()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
