#!/usr/bin/env python3
"""
Example usage script for Palm Detection and Segmentation System

This script demonstrates various ways to use the palm segmentation pipeline,
including basic usage, batch processing, and advanced features.
"""

import os
from src.palm_segmentation_pipeline import PalmSegmentationPipeline
from src.image_loader import ImageLoader


def example_1_basic_usage():
    """Example 1: Basic palm segmentation with default settings"""
    print("\n" + "="*60)
    print("Example 1: Basic Palm Segmentation")
    print("="*60)
    
    # Initialize the pipeline
    pipeline = PalmSegmentationPipeline()
    
    # Process an image
    result = pipeline.process_image(
        image_path="data/samples/hand1.jpg",
        output_mode='RGB',
        background_color=(0, 0, 0)
    )
    
    # Check result
    if result.success:
        print("✓ Palm segmentation successful!")
        print(f"  Palm center: {result.palm_region.center}")
        print(f"  Palm area: {result.palm_region.area:.2f} pixels")
        
        # Save output
        pipeline.save_output(result.output_image, "output/example1_basic.png")
        print("  Output saved to: output/example1_basic.png")
    else:
        print(f"✗ Processing failed: {result.error_message}")


def example_2_transparent_background():
    """Example 2: Generate output with transparent background"""
    print("\n" + "="*60)
    print("Example 2: Transparent Background (RGBA)")
    print("="*60)
    
    pipeline = PalmSegmentationPipeline()
    
    # Process with RGBA output mode
    result = pipeline.process_image(
        image_path="data/samples/hand1.jpg",
        output_mode='RGBA'
    )
    
    if result.success:
        print("✓ Generated RGBA output with transparent background")
        pipeline.save_output(result.output_image, "output/example2_transparent.png")
        print("  Output saved to: output/example2_transparent.png")
    else:
        print(f"✗ Processing failed: {result.error_message}")


def example_3_custom_background():
    """Example 3: Use custom background color"""
    print("\n" + "="*60)
    print("Example 3: Custom Background Color")
    print("="*60)
    
    pipeline = PalmSegmentationPipeline()
    
    # Process with custom blue background
    result = pipeline.process_image(
        image_path="data/samples/hand1.jpg",
        output_mode='RGB',
        background_color=(50, 100, 200)  # Blue background
    )
    
    if result.success:
        print("✓ Applied custom blue background")
        pipeline.save_output(result.output_image, "output/example3_custom_bg.png")
        print("  Output saved to: output/example3_custom_bg.png")
    else:
        print(f"✗ Processing failed: {result.error_message}")


def example_4_visualization():
    """Example 4: Visualize landmarks and palm region"""
    print("\n" + "="*60)
    print("Example 4: Landmark Visualization")
    print("="*60)
    
    pipeline = PalmSegmentationPipeline()
    image_loader = ImageLoader()
    
    # Load original image
    original_image = image_loader.load_image("data/samples/hand1.jpg")
    
    # Process image
    result = pipeline.process_image("data/samples/hand1.jpg")
    
    if result.success:
        # Detect hand landmarks for visualization
        hand_landmarks = pipeline.hand_detector.detect_hand(original_image)
        
        # Create visualization with landmarks and palm region
        vis_image = pipeline.visualize_landmarks(
            original_image,
            hand_landmarks,
            palm_region=result.palm_region,
            show_palm_center=True
        )
        
        print("✓ Created visualization with landmarks and palm contour")
        pipeline.save_output(vis_image, "output/example4_visualization.png")
        print("  Output saved to: output/example4_visualization.png")
    else:
        print(f"✗ Processing failed: {result.error_message}")


def example_5_side_by_side():
    """Example 5: Create side-by-side comparison"""
    print("\n" + "="*60)
    print("Example 5: Side-by-Side Comparison")
    print("="*60)
    
    pipeline = PalmSegmentationPipeline()
    image_loader = ImageLoader()
    
    # Load original image
    original_image = image_loader.load_image("data/samples/hand1.jpg")
    
    # Process image
    result = pipeline.process_image("data/samples/hand1.jpg")
    
    if result.success:
        # Create side-by-side comparison
        comparison = pipeline.create_side_by_side(
            original_image,
            result.output_image,
            add_labels=True
        )
        
        print("✓ Created side-by-side comparison")
        pipeline.save_output(comparison, "output/example5_comparison.png")
        print("  Output saved to: output/example5_comparison.png")
    else:
        print(f"✗ Processing failed: {result.error_message}")


def example_6_batch_processing():
    """Example 6: Process multiple images"""
    print("\n" + "="*60)
    print("Example 6: Batch Processing")
    print("="*60)
    
    pipeline = PalmSegmentationPipeline()
    
    # List of images to process
    image_files = [
        "data/samples/hand1.jpg",
        "data/samples/hand2.jpg",
        "data/samples/hand3.jpg"
    ]
    
    # Process each image
    successful = 0
    failed = 0
    
    for idx, image_path in enumerate(image_files, 1):
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"  [{idx}/{len(image_files)}] Skipping {image_path} (file not found)")
            failed += 1
            continue
        
        print(f"  [{idx}/{len(image_files)}] Processing {image_path}...")
        
        result = pipeline.process_image(image_path)
        
        if result.success:
            output_path = f"output/example6_batch_{idx}.png"
            pipeline.save_output(result.output_image, output_path)
            print(f"      ✓ Saved to {output_path}")
            successful += 1
        else:
            print(f"      ✗ Failed: {result.error_message}")
            failed += 1
    
    print(f"\nBatch processing complete: {successful} successful, {failed} failed")


def example_7_custom_parameters():
    """Example 7: Use custom detection parameters"""
    print("\n" + "="*60)
    print("Example 7: Custom Detection Parameters")
    print("="*60)
    
    # Initialize pipeline with custom parameters
    pipeline = PalmSegmentationPipeline(
        min_detection_confidence=0.8,  # Higher confidence threshold
        blur_kernel_size=7,            # Larger blur for smoother edges
        blur_sigma=2.0                 # Stronger blur effect
    )
    
    result = pipeline.process_image("data/samples/hand1.jpg")
    
    if result.success:
        print("✓ Processed with custom parameters:")
        print("  - Detection confidence: 0.8")
        print("  - Blur kernel size: 7")
        print("  - Blur sigma: 2.0")
        pipeline.save_output(result.output_image, "output/example7_custom_params.png")
        print("  Output saved to: output/example7_custom_params.png")
    else:
        print(f"✗ Processing failed: {result.error_message}")


def example_8_verbose_mode():
    """Example 8: Enable verbose mode for debugging"""
    print("\n" + "="*60)
    print("Example 8: Verbose Mode with Intermediate Steps")
    print("="*60)
    
    pipeline = PalmSegmentationPipeline()
    
    # Process with verbose mode enabled
    result = pipeline.process_image(
        image_path="data/samples/hand1.jpg",
        verbose=True
    )
    
    if result.success:
        print("\n✓ Processing completed with verbose output")
        
        # Save intermediate steps
        if result.intermediate_steps:
            print("\nSaving intermediate steps...")
            pipeline.save_intermediate_steps(
                result.intermediate_steps,
                "output/example8_steps",
                base_filename="debug"
            )
    else:
        print(f"✗ Processing failed: {result.error_message}")


def example_9_performance_measurement():
    """Example 9: Measure processing performance"""
    print("\n" + "="*60)
    print("Example 9: Performance Measurement")
    print("="*60)
    
    pipeline = PalmSegmentationPipeline()
    
    # Process with performance measurement
    result = pipeline.process_image(
        image_path="data/samples/hand1.jpg",
        measure_performance=True
    )
    
    if result.success:
        print("\n✓ Processing completed with timing measurements")
        print("  (See timing report above)")
        pipeline.save_output(result.output_image, "output/example9_performance.png")
    else:
        print(f"✗ Processing failed: {result.error_message}")


def example_10_error_handling():
    """Example 10: Demonstrate error handling"""
    print("\n" + "="*60)
    print("Example 10: Error Handling")
    print("="*60)
    
    pipeline = PalmSegmentationPipeline()
    
    # Test with non-existent file
    print("\n  Test 1: Non-existent file")
    result = pipeline.process_image("nonexistent.jpg")
    if not result.success:
        print(f"  ✓ Handled error: {result.error_message}")
    
    # Test with image containing no hand
    print("\n  Test 2: Image with no hand")
    # This would require an actual image without a hand
    # result = pipeline.process_image("data/samples/no_hand.jpg")
    # if not result.success:
    #     print(f"  ✓ Handled error: {result.error_message}")
    print("  (Requires test image without hand)")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print(" Palm Detection and Segmentation - Example Usage")
    print("="*70)
    print("\nThis script demonstrates various features of the palm segmentation system.")
    print("Make sure you have sample images in data/samples/ directory.")
    print("\nNote: Some examples may fail if sample images are not available.")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Run examples
    try:
        example_1_basic_usage()
        example_2_transparent_background()
        example_3_custom_background()
        example_4_visualization()
        example_5_side_by_side()
        example_6_batch_processing()
        example_7_custom_parameters()
        example_8_verbose_mode()
        example_9_performance_measurement()
        example_10_error_handling()
        
        print("\n" + "="*70)
        print(" All examples completed!")
        print("="*70)
        print("\nCheck the 'output/' directory for generated images.")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {str(e)}")
        print("\nMake sure you have:")
        print("  1. Installed all dependencies (pip install -r requirements.txt)")
        print("  2. Sample images in data/samples/ directory")


if __name__ == "__main__":
    main()
