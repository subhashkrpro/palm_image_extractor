#!/usr/bin/env python3
"""
Example demonstrating output generation and saving functionality.
This shows how to use save_output(), create_side_by_side(), and display_output().
"""
import sys
from src.palm_segmentation_pipeline import PalmSegmentationPipeline

def main():
    if len(sys.argv) < 2:
        print("Usage: python example_output_usage.py <input_image_path>")
        print("\nExample:")
        print("  python example_output_usage.py data/hand_image.jpg")
        return
    
    input_path = sys.argv[1]
    
    # Initialize pipeline
    print("Initializing palm segmentation pipeline...")
    pipeline = PalmSegmentationPipeline(min_detection_confidence=0.7)
    
    # Process the image
    print(f"\nProcessing image: {input_path}")
    result = pipeline.process_image(input_path, output_mode='RGB')
    
    if not result.success:
        print(f"Error: {result.error_message}")
        return
    
    print("✓ Palm segmentation successful!")
    
    # Example 1: Save the masked output
    print("\n1. Saving masked palm output...")
    output_path = "data/output_masked.png"
    if pipeline.save_output(result.output_image, output_path):
        print(f"   ✓ Saved to: {output_path}")
    
    # Example 2: Save with RGBA (transparent background)
    print("\n2. Processing with transparent background...")
    result_rgba = pipeline.process_image(input_path, output_mode='RGBA')
    if result_rgba.success:
        output_path_rgba = "data/output_transparent.png"
        if pipeline.save_output(result_rgba.output_image, output_path_rgba):
            print(f"   ✓ Saved to: {output_path_rgba}")
    
    # Example 3: Create and save side-by-side comparison
    print("\n3. Creating side-by-side comparison...")
    original_image = pipeline.image_loader.load_image(input_path)
    side_by_side = pipeline.create_side_by_side(
        original_image,
        result.output_image,
        add_labels=True
    )
    output_path_sbs = "data/output_comparison.png"
    if pipeline.save_output(side_by_side, output_path_sbs):
        print(f"   ✓ Saved to: {output_path_sbs}")
    
    # Example 4: Display output (optional - uncomment to use)
    # print("\n4. Displaying output (press any key to close)...")
    # pipeline.display_output(result.output_image, "Palm Segmentation Result")
    
    print("\n✓ All outputs generated successfully!")
    print("\nGenerated files:")
    print("  - data/output_masked.png (palm with black background)")
    print("  - data/output_transparent.png (palm with transparent background)")
    print("  - data/output_comparison.png (side-by-side comparison)")

if __name__ == "__main__":
    main()
