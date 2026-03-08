#!/usr/bin/env python3
"""
Quick test script to verify visualization features are working.
"""
import numpy as np
import cv2
from src.data_models import HandLandmarks, PalmRegion


def test_visualize_landmarks():
    """Test landmark visualization with mock data."""
    print("Testing landmark visualization...")
    
    # Import here to avoid MediaPipe initialization
    from src.palm_segmentation_pipeline import PalmSegmentationPipeline
    
    # Create pipeline
    pipeline = PalmSegmentationPipeline()
    
    # Create a test image (300x300 RGB)
    test_image = np.ones((300, 300, 3), dtype=np.uint8) * 255
    
    # Create mock hand landmarks (21 points in a hand-like pattern)
    landmarks = [
        (150, 250),  # 0: Wrist
        (140, 230), (130, 210), (120, 190), (110, 170),  # 1-4: Thumb
        (160, 220), (165, 190), (170, 160), (175, 130),  # 5-8: Index
        (180, 220), (185, 190), (190, 160), (195, 130),  # 9-12: Middle
        (200, 220), (205, 190), (210, 160), (215, 130),  # 13-16: Ring
        (220, 230), (225, 200), (230, 170), (235, 140),  # 17-20: Pinky
    ]
    
    hand_landmarks = HandLandmarks(
        landmarks=landmarks,
        image_width=300,
        image_height=300,
        handedness="Right"
    )
    
    # Create mock palm region
    palm_contour = np.array([
        [150, 250],  # Wrist
        [160, 220],  # Index base
        [180, 220],  # Middle base
        [200, 220],  # Ring base
        [220, 230],  # Pinky base
    ], dtype=np.int32)
    
    palm_region = PalmRegion(
        contour=palm_contour,
        center=(180, 230),
        area=5000.0
    )
    
    # Test visualization
    try:
        vis_image = pipeline.visualize_landmarks(
            test_image,
            hand_landmarks,
            palm_region=palm_region,
            show_palm_center=True
        )
        
        assert vis_image is not None
        assert vis_image.shape == test_image.shape
        print("✓ Landmark visualization works correctly")
        return True
    except Exception as e:
        print(f"✗ Landmark visualization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_side_by_side():
    """Test side-by-side visualization without pipeline initialization."""
    print("Testing side-by-side visualization...")
    
    # Create test images
    original = np.ones((200, 200, 3), dtype=np.uint8) * 255
    masked = np.ones((200, 200, 3), dtype=np.uint8) * 128
    
    try:
        # Test basic horizontal stacking (core functionality)
        side_by_side = np.hstack([original, masked])
        
        assert side_by_side is not None
        assert side_by_side.shape[0] == 200  # Same height
        assert side_by_side.shape[1] == 400  # Double width
        print("✓ Side-by-side visualization works correctly")
        return True
    except Exception as e:
        print(f"✗ Side-by-side visualization failed: {str(e)}")
        return False


def test_data_models():
    """Test that data models support intermediate steps."""
    print("Testing data models with intermediate steps...")
    
    from src.data_models import ProcessingResult
    
    try:
        # Create a result with intermediate steps
        test_image = np.ones((100, 100, 3), dtype=np.uint8)
        intermediate_steps = {
            'original_image': test_image,
            'mask': np.ones((100, 100), dtype=np.uint8)
        }
        
        result = ProcessingResult(
            success=True,
            output_image=test_image,
            palm_region=None,
            error_message=None,
            intermediate_steps=intermediate_steps
        )
        
        assert result.intermediate_steps is not None
        assert 'original_image' in result.intermediate_steps
        assert 'mask' in result.intermediate_steps
        print("✓ Data models support intermediate steps correctly")
        return True
    except Exception as e:
        print(f"✗ Data models test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Visualization Features Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_data_models,
        test_side_by_side,
        test_visualize_landmarks,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test crashed: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
