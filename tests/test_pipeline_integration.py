"""
Integration tests for PalmSegmentationPipeline.
Tests end-to-end processing, error propagation, and edge cases.
"""
import unittest
import os
import tempfile
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from src.palm_segmentation_pipeline import PalmSegmentationPipeline
from src.data_models import ProcessingResult, HandLandmarks, PalmRegion


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete palm segmentation pipeline."""
    
    def setUp(self):
        """Set up test fixtures and temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock MediaPipe to avoid dependency issues
        self.mp_patcher = patch('src.hand_detector.mp')
        self.mock_mp = self.mp_patcher.start()
        
        # Setup mock MediaPipe Hands
        self.mock_hands_instance = MagicMock()
        self.mock_mp.solutions.hands.Hands.return_value = self.mock_hands_instance
        
        self.pipeline = PalmSegmentationPipeline(min_detection_confidence=0.7)
    
    def tearDown(self):
        """Clean up temporary files and mocks."""
        # Clean up temp directory
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
        
        self.mp_patcher.stop()
        del self.pipeline
    
    def create_test_hand_image(self, filename: str = "test_hand.jpg") -> str:
        """
        Create a test image with a hand-like shape and save it.
        
        Args:
            filename: Name for the test image file
            
        Returns:
            Path to the created test image
        """
        # Create a realistic hand-like image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a hand-like shape with palm and fingers
        # Palm region
        cv2.rectangle(image, (250, 200), (390, 350), (220, 180, 150), -1)
        
        # Fingers
        cv2.rectangle(image, (260, 150), (290, 200), (220, 180, 150), -1)  # Index
        cv2.rectangle(image, (300, 140), (330, 200), (220, 180, 150), -1)  # Middle
        cv2.rectangle(image, (340, 150), (370, 200), (220, 180, 150), -1)  # Ring
        cv2.rectangle(image, (375, 170), (395, 200), (220, 180, 150), -1)  # Pinky
        cv2.rectangle(image, (220, 280), (250, 330), (220, 180, 150), -1)  # Thumb
        
        # Add some texture
        noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save image
        image_path = os.path.join(self.temp_dir, filename)
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        return image_path
    
    def create_test_image_no_hand(self, filename: str = "no_hand.jpg") -> str:
        """
        Create a test image without a hand.
        
        Args:
            filename: Name for the test image file
            
        Returns:
            Path to the created test image
        """
        # Create image with random objects but no hand
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some random shapes
        cv2.circle(image, (320, 240), 50, (100, 150, 200), -1)
        cv2.rectangle(image, (100, 100), (200, 150), (150, 100, 50), -1)
        
        image_path = os.path.join(self.temp_dir, filename)
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        return image_path
    
    def create_test_image_multiple_hands(self, filename: str = "multiple_hands.jpg") -> str:
        """
        Create a test image with multiple hands.
        
        Args:
            filename: Name for the test image file
            
        Returns:
            Path to the created test image
        """
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # First hand (left side)
        cv2.rectangle(image, (50, 200), (150, 350), (220, 180, 150), -1)
        cv2.rectangle(image, (60, 150), (80, 200), (220, 180, 150), -1)
        cv2.rectangle(image, (90, 140), (110, 200), (220, 180, 150), -1)
        
        # Second hand (right side)
        cv2.rectangle(image, (450, 200), (550, 350), (220, 180, 150), -1)
        cv2.rectangle(image, (460, 150), (480, 200), (220, 180, 150), -1)
        cv2.rectangle(image, (490, 140), (510, 200), (220, 180, 150), -1)
        
        image_path = os.path.join(self.temp_dir, filename)
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        return image_path
    
    def create_poor_quality_image(self, filename: str = "poor_quality.jpg") -> str:
        """
        Create a poor quality/blurry test image.
        
        Args:
            filename: Name for the test image file
            
        Returns:
            Path to the created test image
        """
        # Create a hand image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (250, 200), (390, 350), (220, 180, 150), -1)
        cv2.rectangle(image, (260, 150), (290, 200), (220, 180, 150), -1)
        
        # Apply heavy blur to simulate poor quality
        image = cv2.GaussianBlur(image, (51, 51), 0)
        
        # Add noise
        noise = np.random.randint(-50, 50, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        image_path = os.path.join(self.temp_dir, filename)
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        return image_path
    
    def mock_successful_hand_detection(self):
        """Configure mock to return successful hand detection."""
        mock_results = MagicMock()
        mock_hand_landmarks = MagicMock()
        
        # Create realistic hand landmarks (21 points)
        # These represent a typical hand pose
        landmark_positions = [
            (0.5, 0.7),   # 0: Wrist
            (0.45, 0.65), # 1: Thumb CMC
            (0.42, 0.6),  # 2: Thumb MCP
            (0.4, 0.55),  # 3: Thumb IP
            (0.38, 0.5),  # 4: Thumb tip
            (0.48, 0.5),  # 5: Index MCP
            (0.48, 0.4),  # 6: Index PIP
            (0.48, 0.3),  # 7: Index DIP
            (0.48, 0.2),  # 8: Index tip
            (0.52, 0.48), # 9: Middle MCP
            (0.52, 0.35), # 10: Middle PIP
            (0.52, 0.25), # 11: Middle DIP
            (0.52, 0.15), # 12: Middle tip
            (0.56, 0.5),  # 13: Ring MCP
            (0.56, 0.4),  # 14: Ring PIP
            (0.56, 0.3),  # 15: Ring DIP
            (0.56, 0.22), # 16: Ring tip
            (0.6, 0.52),  # 17: Pinky MCP
            (0.6, 0.45),  # 18: Pinky PIP
            (0.6, 0.38),  # 19: Pinky DIP
            (0.6, 0.32),  # 20: Pinky tip
        ]
        
        mock_landmarks = []
        for x, y in landmark_positions:
            mock_landmark = MagicMock()
            mock_landmark.x = x
            mock_landmark.y = y
            mock_landmarks.append(mock_landmark)
        
        mock_hand_landmarks.landmark = mock_landmarks
        mock_results.multi_hand_landmarks = [mock_hand_landmarks]
        
        # Mock handedness
        mock_handedness = MagicMock()
        mock_handedness.classification = [MagicMock()]
        mock_handedness.classification[0].label = "Right"
        mock_results.multi_handedness = [mock_handedness]
        
        self.mock_hands_instance.process.return_value = mock_results
    
    def mock_no_hand_detection(self):
        """Configure mock to return no hand detected."""
        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = None
        self.mock_hands_instance.process.return_value = mock_results
    
    # Test 1: End-to-end processing with sample hand images
    def test_end_to_end_processing_success(self):
        """Test complete pipeline processing with a valid hand image."""
        # Create test image
        image_path = self.create_test_hand_image()
        
        # Mock successful hand detection
        self.mock_successful_hand_detection()
        
        # Process image through pipeline
        result = self.pipeline.process_image(image_path)
        
        # Verify successful processing
        self.assertIsInstance(result, ProcessingResult)
        self.assertTrue(result.success, "Pipeline should succeed with valid hand image")
        self.assertIsNone(result.error_message)
        self.assertIsNotNone(result.output_image)
        self.assertIsNotNone(result.palm_region)
        
        # Verify output image properties
        self.assertIsInstance(result.output_image, np.ndarray)
        self.assertEqual(len(result.output_image.shape), 3, "Output should be 3D array")
        self.assertEqual(result.output_image.shape[2], 3, "Output should have 3 channels (RGB)")
        
        # Verify palm region properties
        self.assertIsInstance(result.palm_region, PalmRegion)
        self.assertIsNotNone(result.palm_region.contour)
        self.assertIsNotNone(result.palm_region.center)
        self.assertGreater(result.palm_region.area, 0)
    
    def test_end_to_end_with_rgba_output(self):
        """Test pipeline with RGBA output mode."""
        image_path = self.create_test_hand_image()
        self.mock_successful_hand_detection()
        
        # Process with RGBA output
        result = self.pipeline.process_image(image_path, output_mode='RGBA')
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.output_image)
        self.assertEqual(result.output_image.shape[2], 4, "Output should have 4 channels (RGBA)")
    
    def test_end_to_end_with_custom_background(self):
        """Test pipeline with custom background color."""
        image_path = self.create_test_hand_image()
        self.mock_successful_hand_detection()
        
        # Process with custom background color (blue)
        result = self.pipeline.process_image(
            image_path,
            background_color=(0, 0, 255)
        )
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.output_image)
        
        # Verify background has blue color (check corners which should be background)
        corner_pixel = result.output_image[0, 0]
        # Background should be close to blue (allowing for some blur effects)
        self.assertLess(corner_pixel[0], 50, "Red channel should be low")
        self.assertLess(corner_pixel[1], 50, "Green channel should be low")
    
    def test_pipeline_output_dimensions_match_input(self):
        """Test that output image dimensions match input image."""
        image_path = self.create_test_hand_image()
        self.mock_successful_hand_detection()
        
        # Load original image to get dimensions
        original = cv2.imread(image_path)
        original_height, original_width = original.shape[:2]
        
        result = self.pipeline.process_image(image_path)
        
        self.assertTrue(result.success)
        output_height, output_width = result.output_image.shape[:2]
        
        self.assertEqual(output_height, original_height)
        self.assertEqual(output_width, original_width)
    
    # Test 2: Error propagation through pipeline stages
    def test_error_propagation_file_not_found(self):
        """Test that file not found error propagates correctly."""
        # Try to process non-existent file
        result = self.pipeline.process_image("nonexistent_file.jpg")
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertIn("File not found", result.error_message)
        self.assertIsNone(result.output_image)
        self.assertIsNone(result.palm_region)
    
    def test_error_propagation_invalid_image(self):
        """Test that invalid image error propagates correctly."""
        # Create an invalid image file (text file with .jpg extension)
        invalid_path = os.path.join(self.temp_dir, "invalid.jpg")
        with open(invalid_path, 'w') as f:
            f.write("This is not an image")
        
        result = self.pipeline.process_image(invalid_path)
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertIn("Invalid image", result.error_message)
        self.assertIsNone(result.output_image)
        self.assertIsNone(result.palm_region)
    
    def test_error_propagation_no_hand_detected(self):
        """Test that no hand detected error propagates correctly."""
        image_path = self.create_test_image_no_hand()
        self.mock_no_hand_detection()
        
        result = self.pipeline.process_image(image_path)
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertIn("No hand detected", result.error_message)
        self.assertIsNone(result.output_image)
        self.assertIsNone(result.palm_region)
    
    def test_error_propagation_hand_detection_exception(self):
        """Test that exceptions in hand detection are caught and propagated."""
        image_path = self.create_test_hand_image()
        
        # Mock hand detector to raise an exception
        self.mock_hands_instance.process.side_effect = Exception("MediaPipe error")
        
        result = self.pipeline.process_image(image_path)
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertIn("Hand detection failed", result.error_message)
    
    def test_error_messages_are_informative(self):
        """Test that error messages provide helpful information."""
        # Test file not found
        result = self.pipeline.process_image("missing.jpg")
        self.assertIn("File not found", result.error_message)
        
        # Test no hand detected
        image_path = self.create_test_image_no_hand()
        self.mock_no_hand_detection()
        result = self.pipeline.process_image(image_path)
        self.assertIn("clearly visible", result.error_message)
        self.assertIn("good lighting", result.error_message)
    
    # Test 3: Edge cases
    def test_edge_case_no_hand_in_image(self):
        """Test processing image with no hand present."""
        image_path = self.create_test_image_no_hand()
        self.mock_no_hand_detection()
        
        result = self.pipeline.process_image(image_path)
        
        self.assertFalse(result.success)
        self.assertIn("No hand detected", result.error_message)
        self.assertIsNone(result.output_image)
        self.assertIsNone(result.palm_region)
    
    def test_edge_case_multiple_hands(self):
        """Test processing image with multiple hands (should process first hand)."""
        image_path = self.create_test_image_multiple_hands()
        
        # Mock detection of multiple hands (but detector returns only first)
        self.mock_successful_hand_detection()
        
        result = self.pipeline.process_image(image_path)
        
        # Should succeed with first detected hand
        self.assertTrue(result.success)
        self.assertIsNotNone(result.output_image)
        self.assertIsNotNone(result.palm_region)
    
    def test_edge_case_poor_quality_image(self):
        """Test processing poor quality/blurry image."""
        image_path = self.create_poor_quality_image()
        
        # Mock that hand is still detected despite poor quality
        self.mock_successful_hand_detection()
        
        result = self.pipeline.process_image(image_path)
        
        # Should still succeed if hand is detected
        self.assertTrue(result.success)
        self.assertIsNotNone(result.output_image)
    
    def test_edge_case_very_small_image(self):
        """Test processing very small image."""
        # Create tiny image
        small_image = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.rectangle(small_image, (10, 10), (40, 40), (220, 180, 150), -1)
        
        small_path = os.path.join(self.temp_dir, "small.jpg")
        cv2.imwrite(small_path, cv2.cvtColor(small_image, cv2.COLOR_RGB2BGR))
        
        self.mock_successful_hand_detection()
        
        result = self.pipeline.process_image(small_path)
        
        # Should handle small images
        self.assertTrue(result.success)
        self.assertEqual(result.output_image.shape[:2], (50, 50))
    
    def test_edge_case_large_image(self):
        """Test processing large image - should be resized for performance."""
        # Create large image
        large_image = np.zeros((2000, 2000, 3), dtype=np.uint8)
        cv2.rectangle(large_image, (800, 800), (1200, 1400), (220, 180, 150), -1)
        
        large_path = os.path.join(self.temp_dir, "large.jpg")
        cv2.imwrite(large_path, cv2.cvtColor(large_image, cv2.COLOR_RGB2BGR))
        
        self.mock_successful_hand_detection()
        
        result = self.pipeline.process_image(large_path)
        
        # Should handle large images and resize them for performance
        self.assertTrue(result.success)
        # Image should be resized to max 1920x1080 while maintaining aspect ratio
        # For a 2000x2000 image, it should be resized to 1080x1080
        self.assertEqual(result.output_image.shape[:2], (1080, 1080))
    
    def test_visualize_landmarks_with_valid_data(self):
        """Test landmark visualization functionality."""
        image_path = self.create_test_hand_image()
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create mock hand landmarks
        landmarks = [(int(320 + i*5), int(240 + i*3)) for i in range(21)]
        hand_landmarks = HandLandmarks(
            landmarks=landmarks,
            image_width=640,
            image_height=480,
            handedness="Right"
        )
        
        # Create mock palm region
        palm_contour = np.array([[300, 250], [350, 250], [350, 300], [300, 300]])
        palm_region = PalmRegion(
            contour=palm_contour,
            center=(325, 275),
            area=2500.0
        )
        
        # Visualize
        vis_image = self.pipeline.visualize_landmarks(
            image,
            hand_landmarks,
            palm_region,
            show_palm_center=True
        )
        
        self.assertIsNotNone(vis_image)
        self.assertEqual(vis_image.shape, image.shape)
        self.assertIsInstance(vis_image, np.ndarray)
    
    def test_visualize_landmarks_without_palm_region(self):
        """Test landmark visualization without palm region."""
        image_path = self.create_test_hand_image()
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        landmarks = [(int(320 + i*5), int(240 + i*3)) for i in range(21)]
        hand_landmarks = HandLandmarks(
            landmarks=landmarks,
            image_width=640,
            image_height=480,
            handedness="Left"
        )
        
        # Visualize without palm region
        vis_image = self.pipeline.visualize_landmarks(image, hand_landmarks)
        
        self.assertIsNotNone(vis_image)
        self.assertEqual(vis_image.shape, image.shape)


if __name__ == '__main__':
    unittest.main()
