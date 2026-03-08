"""
Unit tests for HandDetector component.
"""
import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import cv2
from src.hand_detector import HandDetector, HandLandmarks


class TestHandDetector(unittest.TestCase):
    """Test cases for HandDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock MediaPipe to avoid dependency issues in tests
        self.mp_hands_patcher = patch('src.hand_detector.mp')
        self.mock_mp = self.mp_hands_patcher.start()
        
        # Setup mock MediaPipe Hands
        self.mock_hands_instance = MagicMock()
        self.mock_mp.solutions.hands.Hands.return_value = self.mock_hands_instance
        
        self.detector = HandDetector(min_detection_confidence=0.7)
    
    def tearDown(self):
        """Clean up after tests."""
        self.mp_hands_patcher.stop()
        del self.detector
    
    def create_test_image_with_hand(self) -> np.ndarray:
        """
        Create a simple test image with a hand-like shape.
        This creates a white hand silhouette on a black background.
        """
        # Create a blank image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a hand-like shape (simplified palm and fingers)
        # Palm region
        cv2.rectangle(image, (250, 200), (390, 350), (255, 255, 255), -1)
        
        # Fingers
        cv2.rectangle(image, (260, 150), (290, 200), (255, 255, 255), -1)  # Index
        cv2.rectangle(image, (300, 140), (330, 200), (255, 255, 255), -1)  # Middle
        cv2.rectangle(image, (340, 150), (370, 200), (255, 255, 255), -1)  # Ring
        cv2.rectangle(image, (375, 170), (395, 200), (255, 255, 255), -1)  # Pinky
        cv2.rectangle(image, (220, 280), (250, 330), (255, 255, 255), -1)  # Thumb
        
        return image
    
    def create_test_image_with_multiple_hands(self) -> np.ndarray:
        """
        Create a test image with two hand-like shapes.
        """
        # Create a blank image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # First hand (left side)
        cv2.rectangle(image, (50, 200), (150, 350), (255, 255, 255), -1)
        cv2.rectangle(image, (60, 150), (80, 200), (255, 255, 255), -1)
        cv2.rectangle(image, (90, 140), (110, 200), (255, 255, 255), -1)
        cv2.rectangle(image, (120, 150), (140, 200), (255, 255, 255), -1)
        
        # Second hand (right side)
        cv2.rectangle(image, (450, 200), (550, 350), (255, 255, 255), -1)
        cv2.rectangle(image, (460, 150), (480, 200), (255, 255, 255), -1)
        cv2.rectangle(image, (490, 140), (510, 200), (255, 255, 255), -1)
        cv2.rectangle(image, (520, 150), (540, 200), (255, 255, 255), -1)
        
        return image
    
    def create_test_image_without_hand(self) -> np.ndarray:
        """
        Create a test image with no hand present.
        """
        # Create an image with random objects but no hand
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some random shapes that are not hand-like
        cv2.circle(image, (320, 240), 50, (100, 100, 100), -1)
        cv2.rectangle(image, (100, 100), (200, 150), (150, 150, 150), -1)
        
        return image
    
    # Test 1: Hand detection with clear hand images
    def test_detect_hand_with_clear_image(self):
        """Test that HandDetector successfully detects a hand in a clear image."""
        # Create test image with hand
        image = self.create_test_image_with_hand()
        
        # Mock MediaPipe to return hand landmarks
        mock_results = MagicMock()
        mock_hand_landmarks = MagicMock()
        
        # Create mock landmarks (21 points)
        mock_landmarks = []
        for i in range(21):
            mock_landmark = MagicMock()
            mock_landmark.x = 0.5 + (i * 0.01)
            mock_landmark.y = 0.5 + (i * 0.01)
            mock_landmarks.append(mock_landmark)
        
        mock_hand_landmarks.landmark = mock_landmarks
        mock_results.multi_hand_landmarks = [mock_hand_landmarks]
        
        # Mock handedness
        mock_handedness = MagicMock()
        mock_handedness.classification = [MagicMock()]
        mock_handedness.classification[0].label = "Right"
        mock_results.multi_handedness = [mock_handedness]
        
        self.mock_hands_instance.process.return_value = mock_results
        
        # Detect hand
        result = self.detector.detect_hand(image)
        
        # Verify detection occurred
        self.assertIsNotNone(result)
        self.assertIsInstance(result, HandLandmarks)
        self.assertEqual(len(result.landmarks), 21)
        self.assertEqual(result.image_width, 640)
        self.assertEqual(result.image_height, 480)
        self.assertEqual(result.handedness, "Right")
        
        # Verify all landmarks are tuples of integers
        for landmark in result.landmarks:
            self.assertIsInstance(landmark, tuple)
            self.assertEqual(len(landmark), 2)
            self.assertIsInstance(landmark[0], int)
            self.assertIsInstance(landmark[1], int)
    
    def test_detect_hand_returns_21_landmarks(self):
        """Test that detected hand has exactly 21 landmarks."""
        # Create a more realistic hand image using a solid color hand shape
        image = self.create_test_image_with_hand()
        
        # Mock MediaPipe to return hand landmarks
        mock_results = MagicMock()
        mock_hand_landmarks = MagicMock()
        
        # Create mock landmarks (21 points)
        mock_landmarks = []
        for i in range(21):
            mock_landmark = MagicMock()
            mock_landmark.x = 0.3 + (i * 0.02)
            mock_landmark.y = 0.4 + (i * 0.015)
            mock_landmarks.append(mock_landmark)
        
        mock_hand_landmarks.landmark = mock_landmarks
        mock_results.multi_hand_landmarks = [mock_hand_landmarks]
        
        # Mock handedness
        mock_handedness = MagicMock()
        mock_handedness.classification = [MagicMock()]
        mock_handedness.classification[0].label = "Left"
        mock_results.multi_handedness = [mock_handedness]
        
        self.mock_hands_instance.process.return_value = mock_results
        
        result = self.detector.detect_hand(image)
        
        # Verify landmark count
        self.assertIsNotNone(result)
        self.assertEqual(len(result.landmarks), 21,
                       "Hand should have exactly 21 landmarks")
    
    def test_detect_hand_with_invalid_input(self):
        """Test that HandDetector handles invalid input gracefully."""
        # Test with None
        result = self.detector.detect_hand(None)
        self.assertIsNone(result)
        
        # Test with invalid type
        result = self.detector.detect_hand("not an image")
        self.assertIsNone(result)
    
    # Test 2: Behavior when no hand is present
    def test_no_hand_detected(self):
        """Test that HandDetector returns None when no hand is present."""
        # Create image without hand
        image = self.create_test_image_without_hand()
        
        # Mock MediaPipe to return no hand landmarks
        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = None
        self.mock_hands_instance.process.return_value = mock_results
        
        # Attempt detection
        result = self.detector.detect_hand(image)
        
        # Should return None
        self.assertIsNone(result, "Should return None when no hand is detected")
    
    def test_no_hand_in_blank_image(self):
        """Test detection on a completely blank image."""
        # Create blank image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock MediaPipe to return no hand landmarks
        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = None
        self.mock_hands_instance.process.return_value = mock_results
        
        result = self.detector.detect_hand(image)
        
        self.assertIsNone(result, "Should return None for blank image")
    
    def test_no_hand_in_noise_image(self):
        """Test detection on an image with random noise."""
        # Create noisy image
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # Mock MediaPipe to return no hand landmarks
        mock_results = MagicMock()
        mock_results.multi_hand_landmarks = None
        self.mock_hands_instance.process.return_value = mock_results
        
        result = self.detector.detect_hand(image)
        
        # Should return None (noise shouldn't be detected as hand)
        self.assertIsNone(result, "Should return None for noisy image")
    
    # Test 3: Handling of multiple hands in image
    def test_multiple_hands_returns_one_hand(self):
        """Test that HandDetector returns only one hand when multiple are present."""
        # Create image with multiple hands
        image = self.create_test_image_with_multiple_hands()
        
        # Mock MediaPipe to return multiple hand landmarks
        mock_results = MagicMock()
        
        # Create first hand landmarks
        mock_hand_landmarks_1 = MagicMock()
        mock_landmarks_1 = []
        for i in range(21):
            mock_landmark = MagicMock()
            mock_landmark.x = 0.2 + (i * 0.01)
            mock_landmark.y = 0.3 + (i * 0.01)
            mock_landmarks_1.append(mock_landmark)
        mock_hand_landmarks_1.landmark = mock_landmarks_1
        
        # Create second hand landmarks
        mock_hand_landmarks_2 = MagicMock()
        mock_landmarks_2 = []
        for i in range(21):
            mock_landmark = MagicMock()
            mock_landmark.x = 0.7 + (i * 0.01)
            mock_landmark.y = 0.3 + (i * 0.01)
            mock_landmarks_2.append(mock_landmark)
        mock_hand_landmarks_2.landmark = mock_landmarks_2
        
        # Return both hands
        mock_results.multi_hand_landmarks = [mock_hand_landmarks_1, mock_hand_landmarks_2]
        
        # Mock handedness for both hands
        mock_handedness_1 = MagicMock()
        mock_handedness_1.classification = [MagicMock()]
        mock_handedness_1.classification[0].label = "Left"
        
        mock_handedness_2 = MagicMock()
        mock_handedness_2.classification = [MagicMock()]
        mock_handedness_2.classification[0].label = "Right"
        
        mock_results.multi_handedness = [mock_handedness_1, mock_handedness_2]
        
        self.mock_hands_instance.process.return_value = mock_results
        
        result = self.detector.detect_hand(image)
        
        # Should return a single HandLandmarks object (first/most prominent hand)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, HandLandmarks)
        self.assertEqual(len(result.landmarks), 21)
        # Should return the first detected hand
        self.assertEqual(result.handedness, "Left")
    
    def test_detector_initialization(self):
        """Test that HandDetector initializes correctly with custom confidence."""
        detector = HandDetector(min_detection_confidence=0.5)
        self.assertEqual(detector.min_detection_confidence, 0.5)
        del detector
        
        detector = HandDetector(min_detection_confidence=0.9)
        self.assertEqual(detector.min_detection_confidence, 0.9)
        del detector
    
    def test_landmarks_within_image_bounds(self):
        """Test that detected landmarks are within image boundaries."""
        image = self.create_test_image_with_hand()
        
        # Mock MediaPipe to return hand landmarks within bounds
        mock_results = MagicMock()
        mock_hand_landmarks = MagicMock()
        
        # Create mock landmarks with normalized coordinates
        mock_landmarks = []
        for i in range(21):
            mock_landmark = MagicMock()
            # Ensure coordinates are within [0, 1] range
            mock_landmark.x = min(0.9, 0.2 + (i * 0.03))
            mock_landmark.y = min(0.9, 0.3 + (i * 0.025))
            mock_landmarks.append(mock_landmark)
        
        mock_hand_landmarks.landmark = mock_landmarks
        mock_results.multi_hand_landmarks = [mock_hand_landmarks]
        
        # Mock handedness
        mock_handedness = MagicMock()
        mock_handedness.classification = [MagicMock()]
        mock_handedness.classification[0].label = "Right"
        mock_results.multi_handedness = [mock_handedness]
        
        self.mock_hands_instance.process.return_value = mock_results
        
        result = self.detector.detect_hand(image)
        
        self.assertIsNotNone(result)
        image_height, image_width = image.shape[:2]
        
        for x, y in result.landmarks:
            self.assertGreaterEqual(x, 0, "X coordinate should be >= 0")
            self.assertGreaterEqual(y, 0, "Y coordinate should be >= 0")
            self.assertLessEqual(x, image_width, 
                               f"X coordinate should be <= {image_width}")
            self.assertLessEqual(y, image_height,
                               f"Y coordinate should be <= {image_height}")


if __name__ == '__main__':
    unittest.main()
