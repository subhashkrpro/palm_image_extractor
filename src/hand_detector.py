"""
HandDetector component for detecting hands and extracting landmarks using MediaPipe.
"""
from typing import Optional, List, Tuple
from dataclasses import dataclass
import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os


@dataclass
class HandLandmarks:
    """Data class representing detected hand landmarks."""
    landmarks: List[Tuple[int, int]]  # 21 (x, y) coordinates
    image_width: int
    image_height: int
    handedness: str  # "Left" or "Right"


class HandDetector:
    """Detects hands and extracts landmarks using MediaPipe Hands."""
    
    def __init__(self, min_detection_confidence: float = 0.7):
        """
        Initialize the HandDetector with MediaPipe HandLandmarker.
        
        Args:
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for hand
                                     detection to be considered successful
        """
        self.min_detection_confidence = min_detection_confidence
        
        # Download model file if not present
        model_path = self._get_model_path()
        
        # Create HandLandmarker options
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,  # Detect up to 2 hands to handle multiple hands case
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Create HandLandmarker
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._last_results = None
    
    def _get_model_path(self) -> str:
        """
        Get the path to the hand landmarker model file.
        Downloads it if not present.
        
        Returns:
            Path to the model file
        """
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'hand_landmarker.task')
        
        if not os.path.exists(model_path):
            print("Downloading hand landmarker model (first time only)...")
            model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print(f"Model downloaded successfully to {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download hand landmarker model: {str(e)}")
        
        return model_path
    
    def detect_hand(self, image: np.ndarray) -> Optional[HandLandmarks]:
        """
        Process image and detect hand landmarks.
        
        Args:
            image: Input image in RGB format as numpy array
            
        Returns:
            HandLandmarks object if hand detected, None otherwise
        """
        if image is None or not isinstance(image, np.ndarray):
            return None
        
        # Convert numpy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # Process the image with MediaPipe
        results = self.detector.detect(mp_image)
        self._last_results = results
        
        # Check if any hands were detected
        if not results.hand_landmarks:
            return None
        
        # Handle multiple hands - select the most prominent one (first detected)
        hand_landmarks = results.hand_landmarks[0]
        handedness = results.handedness[0][0].category_name
        
        # Extract (x, y) coordinates
        image_height, image_width = image.shape[:2]
        landmarks = self.get_landmarks(hand_landmarks, image_width, image_height)
        
        return HandLandmarks(
            landmarks=landmarks,
            image_width=image_width,
            image_height=image_height,
            handedness=handedness
        )
    
    def get_landmarks(self, hand_landmarks, image_width: int, image_height: int) -> List[Tuple[int, int]]:
        """
        Extract (x, y) coordinates from MediaPipe hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks list
            image_width: Width of the image
            image_height: Height of the image
            
        Returns:
            List of 21 (x, y) coordinate tuples
        """
        landmarks = []
        for landmark in hand_landmarks:
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            landmarks.append((x, y))
        
        return landmarks
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'detector'):
            self.detector.close()

