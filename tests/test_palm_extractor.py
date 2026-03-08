"""
Unit tests for PalmExtractor component.
"""
import pytest
import numpy as np
from src.palm_extractor import PalmExtractor, PalmRegion


class TestPalmExtractor:
    """Test suite for PalmExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = PalmExtractor()
        
        # Create sample hand landmarks (21 points)
        # Simulating a hand in neutral position with realistic palm shape
        self.sample_landmarks = [
            (150, 350),  # 0: Wrist
            (140, 330),  # 1: Thumb CMC
            (130, 310),  # 2: Thumb MCP
            (120, 290),  # 3: Thumb IP
            (110, 270),  # 4: Thumb tip
            (180, 320),  # 5: Index finger MCP (base)
            (190, 270),  # 6: Index finger PIP
            (200, 220),  # 7: Index finger DIP
            (210, 170),  # 8: Index finger tip
            (210, 320),  # 9: Middle finger MCP (base)
            (220, 260),  # 10: Middle finger PIP
            (230, 200),  # 11: Middle finger DIP
            (240, 140),  # 12: Middle finger tip
            (240, 325),  # 13: Ring finger MCP (base)
            (250, 270),  # 14: Ring finger PIP
            (260, 220),  # 15: Ring finger DIP
            (270, 170),  # 16: Ring finger tip
            (270, 335),  # 17: Pinky MCP (base)
            (280, 290),  # 18: Pinky PIP
            (290, 250),  # 19: Pinky DIP
            (300, 210),  # 20: Pinky tip
        ]
        
        # Sample image shape
        self.image_shape = (480, 640)  # height, width
    
    def test_get_palm_landmarks_valid_input(self):
        """Test palm landmark extraction from full hand landmarks."""
        palm_landmarks = self.extractor.get_palm_landmarks(self.sample_landmarks)
        
        # Should return exactly 5 landmarks
        assert len(palm_landmarks) == 5
        
        # Verify correct landmarks are extracted (indices 0, 5, 9, 13, 17)
        expected_landmarks = [
            (150, 350),  # Wrist
            (180, 320),  # Index base
            (210, 320),  # Middle base
            (240, 325),  # Ring base
            (270, 335),  # Pinky base
        ]
        assert palm_landmarks == expected_landmarks
    
    def test_get_palm_landmarks_invalid_input_empty(self):
        """Test palm landmark extraction with empty landmarks."""
        with pytest.raises(ValueError, match="Invalid landmarks: expected 21 landmarks"):
            self.extractor.get_palm_landmarks([])
    
    def test_get_palm_landmarks_invalid_input_insufficient(self):
        """Test palm landmark extraction with insufficient landmarks."""
        insufficient_landmarks = [(x, y) for x, y in self.sample_landmarks[:10]]
        
        with pytest.raises(ValueError, match="Invalid landmarks: expected 21 landmarks"):
            self.extractor.get_palm_landmarks(insufficient_landmarks)
    
    def test_calculate_palm_center_neutral_pose(self):
        """Test palm center calculation with neutral hand pose."""
        palm_landmarks = self.extractor.get_palm_landmarks(self.sample_landmarks)
        center = self.extractor.calculate_palm_center(palm_landmarks)
        
        # Center should be tuple of two integers
        assert isinstance(center, tuple)
        assert len(center) == 2
        assert isinstance(center[0], (int, np.integer))
        assert isinstance(center[1], (int, np.integer))
        
        # For our sample data:
        # x values are 150, 180, 210, 240, 270 -> mean = 210
        # y values are 350, 320, 320, 325, 335 -> mean = 330
        assert center == (210, 330)
    
    def test_calculate_palm_center_varied_pose(self):
        """Test palm center calculation with varied hand pose."""
        # Create landmarks with different positions
        varied_landmarks = [
            (50, 250),   # Wrist
            (100, 200),  # Index base
            (150, 180),  # Middle base
            (200, 190),  # Ring base
            (250, 210),  # Pinky base
        ]
        
        center = self.extractor.calculate_palm_center(varied_landmarks)
        
        # Mean x: (50+100+150+200+250)/5 = 150
        # Mean y: (250+200+180+190+210)/5 = 206
        assert center == (150, 206)
    
    def test_calculate_palm_center_empty_landmarks(self):
        """Test palm center calculation with empty landmarks."""
        with pytest.raises(ValueError, match="Palm landmarks cannot be empty"):
            self.extractor.calculate_palm_center([])
    
    def test_extract_palm_region_returns_palm_region_object(self):
        """Test that extract_palm_region returns a PalmRegion object."""
        palm_region = self.extractor.extract_palm_region(
            self.sample_landmarks, 
            self.image_shape
        )
        
        # Should return PalmRegion instance
        assert isinstance(palm_region, PalmRegion)
        
        # Verify all attributes are present
        assert hasattr(palm_region, 'contour')
        assert hasattr(palm_region, 'center')
        assert hasattr(palm_region, 'area')
    
    def test_extract_palm_region_contour_properties(self):
        """Test contour generation with various hand poses."""
        palm_region = self.extractor.extract_palm_region(
            self.sample_landmarks, 
            self.image_shape
        )
        
        # Contour should be a numpy array
        assert isinstance(palm_region.contour, np.ndarray)
        
        # Contour should have at least 3 points (minimum for a polygon)
        assert len(palm_region.contour) >= 3
        
        # Contour points should be 2D coordinates
        assert palm_region.contour.shape[1] == 1
        assert palm_region.contour.shape[2] == 2
    
    def test_extract_palm_region_center_calculation(self):
        """Test that palm center is correctly calculated in extract_palm_region."""
        palm_region = self.extractor.extract_palm_region(
            self.sample_landmarks, 
            self.image_shape
        )
        
        # Center should match expected value
        assert palm_region.center == (210, 330)
    
    def test_extract_palm_region_area_calculation(self):
        """Test that palm area is calculated and is positive."""
        palm_region = self.extractor.extract_palm_region(
            self.sample_landmarks, 
            self.image_shape
        )
        
        # Area should be a positive number
        assert isinstance(palm_region.area, (float, np.floating))
        assert palm_region.area > 0
    
    def test_extract_palm_region_with_rotated_hand(self):
        """Test contour generation with rotated hand pose."""
        # Create landmarks for a rotated hand (diagonal orientation)
        rotated_landmarks = [
            (200, 200),  # 0: Wrist
            (210, 190),  # 1: Thumb CMC
            (220, 180),  # 2: Thumb MCP
            (230, 170),  # 3: Thumb IP
            (240, 160),  # 4: Thumb tip
            (250, 200),  # 5: Index finger MCP
            (270, 180),  # 6: Index finger PIP
            (290, 160),  # 7: Index finger DIP
            (310, 140),  # 8: Index finger tip
            (280, 210),  # 9: Middle finger MCP
            (300, 190),  # 10: Middle finger PIP
            (320, 170),  # 11: Middle finger DIP
            (340, 150),  # 12: Middle finger tip
            (300, 230),  # 13: Ring finger MCP
            (320, 210),  # 14: Ring finger PIP
            (340, 190),  # 15: Ring finger DIP
            (360, 170),  # 16: Ring finger tip
            (320, 250),  # 17: Pinky MCP
            (340, 230),  # 18: Pinky PIP
            (360, 210),  # 19: Pinky DIP
            (380, 190),  # 20: Pinky tip
        ]
        
        palm_region = self.extractor.extract_palm_region(
            rotated_landmarks, 
            self.image_shape
        )
        
        # Should still produce valid palm region
        assert isinstance(palm_region, PalmRegion)
        assert palm_region.area > 0
        assert len(palm_region.contour) >= 3
    
    def test_extract_palm_region_with_closed_fist(self):
        """Test contour generation with closed fist pose."""
        # Create landmarks for a closed fist (fingers closer to palm)
        fist_landmarks = [
            (150, 280),  # 0: Wrist
            (160, 270),  # 1: Thumb CMC
            (170, 260),  # 2: Thumb MCP
            (175, 255),  # 3: Thumb IP
            (180, 250),  # 4: Thumb tip
            (175, 260),  # 5: Index finger MCP
            (180, 250),  # 6: Index finger PIP
            (185, 245),  # 7: Index finger DIP
            (190, 240),  # 8: Index finger tip
            (195, 255),  # 9: Middle finger MCP
            (200, 245),  # 10: Middle finger PIP
            (205, 240),  # 11: Middle finger DIP
            (210, 235),  # 12: Middle finger tip
            (215, 258),  # 13: Ring finger MCP
            (220, 250),  # 14: Ring finger PIP
            (225, 245),  # 15: Ring finger DIP
            (230, 240),  # 16: Ring finger tip
            (235, 265),  # 17: Pinky MCP
            (240, 258),  # 18: Pinky PIP
            (245, 252),  # 19: Pinky DIP
            (250, 248),  # 20: Pinky tip
        ]
        
        palm_region = self.extractor.extract_palm_region(
            fist_landmarks, 
            self.image_shape
        )
        
        # Should produce valid palm region even with closed fist
        assert isinstance(palm_region, PalmRegion)
        assert palm_region.area > 0
        
        # Palm center should be within reasonable bounds
        assert 0 <= palm_region.center[0] <= self.image_shape[1]
        assert 0 <= palm_region.center[1] <= self.image_shape[0]
    
    def test_palm_landmark_indices_constant(self):
        """Test that palm landmark indices are correctly defined."""
        expected_indices = [0, 5, 9, 13, 17]
        assert self.extractor.PALM_LANDMARK_INDICES == expected_indices
