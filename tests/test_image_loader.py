"""
Unit tests for ImageLoader component.
"""
import os
import pytest
import numpy as np
import cv2
from src.image_loader import ImageLoader


class TestImageLoader:
    """Test suite for ImageLoader class."""
    
    @pytest.fixture
    def image_loader(self):
        """Create an ImageLoader instance for testing."""
        return ImageLoader()
    
    @pytest.fixture
    def test_images_dir(self, tmp_path):
        """Create a temporary directory for test images."""
        test_dir = tmp_path / "test_images"
        test_dir.mkdir()
        return test_dir
    
    @pytest.fixture
    def valid_jpg_image(self, test_images_dir):
        """Create a valid JPG test image."""
        image_path = test_images_dir / "test_image.jpg"
        # Create a simple test image (100x100 red square)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :] = [255, 0, 0]  # Red in RGB
        # Convert to BGR for saving with OpenCV
        test_image_bgr = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), test_image_bgr)
        return image_path
    
    @pytest.fixture
    def valid_png_image(self, test_images_dir):
        """Create a valid PNG test image."""
        image_path = test_images_dir / "test_image.png"
        # Create a simple test image (100x100 green square)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :] = [0, 255, 0]  # Green in RGB
        # Convert to BGR for saving with OpenCV
        test_image_bgr = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), test_image_bgr)
        return image_path
    
    @pytest.fixture
    def valid_jpeg_image(self, test_images_dir):
        """Create a valid JPEG test image."""
        image_path = test_images_dir / "test_image.jpeg"
        # Create a simple test image (100x100 blue square)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :] = [0, 0, 255]  # Blue in RGB
        # Convert to BGR for saving with OpenCV
        test_image_bgr = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), test_image_bgr)
        return image_path
    
    @pytest.fixture
    def corrupted_image(self, test_images_dir):
        """Create a corrupted image file."""
        image_path = test_images_dir / "corrupted.jpg"
        # Write invalid data to simulate corruption
        with open(image_path, 'wb') as f:
            f.write(b'This is not a valid image file')
        return image_path
    
    # Test loading valid images in different formats
    
    def test_load_valid_jpg_image(self, image_loader, valid_jpg_image):
        """Test loading a valid JPG image."""
        image = image_loader.load_image(str(valid_jpg_image))
        
        assert image is not None
        assert isinstance(image, np.ndarray)
        assert image.shape == (100, 100, 3)
        assert image.dtype == np.uint8
        # Check that it's in RGB format (red channel should be high)
        assert np.mean(image[:, :, 0]) > 200  # Red channel
    
    def test_load_valid_png_image(self, image_loader, valid_png_image):
        """Test loading a valid PNG image."""
        image = image_loader.load_image(str(valid_png_image))
        
        assert image is not None
        assert isinstance(image, np.ndarray)
        assert image.shape == (100, 100, 3)
        assert image.dtype == np.uint8
        # Check that it's in RGB format (green channel should be high)
        assert np.mean(image[:, :, 1]) > 200  # Green channel
    
    def test_load_valid_jpeg_image(self, image_loader, valid_jpeg_image):
        """Test loading a valid JPEG image."""
        image = image_loader.load_image(str(valid_jpeg_image))
        
        assert image is not None
        assert isinstance(image, np.ndarray)
        assert image.shape == (100, 100, 3)
        assert image.dtype == np.uint8
        # Check that it's in RGB format (blue channel should be high)
        assert np.mean(image[:, :, 2]) > 200  # Blue channel
    
    # Test error handling for invalid paths
    
    def test_load_nonexistent_file(self, image_loader):
        """Test loading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError) as exc_info:
            image_loader.load_image("nonexistent_file.jpg")
        
        assert "Image file not found" in str(exc_info.value)
    
    def test_load_invalid_path(self, image_loader):
        """Test loading with an invalid path."""
        with pytest.raises(FileNotFoundError):
            image_loader.load_image("/invalid/path/to/image.jpg")
    
    # Test error handling for unsupported formats
    
    def test_load_unsupported_format(self, image_loader, test_images_dir):
        """Test loading an unsupported image format."""
        unsupported_file = test_images_dir / "test.bmp"
        # Create a dummy file
        unsupported_file.touch()
        
        with pytest.raises(ValueError) as exc_info:
            image_loader.load_image(str(unsupported_file))
        
        assert "Unsupported image format" in str(exc_info.value)
        assert ".bmp" in str(exc_info.value)
    
    # Test error handling for corrupted images
    
    def test_load_corrupted_image(self, image_loader, corrupted_image):
        """Test loading a corrupted image file."""
        with pytest.raises(ValueError) as exc_info:
            image_loader.load_image(str(corrupted_image))
        
        assert "Failed to load image" in str(exc_info.value) or "corrupted" in str(exc_info.value).lower()
    
    # Test validate_image method
    
    def test_validate_valid_image(self, image_loader):
        """Test validation of a valid image."""
        valid_image = np.zeros((100, 100, 3), dtype=np.uint8)
        assert image_loader.validate_image(valid_image) is True
    
    def test_validate_grayscale_image(self, image_loader):
        """Test validation of a grayscale image."""
        grayscale_image = np.zeros((100, 100), dtype=np.uint8)
        assert image_loader.validate_image(grayscale_image) is True
    
    def test_validate_none_image(self, image_loader):
        """Test validation of None."""
        assert image_loader.validate_image(None) is False
    
    def test_validate_non_array(self, image_loader):
        """Test validation of non-numpy array."""
        assert image_loader.validate_image([1, 2, 3]) is False
        assert image_loader.validate_image("not an image") is False
    
    def test_validate_invalid_dimensions(self, image_loader):
        """Test validation of image with invalid dimensions."""
        # 1D array
        invalid_image_1d = np.zeros(100, dtype=np.uint8)
        assert image_loader.validate_image(invalid_image_1d) is False
        
        # 4D array
        invalid_image_4d = np.zeros((10, 10, 3, 3), dtype=np.uint8)
        assert image_loader.validate_image(invalid_image_4d) is False
    
    def test_validate_empty_image(self, image_loader):
        """Test validation of empty image."""
        empty_image = np.zeros((0, 0, 3), dtype=np.uint8)
        assert image_loader.validate_image(empty_image) is False
    
    def test_validate_invalid_dtype(self, image_loader):
        """Test validation of image with invalid data type."""
        invalid_dtype_image = np.zeros((100, 100, 3), dtype=np.int32)
        assert image_loader.validate_image(invalid_dtype_image) is False
    
    def test_validate_float_image(self, image_loader):
        """Test validation of float image (should be valid)."""
        float_image = np.zeros((100, 100, 3), dtype=np.float32)
        assert image_loader.validate_image(float_image) is True
