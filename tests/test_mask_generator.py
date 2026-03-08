"""
Unit tests for MaskGenerator component.
"""
import unittest
import numpy as np
import cv2
from src.mask_generator import MaskGenerator


class TestMaskGenerator(unittest.TestCase):
    """Test cases for MaskGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = MaskGenerator(blur_kernel_size=5, blur_sigma=1.0)
        self.image_shape = (480, 640)  # height, width
    
    def tearDown(self):
        """Clean up after tests."""
        del self.generator
    
    # Helper methods for creating test contours
    def create_rectangular_contour(self) -> np.ndarray:
        """Create a simple rectangular contour."""
        return np.array([
            [200, 150],
            [400, 150],
            [400, 350],
            [200, 350]
        ], dtype=np.float32)
    
    def create_triangular_contour(self) -> np.ndarray:
        """Create a triangular contour."""
        return np.array([
            [320, 100],
            [500, 400],
            [140, 400]
        ], dtype=np.float32)
    
    def create_pentagon_contour(self) -> np.ndarray:
        """Create a pentagon-shaped contour."""
        return np.array([
            [320, 100],
            [450, 200],
            [400, 350],
            [240, 350],
            [190, 200]
        ], dtype=np.float32)
    
    def create_complex_palm_contour(self) -> np.ndarray:
        """Create a more realistic palm-like contour with 5 points."""
        return np.array([
            [320, 350],  # Wrist
            [250, 200],  # Index base
            [300, 180],  # Middle base
            [350, 180],  # Ring base
            [390, 200]   # Pinky base
        ], dtype=np.float32)
    
    def create_small_contour(self) -> np.ndarray:
        """Create a small contour."""
        return np.array([
            [310, 230],
            [330, 230],
            [330, 250],
            [310, 250]
        ], dtype=np.float32)
    
    def create_test_image(self) -> np.ndarray:
        """Create a test RGB image with gradient pattern."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Create a gradient pattern
        for i in range(480):
            for j in range(640):
                image[i, j] = [i % 256, j % 256, (i + j) % 256]
        return image
    
    # Test 1: Mask creation with various contour shapes
    def test_create_mask_with_rectangular_contour(self):
        """Test mask creation with a rectangular contour."""
        contour = self.create_rectangular_contour()
        mask = self.generator.create_palm_mask(self.image_shape, contour)
        
        # Verify mask properties
        self.assertEqual(mask.shape, self.image_shape)
        self.assertEqual(mask.dtype, np.uint8)
        
        # Verify mask has non-zero values in the contour region
        # Check center of rectangle should be white
        center_value = mask[250, 300]
        self.assertGreater(center_value, 200, "Center of mask should be bright")
        
        # Check outside rectangle should be black
        outside_value = mask[50, 50]
        self.assertLess(outside_value, 50, "Outside mask should be dark")
    
    def test_create_mask_with_triangular_contour(self):
        """Test mask creation with a triangular contour."""
        contour = self.create_triangular_contour()
        mask = self.generator.create_palm_mask(self.image_shape, contour)
        
        self.assertEqual(mask.shape, self.image_shape)
        
        # Check a point inside the triangle
        inside_value = mask[300, 320]
        self.assertGreater(inside_value, 200, "Inside triangle should be bright")
        
        # Check a point outside the triangle
        outside_value = mask[50, 50]
        self.assertLess(outside_value, 50, "Outside triangle should be dark")
    
    def test_create_mask_with_pentagon_contour(self):
        """Test mask creation with a pentagon contour."""
        contour = self.create_pentagon_contour()
        mask = self.generator.create_palm_mask(self.image_shape, contour)
        
        self.assertEqual(mask.shape, self.image_shape)
        
        # Check center of pentagon
        center_value = mask[250, 320]
        self.assertGreater(center_value, 200, "Center of pentagon should be bright")
    
    def test_create_mask_with_palm_like_contour(self):
        """Test mask creation with a realistic palm-shaped contour."""
        contour = self.create_complex_palm_contour()
        mask = self.generator.create_palm_mask(self.image_shape, contour)
        
        self.assertEqual(mask.shape, self.image_shape)
        self.assertEqual(mask.dtype, np.uint8)
        
        # Verify mask contains both 0 and high values
        self.assertGreater(np.max(mask), 200, "Mask should have bright regions")
        self.assertEqual(np.min(mask), 0, "Mask should have dark regions")
    
    def test_create_mask_with_small_contour(self):
        """Test mask creation with a very small contour."""
        contour = self.create_small_contour()
        mask = self.generator.create_palm_mask(self.image_shape, contour)
        
        self.assertEqual(mask.shape, self.image_shape)
        
        # Small contour should still create a valid mask
        self.assertGreater(np.max(mask), 0, "Small contour should create visible mask")
    
    def test_create_mask_invalid_contour(self):
        """Test that invalid contours raise appropriate errors."""
        # Test with too few points
        with self.assertRaises(ValueError):
            self.generator.create_palm_mask(self.image_shape, np.array([[100, 100]]))
        
        # Test with None
        with self.assertRaises(ValueError):
            self.generator.create_palm_mask(self.image_shape, None)
        
        # Test with empty array
        with self.assertRaises(ValueError):
            self.generator.create_palm_mask(self.image_shape, np.array([]))
    
    def test_create_mask_invalid_image_shape(self):
        """Test that invalid image shapes raise appropriate errors."""
        contour = self.create_rectangular_contour()
        
        # Test with wrong shape tuple length
        with self.assertRaises(ValueError):
            self.generator.create_palm_mask((480, 640, 3), contour)
        
        with self.assertRaises(ValueError):
            self.generator.create_palm_mask((480,), contour)
    
    # Test 2: Mask application with different background colors
    def test_apply_mask_with_black_background(self):
        """Test applying mask with default black background."""
        image = self.create_test_image()
        contour = self.create_rectangular_contour()
        mask = self.generator.create_palm_mask(self.image_shape, contour)
        
        result = self.generator.apply_mask(image, mask)
        
        # Verify output shape and type
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        
        # Check that masked region preserves image data
        center_pixel = result[250, 300]
        original_pixel = image[250, 300]
        # Should be similar to original (allowing for some blending)
        np.testing.assert_array_less(np.abs(center_pixel - original_pixel), 50)
        
        # Check that background is dark
        outside_pixel = result[50, 50]
        self.assertLess(np.max(outside_pixel), 50, "Background should be dark")
    
    def test_apply_mask_with_white_background(self):
        """Test applying mask with white background."""
        image = self.create_test_image()
        contour = self.create_rectangular_contour()
        mask = self.generator.create_palm_mask(self.image_shape, contour)
        
        result = self.generator.apply_mask(image, mask, background_color=(255, 255, 255))
        
        self.assertEqual(result.shape, image.shape)
        
        # Check that background is white
        outside_pixel = result[50, 50]
        self.assertGreater(np.min(outside_pixel), 200, "Background should be white")
    
    def test_apply_mask_with_red_background(self):
        """Test applying mask with red background."""
        image = self.create_test_image()
        contour = self.create_rectangular_contour()
        mask = self.generator.create_palm_mask(self.image_shape, contour)
        
        result = self.generator.apply_mask(image, mask, background_color=(255, 0, 0))
        
        self.assertEqual(result.shape, image.shape)
        
        # Check that background is red
        outside_pixel = result[50, 50]
        self.assertGreater(outside_pixel[0], 200, "Background R channel should be high")
        self.assertLess(outside_pixel[1], 50, "Background G channel should be low")
        self.assertLess(outside_pixel[2], 50, "Background B channel should be low")
    
    def test_apply_mask_with_blue_background(self):
        """Test applying mask with blue background."""
        image = self.create_test_image()
        contour = self.create_rectangular_contour()
        mask = self.generator.create_palm_mask(self.image_shape, contour)
        
        result = self.generator.apply_mask(image, mask, background_color=(0, 0, 255))
        
        # Check that background is blue
        outside_pixel = result[50, 50]
        self.assertLess(outside_pixel[0], 50, "Background R channel should be low")
        self.assertLess(outside_pixel[1], 50, "Background G channel should be low")
        self.assertGreater(outside_pixel[2], 200, "Background B channel should be high")
    
    def test_apply_mask_with_custom_gray_background(self):
        """Test applying mask with custom gray background."""
        image = self.create_test_image()
        contour = self.create_rectangular_contour()
        mask = self.generator.create_palm_mask(self.image_shape, contour)
        
        result = self.generator.apply_mask(image, mask, background_color=(128, 128, 128))
        
        # Check that background is gray
        outside_pixel = result[50, 50]
        for channel in outside_pixel:
            self.assertGreater(channel, 100, "Gray background should be mid-range")
            self.assertLess(channel, 150, "Gray background should be mid-range")
    
    def test_apply_mask_rgba_mode(self):
        """Test applying mask with RGBA output mode (transparent background)."""
        image = self.create_test_image()
        contour = self.create_rectangular_contour()
        mask = self.generator.create_palm_mask(self.image_shape, contour)
        
        result = self.generator.apply_mask(image, mask, output_mode='RGBA')
        
        # Verify RGBA output
        self.assertEqual(result.shape, (480, 640, 4))
        self.assertEqual(result.dtype, np.uint8)
        
        # Check that RGB channels match original in masked region
        center_rgb = result[250, 300, :3]
        original_rgb = image[250, 300]
        np.testing.assert_array_equal(center_rgb, original_rgb)
        
        # Check that alpha channel matches mask
        center_alpha = result[250, 300, 3]
        mask_value = mask[250, 300]
        self.assertEqual(center_alpha, mask_value)
        
        # Check that outside region has low alpha
        outside_alpha = result[50, 50, 3]
        self.assertLess(outside_alpha, 50, "Outside region should have low alpha")
    
    def test_apply_mask_mismatched_dimensions(self):
        """Test that mismatched image and mask dimensions raise error."""
        image = self.create_test_image()
        wrong_shape = (240, 320)  # Different from image shape
        contour = self.create_rectangular_contour()
        mask = self.generator.create_palm_mask(wrong_shape, contour)
        
        with self.assertRaises(ValueError):
            self.generator.apply_mask(image, mask)
    
    def test_apply_mask_invalid_output_mode(self):
        """Test that invalid output mode raises error."""
        image = self.create_test_image()
        contour = self.create_rectangular_contour()
        mask = self.generator.create_palm_mask(self.image_shape, contour)
        
        with self.assertRaises(ValueError):
            self.generator.apply_mask(image, mask, output_mode='INVALID')
    
    # Test 3: Edge smoothing quality
    def test_edge_smoothing_applied(self):
        """Test that edge smoothing is applied to mask."""
        contour = self.create_rectangular_contour()
        mask = self.generator.create_palm_mask(self.image_shape, contour)
        
        # Check for gradient values at edges (not just 0 or 255)
        # Sample edge region
        edge_region = mask[148:152, 200:210]  # Near top edge of rectangle
        
        # Should have intermediate values due to Gaussian blur
        unique_values = np.unique(edge_region)
        self.assertGreater(len(unique_values), 2, 
                          "Edge should have gradient values, not just 0 and 255")
    
    def test_edge_smoothing_with_different_blur_settings(self):
        """Test edge smoothing with different blur kernel sizes."""
        contour = self.create_rectangular_contour()
        
        # Create mask with small blur
        generator_small = MaskGenerator(blur_kernel_size=3, blur_sigma=0.5)
        mask_small = generator_small.create_palm_mask(self.image_shape, contour)
        
        # Create mask with large blur
        generator_large = MaskGenerator(blur_kernel_size=9, blur_sigma=2.0)
        mask_large = generator_large.create_palm_mask(self.image_shape, contour)
        
        # Both should be valid masks
        self.assertEqual(mask_small.shape, self.image_shape)
        self.assertEqual(mask_large.shape, self.image_shape)
        
        # Larger blur should create smoother transitions
        # Check edge gradient width
        edge_row_small = mask_small[150, :]
        edge_row_large = mask_large[150, :]
        
        # Count pixels in transition zone (between 50 and 200)
        transition_small = np.sum((edge_row_small > 50) & (edge_row_small < 200))
        transition_large = np.sum((edge_row_large > 50) & (edge_row_large < 200))
        
        self.assertGreater(transition_large, transition_small,
                          "Larger blur should create wider transition zone")
    
    def test_mask_smoothness_quality(self):
        """Test that mask edges are smooth without jagged artifacts."""
        contour = self.create_complex_palm_contour()
        mask = self.generator.create_palm_mask(self.image_shape, contour)
        
        # Check that mask has gradual transitions
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Smooth edges should have moderate gradients, not extreme jumps
        max_gradient = np.max(gradient_magnitude)
        # With Gaussian blur, gradients should be reasonable (not thousands)
        self.assertLess(max_gradient, 1000, 
                       "Gradients should be moderate due to smoothing")
    
    def test_no_jagged_edges_in_applied_mask(self):
        """Test that applied mask doesn't have jagged edges."""
        image = self.create_test_image()
        contour = self.create_rectangular_contour()
        mask = self.generator.create_palm_mask(self.image_shape, contour)
        result = self.generator.apply_mask(image, mask)
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Count edge pixels - should be reasonable, not excessive
        edge_count = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_ratio = edge_count / total_pixels
        
        # Edge ratio should be small (smooth result)
        self.assertLess(edge_ratio, 0.1, 
                       "Edge ratio should be low for smooth result")
    
    def test_initialization_with_invalid_blur_kernel(self):
        """Test that even blur kernel size raises error."""
        with self.assertRaises(ValueError):
            MaskGenerator(blur_kernel_size=4)  # Even number
        
        with self.assertRaises(ValueError):
            MaskGenerator(blur_kernel_size=10)  # Even number
    
    def test_initialization_with_valid_blur_kernel(self):
        """Test initialization with valid odd blur kernel sizes."""
        generator1 = MaskGenerator(blur_kernel_size=3)
        self.assertEqual(generator1.blur_kernel_size, 3)
        
        generator2 = MaskGenerator(blur_kernel_size=7)
        self.assertEqual(generator2.blur_kernel_size, 7)
        
        generator3 = MaskGenerator(blur_kernel_size=11)
        self.assertEqual(generator3.blur_kernel_size, 11)
    
    def test_mask_preserves_contour_area(self):
        """Test that mask area roughly matches contour area."""
        contour = self.create_rectangular_contour()
        mask = self.generator.create_palm_mask(self.image_shape, contour)
        
        # Calculate expected area from contour
        expected_area = cv2.contourArea(contour.reshape(-1, 1, 2).astype(np.int32))
        
        # Calculate actual mask area (count bright pixels)
        mask_area = np.sum(mask > 127)
        
        # Should be within reasonable tolerance (accounting for blur)
        ratio = mask_area / expected_area
        self.assertGreater(ratio, 0.8, "Mask area should be close to contour area")
        self.assertLess(ratio, 1.2, "Mask area should be close to contour area")


if __name__ == '__main__':
    unittest.main()
