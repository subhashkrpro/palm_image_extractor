# """
# PalmExtractor component for identifying palm region from hand landmarks.
# """
# from typing import List, Tuple
# import numpy as np
# import cv2

# from src.data_models import PalmRegion


# class PalmExtractor:
#     """Extracts central palm region from hand landmarks."""
    
#     # Key landmark indices for central palm calculation
#     WRIST = 0
#     THUMB_CMC = 1
#     THUMB_BASE = 2
#     INDEX_BASE = 5
#     MIDDLE_BASE = 9
#     RING_BASE = 13
#     PINKY_BASE = 17
    
#     def __init__(self, palm_scale: float = 0.8):
#         """
#         Initialize the PalmExtractor.
        
#         Args:
#             palm_scale: Scale factor for central palm region (0.0-1.0).
#                        Lower values = smaller central palm region.
#                        Default 0.6 extracts the central palm area.
#         """
#         self.palm_scale = palm_scale
    
#     def get_palm_landmarks(self, landmarks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
#         """
#         Extract palm-specific landmarks (wrist and finger base landmarks).
        
#         Args:
#             landmarks: List of all 21 hand landmarks as (x, y) tuples
            
#         Returns:
#             List of palm landmarks (wrist at index 0, and finger bases at 5, 9, 13, 17)
#         """
#         if not landmarks or len(landmarks) < 21:
#             raise ValueError("Invalid landmarks: expected 21 landmarks")
        
#         palm_indices = [self.WRIST, self.THUMB_CMC, self.THUMB_BASE, self.INDEX_BASE, self.MIDDLE_BASE, 
#                        self.RING_BASE, self.PINKY_BASE]
#         palm_landmarks = [landmarks[i] for i in palm_indices]
#         return palm_landmarks
    
#     def calculate_palm_center(self, palm_landmarks: List[Tuple[int, int]]) -> Tuple[int, int]:
#         """
#         Compute the centroid of the palm region.
        
#         Args:
#             palm_landmarks: List of palm landmark coordinates
            
#         Returns:
#             (x, y) tuple representing the center point of the palm
#         """
#         if not palm_landmarks:
#             raise ValueError("Palm landmarks cannot be empty")
        
#         # Calculate centroid as the mean of all palm landmark coordinates
#         x_coords = [point[0] for point in palm_landmarks]
#         y_coords = [point[1] for point in palm_landmarks]
        
#         center_x = int(np.mean(x_coords))
#         center_y = int(np.mean(y_coords))
        
#         return (center_x, center_y)
    
#     def create_central_palm_contour(self, landmarks: List[Tuple[int, int]]) -> np.ndarray:
#         """
#         Create a contour for the central palm region by scaling inward from the outer palm boundary.
#         This creates a smaller region focused on the center of the palm.
        
#         Args:
#             landmarks: List of all 21 hand landmarks
            
#         Returns:
#             Numpy array of contour points for the central palm region
#         """
#         # Get the outer palm boundary landmarks
#         wrist = np.array(landmarks[self.WRIST], dtype=np.float32)
#         thumb_cmc = np.array(landmarks[self.THUMB_CMC], dtype=np.float32)
#         thumb_base = np.array(landmarks[self.THUMB_BASE], dtype=np.float32)
#         index_base = np.array(landmarks[self.INDEX_BASE], dtype=np.float32)
#         middle_base = np.array(landmarks[self.MIDDLE_BASE], dtype=np.float32)
#         ring_base = np.array(landmarks[self.RING_BASE], dtype=np.float32)
#         pinky_base = np.array(landmarks[self.PINKY_BASE], dtype=np.float32)
        
#         # Calculate the palm center
#         palm_center = np.array([
#             np.mean([wrist[0], thumb_cmc[0], index_base[0], thumb_base[0], middle_base[0], ring_base[0], pinky_base[0]]),
#             np.mean([wrist[1], thumb_cmc[0], index_base[1], thumb_base[1], middle_base[1], ring_base[1], pinky_base[1]])
#         ], dtype=np.float32)
        
#         # Create central palm region by scaling points toward the center
#         # This creates a smaller region that represents the central palm area
#         central_wrist = palm_center + self.palm_scale * (wrist - palm_center)
#         central_thumb_cmc = palm_center + self.palm_scale * (thumb_cmc - palm_center)
#         central_thumb_base = palm_center + self.palm_scale * (thumb_base - palm_center)
#         central_index = palm_center + self.palm_scale * (index_base - palm_center)
#         central_middle = palm_center + self.palm_scale * (middle_base - palm_center)
#         central_ring = palm_center + self.palm_scale * (ring_base - palm_center)
#         central_pinky = palm_center + self.palm_scale * (pinky_base - palm_center)


        
#         # Create contour points for the central palm region
#         central_palm_points = np.array([
#             central_wrist,
#             central_thumb_cmc,
#             central_thumb_base,
#             central_index,
#             central_middle,
#             central_ring,
#             central_pinky
#         ], dtype=np.int32)
        
#         # Apply convex hull for smooth boundaries
#         hull = cv2.convexHull(central_palm_points)
        
#         return hull

#     def extract_palm_region(self, landmarks: List[Tuple[int, int]], 
#                            image_shape: Tuple[int, int]) -> PalmRegion:
#         """
#         Extract the central palm region from hand landmarks.
#         Creates a smaller region focused on the center of the palm.
        
#         Args:
#             landmarks: List of all 21 hand landmarks as (x, y) tuples
#             image_shape: Tuple of (height, width) of the image
            
#         Returns:
#             PalmRegion object containing contour, center, and area
#         """
#         # Create central palm contour
#         central_palm_contour = self.create_central_palm_contour(landmarks)
        
#         # Calculate palm center from the original palm landmarks
#         palm_landmarks = self.get_palm_landmarks(landmarks)
#         center = self.calculate_palm_center(palm_landmarks)
        
#         # Calculate area of the central palm region
#         area = cv2.contourArea(central_palm_contour)
        
#         return PalmRegion(
#             contour=central_palm_contour,
#             center=center,
#             area=area
#         )
















# """
# PalmExtractor component for identifying palm region from hand landmarks.
# """

# from typing import List, Tuple
# import numpy as np
# import cv2

# from src.data_models import PalmRegion


# class PalmExtractor:
#     """Extracts central palm region from hand landmarks."""
#     WRIST = 0
#     THUMB_CMC = 1
#     THUMB_BASE = 2
#     INDEX_BASE = 5
#     MIDDLE_BASE = 9
#     RING_BASE = 13
#     PINKY_BASE = 17
    

#     def __init__(self, palm_scale: float = 0.8):
#         self.palm_scale = palm_scale

#     def get_palm_landmarks(self, landmarks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:

#         if not landmarks or len(landmarks) < 21:
#             raise ValueError("Invalid landmarks: expected 21 landmarks")

#         palm_indices = [
#             self.WRIST,
#             self.THUMB_CMC,
#             self.THUMB_BASE,
#             self.INDEX_BASE,
#             self.MIDDLE_BASE,
#             self.RING_BASE,
#             self.PINKY_BASE
#         ]

#         return [landmarks[i] for i in palm_indices]

#     def calculate_palm_center(self, palm_landmarks: List[Tuple[int, int]]) -> Tuple[int, int]:

#         if not palm_landmarks:
#             raise ValueError("Palm landmarks cannot be empty")

#         x_coords = [p[0] for p in palm_landmarks]
#         y_coords = [p[1] for p in palm_landmarks]

#         center_x = int(np.mean(x_coords))
#         center_y = int(np.mean(y_coords))

#         return (center_x, center_y)

#     def create_central_palm_contour(self, landmarks: List[Tuple[int, int]]) -> np.ndarray:

#         wrist = np.array(landmarks[self.WRIST], dtype=np.float32)
#         thumb_cmc = np.array(landmarks[self.THUMB_CMC], dtype=np.float32)
#         thumb_base = np.array(landmarks[self.THUMB_BASE], dtype=np.float32)
#         index_base = np.array(landmarks[self.INDEX_BASE], dtype=np.float32)
#         middle_base = np.array(landmarks[self.MIDDLE_BASE], dtype=np.float32)
#         ring_base = np.array(landmarks[self.RING_BASE], dtype=np.float32)
#         pinky_base = np.array(landmarks[self.PINKY_BASE], dtype=np.float32)


#         palm_points = np.array([
#             thumb_cmc,
#             thumb_base,
#             index_base,
#             middle_base,
#             ring_base,
#             pinky_base,
#             wrist
#         ], dtype=np.int32)

#         hull = cv2.convexHull(palm_points)

#         return hull

#     def create_palm_mask(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:

#         mask = np.zeros(image_shape[:2], dtype=np.uint8)

#         cv2.fillConvexPoly(mask, contour, 255)

#         return mask

#     def apply_dark_palm_effect(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:

#         result = image.copy()

#         result[mask == 255] = (result[mask == 255] * 0.5).astype(np.uint8)

#         return result

#     def extract_palm_region(
#         self,
#         landmarks: List[Tuple[int, int]],
#         image_shape: Tuple[int, int]
#     ) -> PalmRegion:

#         central_palm_contour = self.create_central_palm_contour(landmarks)

#         palm_landmarks = self.get_palm_landmarks(landmarks)

#         center = self.calculate_palm_center(palm_landmarks)

#         area = cv2.contourArea(central_palm_contour)

#         return PalmRegion(
#             contour=central_palm_contour,
#             center=center,
#             area=area
#         )





























# """
# PalmExtractor component for identifying palm region from hand landmarks.
# """
# from typing import List, Tuple
# import numpy as np
# import cv2

# from src.data_models import PalmRegion


# class PalmExtractor:
#     """Extracts central palm region from hand landmarks."""
    
#     # Key landmark indices for central palm calculation
#     WRIST = 0
#     THUMB_CMC = 1
#     THUMB_BASE = 2
#     INDEX_BASE = 5
#     MIDDLE_BASE = 9
#     RING_BASE = 13
#     PINKY_BASE = 17
    
#     def __init__(self, palm_scale: float = 1.0, shape: str = "circle"):
#         """
#         Initialize the PalmExtractor.
        
#         Args:
#             palm_scale: Scale factor for central palm region (0.0-1.0).
#                        Lower values = smaller central palm region.
#                        Default 0.6 extracts the central palm area.
#             shape: Shape to extract ("circle", "rectangle", "polygon").
#                    Defaults to "circle", can be passed via config parameters.
#         """
#         self.palm_scale = palm_scale
#         self.shape = shape.lower()
    
#     def get_palm_landmarks(self, landmarks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
#         """
#         Extract palm-specific landmarks (wrist and finger base landmarks).
        
#         Args:
#             landmarks: List of all 21 hand landmarks as (x, y) tuples
            
#         Returns:
#             List of palm landmarks (wrist at index 0, and finger bases at 5, 9, 13, 17)
#         """
#         if not landmarks or len(landmarks) < 21:
#             raise ValueError("Invalid landmarks: expected 21 landmarks")
        
#         palm_indices = [self.WRIST, self.THUMB_CMC, self.THUMB_BASE, self.INDEX_BASE, self.MIDDLE_BASE, 
#                        self.RING_BASE, self.PINKY_BASE]
#         palm_landmarks = [landmarks[i] for i in palm_indices]
#         return palm_landmarks
    
#     def calculate_palm_center(self, palm_landmarks: List[Tuple[int, int]]) -> Tuple[int, int]:
#         """
#         Compute the centroid of the palm region.
        
#         Args:
#             palm_landmarks: List of palm landmark coordinates
            
#         Returns:
#             (x, y) tuple representing the center point of the palm
#         """
#         if not palm_landmarks:
#             raise ValueError("Palm landmarks cannot be empty")
        
#         # Calculate centroid as the mean of all palm landmark coordinates
#         x_coords = [point[0] for point in palm_landmarks]
#         y_coords = [point[1] for point in palm_landmarks]
        
#         center_x = int(np.mean(x_coords))
#         center_y = int(np.mean(y_coords))
        
#         return (center_x, center_y)
    
#     def create_central_palm_contour(self, landmarks: List[Tuple[int, int]]) -> np.ndarray:
#         """
#         Create a contour for the central palm region by scaling inward from the outer palm boundary.
#         This creates a smaller region focused on the center of the palm.
        
#         Args:
#             landmarks: List of all 21 hand landmarks
            
#         Returns:
#             Numpy array of contour points for the central palm region
#         """
#         # Get the outer palm boundary landmarks
#         wrist = np.array(landmarks[self.WRIST], dtype=np.float32)
#         thumb_cmc = np.array(landmarks[self.THUMB_CMC], dtype=np.float32)
#         thumb_base = np.array(landmarks[self.THUMB_BASE], dtype=np.float32)
#         index_base = np.array(landmarks[self.INDEX_BASE], dtype=np.float32)
#         middle_base = np.array(landmarks[self.MIDDLE_BASE], dtype=np.float32)
#         ring_base = np.array(landmarks[self.RING_BASE], dtype=np.float32)
#         pinky_base = np.array(landmarks[self.PINKY_BASE], dtype=np.float32)
        
#         # Calculate the palm center
#         palm_center = np.array([
#             np.mean([wrist[0], thumb_cmc[0], index_base[0], thumb_base[0], middle_base[0], ring_base[0], pinky_base[0]]),
#             np.mean([wrist[1], thumb_cmc[1], index_base[1], thumb_base[1], middle_base[1], ring_base[1], pinky_base[1]])
#         ], dtype=np.float32)
        
#         # Create central palm region by scaling points toward the center
#         # This creates a smaller region that represents the central palm area
#         central_wrist = palm_center + self.palm_scale * (wrist - palm_center)
#         central_thumb_cmc = palm_center + self.palm_scale * (thumb_cmc - palm_center)
#         central_thumb_base = palm_center + self.palm_scale * (thumb_base - palm_center)
#         central_index = palm_center + self.palm_scale * (index_base - palm_center)
#         central_middle = palm_center + self.palm_scale * (middle_base - palm_center)
#         central_ring = palm_center + self.palm_scale * (ring_base - palm_center)
#         central_pinky = palm_center + self.palm_scale * (pinky_base - palm_center)


        
#         # Create contour points for the central palm region
#         central_palm_points = np.array([
#             central_wrist,
#             central_thumb_cmc,
#             central_thumb_base,
#             central_index,
#             central_middle,
#             central_ring,
#             central_pinky
#         ], dtype=np.int32)
        
#         if self.shape == "circle":
#             # Calculate a radius based on average distance to central palm points
#             distances = np.linalg.norm(central_palm_points - palm_center, axis=1)
#             radius = int(np.mean(distances))
            
#             # Generate circle contour using OpenCV
#             center_int = (int(palm_center[0]), int(palm_center[1]))
#             pts = cv2.ellipse2Poly(center_int, (radius, radius), 0, 0, 360, 5)
#             return pts.reshape((-1, 1, 2))
            
#         elif self.shape == "rectangle":
#             # Generate a bounding rectangle matching the scaled area
#             x, y, w, h = cv2.boundingRect(central_palm_points)
#             rect_contour = np.array([
#                 [[x, y]], 
#                 [[x + w, y]], 
#                 [[x + w, y + h]], 
#                 [[x, y + h]]
#             ], dtype=np.int32)
#             return rect_contour
            
#         else:
#             # Default to polygon (convex hull for smooth boundaries)
#             hull = cv2.convexHull(central_palm_points)
#             return hull

#     def extract_palm_region(self, landmarks: List[Tuple[int, int]], 
#                            image_shape: Tuple[int, int]) -> PalmRegion:
#         """
#         Extract the central palm region from hand landmarks.
#         Creates a smaller region focused on the center of the palm.
        
#         Args:
#             landmarks: List of all 21 hand landmarks as (x, y) tuples
#             image_shape: Tuple of (height, width) of the image
            
#         Returns:
#             PalmRegion object containing contour, center, and area
#         """
#         # Create central palm contour
#         central_palm_contour = self.create_central_palm_contour(landmarks)
        
#         # Calculate palm center from the original palm landmarks
#         palm_landmarks = self.get_palm_landmarks(landmarks)
#         center = self.calculate_palm_center(palm_landmarks)
        
#         # Calculate area of the central palm region
#         area = cv2.contourArea(central_palm_contour)
        
#         return PalmRegion(
#             contour=central_palm_contour,
#             center=center,
#             area=area
#         )























# """
# PalmExtractor component for identifying palm region from hand landmarks.
# """
# from typing import List, Tuple
# import numpy as np
# import cv2

# from src.data_models import PalmRegion


# class PalmExtractor:
#     """Extracts central palm region from hand landmarks."""
    
#     # Key landmark indices for central palm calculation
#     WRIST = 0
#     THUMB_CMC = 1
#     THUMB_BASE = 2
#     INDEX_BASE = 5
#     MIDDLE_BASE = 9
#     RING_BASE = 13
#     PINKY_BASE = 17
    
#     def __init__(self, palm_scale: float = 0.8, shape: str = "circle", center_offset: Tuple[int, int] = (25, -35), size_multiplier: float = 1.3):
#         """
#         Initialize the PalmExtractor.
        
#         Args:
#             palm_scale: Scale factor for central palm region (0.0-1.0).
#                        Lower values = smaller central palm region.
#                        Default 0.6 extracts the central palm area.
#             shape: Shape to extract ("circle", "rectangle", "polygon").
#                    Defaults to "circle", can be passed via config parameters.
#             center_offset: (x, y) pixel offset to manually shift the palm center.
#                            Useful for fine-tuning the exact center position.
#             size_multiplier: Direct multiplier for the final shape size (radius/width/height).
#                              Defaults to 1.0.
#         """
#         self.palm_scale = palm_scale
#         self.shape = shape.lower()
#         self.center_offset = center_offset
#         self.size_multiplier = size_multiplier
    
#     def get_palm_landmarks(self, landmarks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
#         """
#         Extract palm-specific landmarks (wrist and finger base landmarks).
        
#         Args:
#             landmarks: List of all 21 hand landmarks as (x, y) tuples
            
#         Returns:
#             List of palm landmarks (wrist at index 0, and finger bases at 5, 9, 13, 17)
#         """
#         if not landmarks or len(landmarks) < 21:
#             raise ValueError("Invalid landmarks: expected 21 landmarks")
        
#         palm_indices = [self.WRIST, self.THUMB_CMC, self.THUMB_BASE, self.INDEX_BASE, self.MIDDLE_BASE, 
#                        self.RING_BASE, self.PINKY_BASE]
#         palm_landmarks = [landmarks[i] for i in palm_indices]
#         return palm_landmarks
    
#     def calculate_palm_center(self, palm_landmarks: List[Tuple[int, int]]) -> Tuple[int, int]:
#         """
#         Compute the centroid of the palm region.
        
#         Args:
#             palm_landmarks: List of palm landmark coordinates
            
#         Returns:
#             (x, y) tuple representing the center point of the palm
#         """
#         if not palm_landmarks:
#             raise ValueError("Palm landmarks cannot be empty")
        
#         # Calculate centroid as the mean of all palm landmark coordinates
#         x_coords = [point[0] for point in palm_landmarks]
#         y_coords = [point[1] for point in palm_landmarks]
        
#         center_x = int(np.mean(x_coords)) + self.center_offset[0]
#         center_y = int(np.mean(y_coords)) + self.center_offset[1]
        
#         return (center_x, center_y)
    
#     def create_central_palm_contour(self, landmarks: List[Tuple[int, int]]) -> np.ndarray:
#         """
#         Create a contour for the central palm region by scaling inward from the outer palm boundary.
#         This creates a smaller region focused on the center of the palm.
        
#         Args:
#             landmarks: List of all 21 hand landmarks
            
#         Returns:
#             Numpy array of contour points for the central palm region
#         """
#         # Get the outer palm boundary landmarks
#         wrist = np.array(landmarks[self.WRIST], dtype=np.float32)
#         thumb_cmc = np.array(landmarks[self.THUMB_CMC], dtype=np.float32)
#         thumb_base = np.array(landmarks[self.THUMB_BASE], dtype=np.float32)
#         index_base = np.array(landmarks[self.INDEX_BASE], dtype=np.float32)
#         middle_base = np.array(landmarks[self.MIDDLE_BASE], dtype=np.float32)
#         ring_base = np.array(landmarks[self.RING_BASE], dtype=np.float32)
#         pinky_base = np.array(landmarks[self.PINKY_BASE], dtype=np.float32)
        
#         # Calculate the true geometric palm center
#         true_palm_center = np.array([
#             np.mean([wrist[0], thumb_cmc[0], index_base[0], thumb_base[0], middle_base[0], ring_base[0], pinky_base[0]]),
#             np.mean([wrist[1], thumb_cmc[1], index_base[1], thumb_base[1], middle_base[1], ring_base[1], pinky_base[1]])
#         ], dtype=np.float32)
        
#         # Create central palm region by scaling points toward the true center
#         central_wrist = true_palm_center + self.palm_scale * (wrist - true_palm_center)
#         central_thumb_cmc = true_palm_center + self.palm_scale * (thumb_cmc - true_palm_center)
#         central_thumb_base = true_palm_center + self.palm_scale * (thumb_base - true_palm_center)
#         central_index = true_palm_center + self.palm_scale * (index_base - true_palm_center)
#         central_middle = true_palm_center + self.palm_scale * (middle_base - true_palm_center)
#         central_ring = true_palm_center + self.palm_scale * (ring_base - true_palm_center)
#         central_pinky = true_palm_center + self.palm_scale * (pinky_base - true_palm_center)

        
#         # Create contour points for the central palm region, then apply the explicit shift (center_offset)
#         central_palm_points = np.array([
#             central_wrist,
#             central_thumb_cmc,
#             central_thumb_base,
#             central_index,
#             central_middle,
#             central_ring,
#             central_pinky
#         ], dtype=np.float32)
        
#         # Apply the explicit center shift as a translation
#         central_palm_points[:, 0] += self.center_offset[0]
#         central_palm_points[:, 1] += self.center_offset[1]
        
#         # The shifted center
#         shifted_center = np.array([
#             true_palm_center[0] + self.center_offset[0],
#             true_palm_center[1] + self.center_offset[1]
#         ])
        
#         if self.shape == "circle":
#             # Calculate a radius based on average distance to central palm points
#             distances = np.linalg.norm(central_palm_points - shifted_center, axis=1)
#             radius = int(np.mean(distances) * self.size_multiplier)
            
#             # Generate circle contour using OpenCV
#             center_int = (int(shifted_center[0]), int(shifted_center[1]))
#             pts = cv2.ellipse2Poly(center_int, (radius, radius), 0, 0, 360, 5)
#             return pts.reshape((-1, 1, 2))
            
#         elif self.shape == "rectangle":
#             # Generate a bounding rectangle matching the scaled area
#             x, y, w, h = cv2.boundingRect(central_palm_points.astype(np.int32))
            
#             # Apply size multiplier to width and height, keeping shifted center same
#             new_w = int(w * self.size_multiplier)
#             new_h = int(h * self.size_multiplier)
            
#             # Adjust x and y so the rectangle expands/shrinks from its center
#             center_rect_x = x + w / 2
#             center_rect_y = y + h / 2
#             new_x = int(center_rect_x - new_w / 2)
#             new_y = int(center_rect_y - new_h / 2)
            
#             rect_contour = np.array([
#                 [[new_x, new_y]], 
#                 [[new_x + new_w, new_y]], 
#                 [[new_x + new_w, new_y + new_h]], 
#                 [[new_x, new_y + new_h]]
#             ], dtype=np.int32)
#             return rect_contour
            
#         else:
#             # Default to polygon (convex hull for smooth boundaries)
#             hull = cv2.convexHull(central_palm_points.astype(np.int32))
#             if self.size_multiplier != 1.0:
#                 # Scale polygon points directly
#                 hull_points = hull.reshape(-1, 2).astype(np.float32)
#                 hull_points = shifted_center + self.size_multiplier * (hull_points - shifted_center)
#                 hull = hull_points.reshape(-1, 1, 2).astype(np.int32)
                
#             return hull

#     def extract_palm_region(self, landmarks: List[Tuple[int, int]], 
#                            image_shape: Tuple[int, int]) -> PalmRegion:
#         """
#         Extract the central palm region from hand landmarks.
#         Creates a smaller region focused on the center of the palm.
        
#         Args:
#             landmarks: List of all 21 hand landmarks as (x, y) tuples
#             image_shape: Tuple of (height, width) of the image
            
#         Returns:
#             PalmRegion object containing contour, center, and area
#         """
#         # Create central palm contour
#         central_palm_contour = self.create_central_palm_contour(landmarks)
        
#         # Calculate palm center from the original palm landmarks
#         palm_landmarks = self.get_palm_landmarks(landmarks)
#         center = self.calculate_palm_center(palm_landmarks)
        
#         # Calculate area of the central palm region
#         area = cv2.contourArea(central_palm_contour)
        
#         return PalmRegion(
#             contour=central_palm_contour,
#             center=center,
#             area=area
#         )














"""
PalmExtractor component for identifying palm region from hand landmarks.
"""
from typing import List, Tuple
import numpy as np
import cv2

from src.data_models import PalmRegion


class PalmExtractor:
    """Extracts central palm region from hand landmarks."""
    
    # Key landmark indices for central palm calculation
    WRIST = 0
    THUMB_CMC = 1
    THUMB_BASE = 2
    INDEX_BASE = 5
    MIDDLE_BASE = 9
    RING_BASE = 13
    PINKY_BASE = 17
    
    def __init__(self, palm_scale: float = 0.8, shape: str = "circle", center_offset: Tuple[int, int] = (-10, 0), size_multiplier: float = 1.3):
        """
        Initialize the PalmExtractor.
        
        Args:
            palm_scale: Scale factor for central palm region (0.0-1.0).
                        Lower values = smaller central palm region.
                        Default 0.6 extracts the central palm area.
            shape: Shape to extract ("circle", "rectangle", "polygon").
                   Defaults to "circle", can be passed via config parameters.
            center_offset: (x, y) pixel offset to manually shift the palm center.
                           Useful for fine-tuning the exact center position.
            size_multiplier: Direct multiplier for the final shape size (radius/width/height).
                             Defaults to 1.0.
        """
        self.palm_scale = palm_scale
        self.shape = shape.lower()
        self.center_offset = center_offset
        self.size_multiplier = size_multiplier

    # def _get_rotated_offset(self, wrist: Tuple[int, int], middle_base: Tuple[int, int]) -> Tuple[float, float]:
    #     """
    #     Calculates the center offset dynamically rotated to match the hand's current orientation.
    #     """
    #     vec = np.array(middle_base) - np.array(wrist)
    #     alpha = np.arctan2(vec[1], vec[0])
        
    #     # Base angle is -pi/2 (upright hand in OpenCV image coordinates where Y goes down)
    #     base_angle = -np.pi / 2
    #     delta = alpha - base_angle
        
    #     # Rotate the fixed offset based on the hand's angle difference
    #     ox, oy = self.center_offset
    #     rot_ox = ox * np.cos(delta) - oy * np.sin(delta)
    #     rot_oy = ox * np.sin(delta) + oy * np.cos(delta)
        
    #     return rot_ox, rot_oy

    def _get_rotated_offset(self, wrist: Tuple[int, int], middle_base: Tuple[int, int]) -> Tuple[float, float]:
        """
        Calculates the center offset dynamically scaled and rotated to match 
        the hand's current orientation and distance from the camera.
        """
        vec = np.array(middle_base) - np.array(wrist)
        
        # 1. Calculate Rotation Angle
        alpha = np.arctan2(vec[1], vec[0])
        # Base angle is -pi/2 (upright hand in OpenCV image coordinates where Y goes down)
        base_angle = -np.pi / 2
        delta = alpha - base_angle
        
        # 2. Calculate Dynamic Scale (Distance from camera)
        current_hand_length = np.linalg.norm(vec)
        # Assuming the original (25, -35) was tuned for a hand length of roughly 150 pixels.
        # Agar offset abhi bhi thoda odd lage, toh is 150.0 ko thoda kam ya zyada karke dekh lijiye.
        reference_length = 150.0 
        scale_factor = current_hand_length / reference_length
        
        # 3. Apply Scale and Rotation to Offset
        ox, oy = self.center_offset
        
        # Pehle offset ko hand ke size ke according scale karein
        scaled_ox = ox * scale_factor
        scaled_oy = oy * scale_factor
        
        # Fir scaled offset ko rotate karein
        rot_ox = scaled_ox * np.cos(delta) - scaled_oy * np.sin(delta)
        rot_oy = scaled_ox * np.sin(delta) + scaled_oy * np.cos(delta)
        
        return rot_ox, rot_oy
    
    def get_palm_landmarks(self, landmarks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Extract palm-specific landmarks (wrist and finger base landmarks).
        
        Args:
            landmarks: List of all 21 hand landmarks as (x, y) tuples
            
        Returns:
            List of palm landmarks (wrist at index 0, and finger bases at 5, 9, 13, 17)
        """
        if not landmarks or len(landmarks) < 21:
            raise ValueError("Invalid landmarks: expected 21 landmarks")
        
        palm_indices = [self.WRIST, self.THUMB_CMC, self.THUMB_BASE, self.INDEX_BASE, self.MIDDLE_BASE, 
                       self.RING_BASE, self.PINKY_BASE]
        palm_landmarks = [landmarks[i] for i in palm_indices]
        return palm_landmarks
    
    def calculate_palm_center(self, palm_landmarks: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Compute the centroid of the palm region.
        
        Args:
            palm_landmarks: List of palm landmark coordinates
            
        Returns:
            (x, y) tuple representing the center point of the palm
        """
        if not palm_landmarks:
            raise ValueError("Palm landmarks cannot be empty")
        
        # Index 0 is WRIST and Index 4 is MIDDLE_BASE in the extracted palm_landmarks list
        wrist = palm_landmarks[0]
        middle_base = palm_landmarks[4]
        
        # Get rotated offset
        rot_ox, rot_oy = self._get_rotated_offset(wrist, middle_base)
        
        # Calculate centroid as the mean of all palm landmark coordinates
        x_coords = [point[0] for point in palm_landmarks]
        y_coords = [point[1] for point in palm_landmarks]
        
        # Add rotated offset instead of the fixed one
        center_x = int(np.mean(x_coords) + rot_ox)
        center_y = int(np.mean(y_coords) + rot_oy)
        
        return (center_x, center_y)
    
    def create_central_palm_contour(self, landmarks: List[Tuple[int, int]]) -> np.ndarray:
        """
        Create a contour for the central palm region by scaling inward from the outer palm boundary.
        This creates a smaller region focused on the center of the palm.
        
        Args:
            landmarks: List of all 21 hand landmarks
            
        Returns:
            Numpy array of contour points for the central palm region
        """
        # Get the outer palm boundary landmarks
        wrist = np.array(landmarks[self.WRIST], dtype=np.float32)
        thumb_cmc = np.array(landmarks[self.THUMB_CMC], dtype=np.float32)
        thumb_base = np.array(landmarks[self.THUMB_BASE], dtype=np.float32)
        index_base = np.array(landmarks[self.INDEX_BASE], dtype=np.float32)
        middle_base = np.array(landmarks[self.MIDDLE_BASE], dtype=np.float32)
        ring_base = np.array(landmarks[self.RING_BASE], dtype=np.float32)
        pinky_base = np.array(landmarks[self.PINKY_BASE], dtype=np.float32)
        
        # Calculate the true geometric palm center
        true_palm_center = np.array([
            np.mean([wrist[0], thumb_cmc[0], index_base[0], thumb_base[0], middle_base[0], ring_base[0], pinky_base[0]]),
            np.mean([wrist[1], thumb_cmc[1], index_base[1], thumb_base[1], middle_base[1], ring_base[1], pinky_base[1]])
        ], dtype=np.float32)
        
        # Create central palm region by scaling points toward the true center
        central_wrist = true_palm_center + self.palm_scale * (wrist - true_palm_center)
        central_thumb_cmc = true_palm_center + self.palm_scale * (thumb_cmc - true_palm_center)
        central_thumb_base = true_palm_center + self.palm_scale * (thumb_base - true_palm_center)
        central_index = true_palm_center + self.palm_scale * (index_base - true_palm_center)
        central_middle = true_palm_center + self.palm_scale * (middle_base - true_palm_center)
        central_ring = true_palm_center + self.palm_scale * (ring_base - true_palm_center)
        central_pinky = true_palm_center + self.palm_scale * (pinky_base - true_palm_center)

        # Create contour points for the central palm region
        central_palm_points = np.array([
            central_wrist,
            central_thumb_cmc,
            central_thumb_base,
            central_index,
            central_middle,
            central_ring,
            central_pinky
        ], dtype=np.float32)
        
        # Get rotated offset
        rot_ox, rot_oy = self._get_rotated_offset(landmarks[self.WRIST], landmarks[self.MIDDLE_BASE])
        
        # Apply the explicit rotated center shift as a translation
        central_palm_points[:, 0] += rot_ox
        central_palm_points[:, 1] += rot_oy
        
        # The shifted center
        shifted_center = np.array([
            true_palm_center[0] + rot_ox,
            true_palm_center[1] + rot_oy
        ])
        
        if self.shape == "circle":
            # Calculate a radius based on average distance to central palm points
            distances = np.linalg.norm(central_palm_points - shifted_center, axis=1)
            radius = int(np.mean(distances) * self.size_multiplier)
            
            # Generate circle contour using OpenCV
            center_int = (int(shifted_center[0]), int(shifted_center[1]))
            pts = cv2.ellipse2Poly(center_int, (radius, radius), 0, 0, 360, 5)
            return pts.reshape((-1, 1, 2))
            
        elif self.shape == "rectangle":
            # Generate a bounding rectangle matching the scaled area
            x, y, w, h = cv2.boundingRect(central_palm_points.astype(np.int32))
            
            # Apply size multiplier to width and height, keeping shifted center same
            new_w = int(w * self.size_multiplier)
            new_h = int(h * self.size_multiplier)
            
            # Adjust x and y so the rectangle expands/shrinks from its center
            center_rect_x = x + w / 2
            center_rect_y = y + h / 2
            new_x = int(center_rect_x - new_w / 2)
            new_y = int(center_rect_y - new_h / 2)
            
            rect_contour = np.array([
                [[new_x, new_y]], 
                [[new_x + new_w, new_y]], 
                [[new_x + new_w, new_y + new_h]], 
                [[new_x, new_y + new_h]]
            ], dtype=np.int32)
            return rect_contour
            
        else:
            # Default to polygon (convex hull for smooth boundaries)
            hull = cv2.convexHull(central_palm_points.astype(np.int32))
            if self.size_multiplier != 1.0:
                # Scale polygon points directly
                hull_points = hull.reshape(-1, 2).astype(np.float32)
                hull_points = shifted_center + self.size_multiplier * (hull_points - shifted_center)
                hull = hull_points.reshape(-1, 1, 2).astype(np.int32)
                
            return hull

    def extract_palm_region(self, landmarks: List[Tuple[int, int]], 
                            image_shape: Tuple[int, int]) -> PalmRegion:
        """
        Extract the central palm region from hand landmarks.
        Creates a smaller region focused on the center of the palm.
        
        Args:
            landmarks: List of all 21 hand landmarks as (x, y) tuples
            image_shape: Tuple of (height, width) of the image
            
        Returns:
            PalmRegion object containing contour, center, and area
        """
        # Create central palm contour
        central_palm_contour = self.create_central_palm_contour(landmarks)
        
        # Calculate palm center from the original palm landmarks
        palm_landmarks = self.get_palm_landmarks(landmarks)
        center = self.calculate_palm_center(palm_landmarks)
        
        # Calculate area of the central palm region
        area = cv2.contourArea(central_palm_contour)
        
        return PalmRegion(
            contour=central_palm_contour,
            center=center,
            area=area
        )