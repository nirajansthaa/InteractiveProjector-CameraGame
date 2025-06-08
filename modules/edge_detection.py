import cv2
import numpy as np

class EdgeDetector:
    def __init__(self, low_threshold=50, high_threshold=150):
        """
        Initialize edge detector with Canny thresholds.
        
        Args:
            low_threshold (int): Lower threshold for Canny edge detection
            high_threshold (int): Upper threshold for Canny edge detection
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def detect_edges(self, image_path):
        """
        Detect edges in an image using Canny edge detection.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            numpy.ndarray: Edge-detected image
        """
        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Perform Canny edge detection
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
        
        return edges

    def detect_edges_from_array(self, image_array):
        """
        Detect edges from a NumPy array (e.g., camera frame).
        
        Args:
            image_array (numpy.ndarray): Input image as a NumPy array (grayscale)
            
        Returns:
            numpy.ndarray: Edge-detected image
        """
        if image_array is None or len(image_array.shape) != 2:
            raise ValueError("Invalid input: Expected grayscale image array")
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image_array, (5, 5), 0)
        
        # Perform Canny edge detection
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
        
        return edges

    def adjust_thresholds(self, low_threshold, high_threshold):
        """
        Adjust Canny edge detection thresholds.
        
        Args:
            low_threshold (int): New lower threshold
            high_threshold (int): New upper threshold
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold