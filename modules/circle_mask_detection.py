import cv2
import numpy as np

def get_circle_mask(frame, background, threshold=30, min_radius=20, max_radius=150, color_lower=(47, 56, 103), color_upper=(69, 139, 255)):
    """
    Enhanced object mask that specifically detects circular objects
    """

    # Convert frame to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create color mask
    color_mask = cv2.inRange(hsv, color_lower, color_upper)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Standard motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if background is None:
        return None, []
    
    # Background subtraction
    diff = cv2.absdiff(background, blur)
    _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Clean up the motion mask
    kernel = np.ones((5,5), np.uint8)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    
    # Combine motion and color masks
    combined_mask = cv2.bitwise_and(motion_mask, color_mask)

    # Detect circles using HoughCircles on the current frame
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,  # Minimum distance between circle centers
        param1=50,   # Upper threshold for edge detection
        param2=45,   # Accumulator threshold for center detection
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    # Create circle mask
    circle_mask = np.zeros_like(motion_mask)
    detected_circles = []
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Only consider circles that overlap with motion areas
            circle_roi = motion_mask[max(0, y-r):min(motion_mask.shape[0], y+r+1),
                                   max(0, x-r):min(motion_mask.shape[1], x+r+1)]
            
            if np.count_nonzero(circle_roi) > 50:  # Minimum motion pixels in circle area
                cv2.circle(circle_mask, (x, y), r, 255, -1)
                detected_circles.append((x, y, r))
    
    # Combine motion mask with circle detection
    combined_mask = cv2.bitwise_and(combined_mask, circle_mask)
    
    return combined_mask, detected_circles