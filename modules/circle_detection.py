import cv2
import numpy as np

def detect_circles_in_frame(frame, callback=None, min_radius=30, max_radius=50, min_dist=80, param1=120, param2=20, intensity_threshold=100):
    """
    Detect circles in a provided frame using Hough Circle Transform.
    
    Args:
        frame (numpy.ndarray): Input frame to process (BGR).
        callback (function, optional): Callback function to process detected circle parameters (x, y, r).
        min_radius (int): Minimum radius of circles to detect (in pixels).
        max_radius (int): Maximum radius of circles to detect (in pixels).
        min_dist (int): Minimum distance between circle centers (in pixels).
        param1 (float): Higher threshold for Canny edge detector.
        param2 (float): Accumulator threshold for circle detection.
        intensity_threshold (float, optional): Minimum average intensity in the circle region.
    
    Returns:
        numpy.ndarray: Frame with detected circles drawn.
    """
    # Resize frame to ensure consistent processing
    frame = cv2.resize(frame, (800, 600))
    
    # for color detection feature
    frame_processing = frame.copy()
    hsv = cv2.cvtColor(frame_processing, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([156, 93, 137]) #np.array([20, 100, 100])
    upper_yellow = np.array([171, 255, 255])

    # Create a Mask for Yellow Color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply Morphological Operations (Optional but Recommended for cleaning mask)
    # Erode to remove small noise
    mask = cv2.erode(mask, None, iterations=2)
    # Dilate to fill gaps
    mask = cv2.dilate(mask, None, iterations=2)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame_processing, cv2.COLOR_BGR2GRAY)
    # Only keep the grayscale values where the mask is white (yellow color)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # blurred = cv2.medianBlur(gray, 5)
    blurred = cv2.medianBlur(masked_gray, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5, # inverse ratio of the accumulator resolution,  A value of 1.5 means the accumulator will have a resolution that's 1/1.5 times that of the input image
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    # if circles is not None:
    #     circles = np.round(circles[0, :]).astype("int")
    #     filtered_circles = []

    #     for (x, y, r) in circles:
    #         if intensity_threshold is not None:
    #             mask = np.zeros_like(gray)
    #             cv2.circle(mask, (x, y), r, 255, -1)
    #             mean_intensity = cv2.mean(gray, mask=mask)[0]
    #             if mean_intensity < intensity_threshold:
    #                 continue

    #         filtered_circles.append((x, y, r))
    #         cv2.circle(frame, (x, y), r, (0, 255, 0), 3)  # Draw circle
    #         # cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  # Draw center
    #         if callback:
    #             callback(x, y, r)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # No need for intensity thresholding anymore if we're filtering by color,
        # unless you want to filter yellow circles based on their *luminosity*.
        # For this specific request, we'll remove it, but you could add it back if needed.
        # filtered_circles = [] # This list might not be necessary if we draw directly after checks

        for (x, y, r) in circles:
            # We already filtered by color. We can add a simple check if the circle is not too close to the edges
            # or any other criteria you might have.

            # Draw a yellow circle on the display frame
            cv2.circle(frame, (x, y), r, (0, 255, 255), 3)  # Yellow circle (BGR: Blue=0, Green=255, Red=255)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  # Red center for clarity

    return circles