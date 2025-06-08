import cv2
import sys
import os
import time
import numpy as np
import pyautogui
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.edge_detection import EdgeDetector
from modules.game_visual import Balloon

def move_window_to_second_monitor(window_name, monitor_index=1):
    """
    Move a window to the second monitor if available.
    
    Args:
        window_name (str): Name of the OpenCV window
        monitor_index (int): Index of the monitor (0-based, 1 for second monitor)
    """
    try:
        screens = pyautogui.size()
        x_offset = screens.width * monitor_index
        y_offset = 100
        cv2.moveWindow(window_name, x_offset, y_offset)
    except ImportError:
        print("pyautogui not installed, window will stay on primary monitor")
    except Exception as e:
        print(f"Could not move window: {str(e)}")

def get_calibration_points(cap, window_name="Calibration"):
    """
    Allow user to select four corners of the monitor in the camera feed.
    
    Args:
        cap: OpenCV VideoCapture object
        window_name: Name of the calibration window
    
    Returns:
        list: Four (x, y) points defining the monitor's corners
    """
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            print(f"Point {len(points)} selected: ({x}, {y})")
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("Click the four corners of the monitor in the camera feed (top-left, top-right, bottom-right, bottom-left). Press 'c' to confirm.")
    
    while len(points) < 4:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera during calibration")
            return None
        
        # Draw selected points
        for pt in points:
            cv2.circle(frame, (pt[0], pt[1]), 5, (0, 255, 0), -1)
        
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('c') and len(points) == 4:
            break
    
    cv2.destroyWindow(window_name)
    return points

def save_calibration_points(points, filename="calibration.json"):
    """
    Save calibration points to a JSON file.
    
    Args:
        points: List of four (x, y) points
        filename: Path to save the JSON file
    """
    with open(filename, 'w') as f:
        json.dump(points, f)

def load_calibration_points(filename="calibration.json"):
    """
    Load calibration points from a JSON file.
    
    Args:
        filename: Path to the JSON file
    
    Returns:
        list: Four (x, y) points or None if file doesn't exist
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def get_perspective_transform(src_points, screen_width=800, screen_height=600):
    """
    Compute perspective transform matrix.
    
    Args:
        src_points: List of four (x, y) points from camera feed
        screen_width: Target width of the game screen
        screen_height: Target height of the game screen
    
    Returns:
        numpy.ndarray: 3x3 homography matrix
    """
    dst_points = np.float32([
        [0, 0],
        [screen_width, 0],
        [screen_width, screen_height],
        [0, screen_height]
    ])
    src_points = np.float32(src_points)
    return cv2.getPerspectiveTransform(src_points, dst_points)

def main():
    # Initialize edge detector
    detector = EdgeDetector(low_threshold=100, high_threshold=200)
    
    # Game screen settings
    screen_width = 800
    screen_height = 600
    
    # Input paths for balloon images
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    balloon_files = ["balloon1.png", "balloon2.png", "balloon3.png"]
    image_paths = [os.path.join(base_dir, "assets", file) for file in balloon_files]
    
    # Validate image paths
    valid_paths = []
    for path in image_paths:
        if os.path.exists(path) and os.path.isfile(path):
            valid_paths.append(path)
        else:
            print(f"Error: Image file does not exist or is not a file: {path}")
    
    if not valid_paths:
        print("Error: No valid image files found. Exiting.")
        return
    
    # Initialize balloons
    balloons = []
    try:
        for path in valid_paths:
            balloon = Balloon(path, screen_width, screen_height)
            balloons.append(balloon)
    except ValueError as e:
        print(f"Error loading balloon images: {str(e)}")
        return
    
    # Initialize camera (try index 1 for external camera)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Load or perform calibration
    calibration_file = os.path.join(base_dir, "calibration.json")
    calibration_points = load_calibration_points(calibration_file)
    if calibration_points is None or len(calibration_points) != 4:
        print("Performing camera calibration...")
        calibration_points = get_calibration_points(cap)
        if calibration_points is None or len(calibration_points) != 4:
            print("Error: Calibration failed. Exiting.")
            cap.release()
            return
        save_calibration_points(calibration_points, calibration_file)
        print(f"Calibration points saved to {calibration_file}")
    
    # Compute perspective transform
    transform_matrix = get_perspective_transform(calibration_points, screen_width, screen_height)
    
    # Create windows
    cv2.namedWindow("Balloon Display", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Camera Edges", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Warped Camera Feed", cv2.WINDOW_NORMAL)
    
    # Move Balloon Display window to second monitor
    move_window_to_second_monitor("Balloon Display", monitor_index=1)
    
    # Create blank game screen
    game_screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    try:
        start_time = time.time()
        while time.time() - start_time < 120:  # Run for 120 seconds
            # Reset game screen
            game_screen[:] = 0  # Clear to black
            
            # Update and draw balloons
            for balloon in balloons:
                balloon.update()
                balloon.move()
                balloon.draw(game_screen, show_bbox=False)
            
            # Display game screen
            cv2.imshow("Balloon Display", game_screen)
            
            # Process camera feed
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from camera")
                break
            
            # Apply perspective transform
            warped_frame = cv2.warpPerspective(frame, transform_matrix, (screen_width, screen_height))
            
            # Convert warped frame to grayscale for edge detection
            gray_frame = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
            # Apply edge detection
            edges = detector.detect_edges_from_array(gray_frame)
            
            # Display warped feed and edges
            # cv2.imshow("Warped Camera Feed", warped_frame)
            cv2.imshow("Camera Edges", edges)
            
            # Check for 'q' to quit or 'r' to recalibrate
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Recalibrating...")
                calibration_points = get_calibration_points(cap)
                if calibration_points and len(calibration_points) == 4:
                    save_calibration_points(calibration_points, calibration_file)
                    transform_matrix = get_perspective_transform(calibration_points, screen_width, screen_height)
                    print(f"New calibration points saved to {calibration_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()