import cv2
import sys
import os
import time
import numpy as np
import pyautogui
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
    
    # Create windows
    cv2.namedWindow("Balloon Display", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Camera Edges", cv2.WINDOW_NORMAL)
    
    # Move Balloon Display window to second monitor
    move_window_to_second_monitor("Balloon Display", monitor_index=1)
    
    # Create blank game screen
    game_screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    try:
        start_time = time.time()
        while time.time() - start_time < 120:  # Run for 30 seconds
            # Reset game screen
            game_screen[:] = 0  # Clear to black
            
            # Update and draw balloons
            for balloon in balloons:
                balloon.update()
                balloon.move()
                balloon.draw(game_screen, show_bbox=False)  # Show bounding box for debugging
            
            # Display game screen
            cv2.imshow("Balloon Display", game_screen)
            
            # Process camera feed
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from camera")
                break
            
            # Convert frame to grayscale for edge detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply edge detection
            edges = detector.detect_edges_from_array(gray_frame)
            
            # Display raw edges
            cv2.imshow("Camera Edges", edges)
            
            # Check for 'q' key to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()