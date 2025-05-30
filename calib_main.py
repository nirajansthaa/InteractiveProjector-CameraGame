import cv2
import numpy as np
import os
import time
from screeninfo import get_monitors
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.camera_capture import capture_camera_points
from modules.projector_display import show_calibration_pattern
from modules.calibrate import compute_homography
from modules.game_visual import Balloon
from modules.circle_detection import detect_circles_in_frame
from modules.hit_detection import HitDetector

# Configuration
CAMERA_ID = 1
SCREEN_WIDTH, SCREEN_HEIGHT = 1900, 1000  # Projector resolution
TARGET_FPS = 30

# Get monitor info for projector and camera feed windows
monitors = get_monitors()
if not monitors:
    print("Error: No monitors detected")
    exit()
main_screen = monitors[0]
external_screen = monitors[1] if len(monitors) > 1 else monitors[0]
main_origin = (main_screen.x, main_screen.y)
external_origin = (external_screen.x, external_screen.y)

# Initialize camera
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print(f"Error: Could not open camera with ID {CAMERA_ID}")
    exit()

# Calibration
def calibrate_system():
    # Step 1: Show calibration pattern on projector
    projector_points = show_calibration_pattern(screen_resolution=(SCREEN_WIDTH, SCREEN_HEIGHT))
    
    # Step 2: Capture camera points
    global clicked_points
    clicked_points = []  # Reset global variable from camera_capture
    camera_points = capture_camera_points(resolution=(SCREEN_WIDTH, SCREEN_HEIGHT))
    if len(camera_points) != 4:
        print("Error: Exactly 4 points required for calibration")
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    # Debug: Print points to verify
    print("Camera points:", camera_points)
    print("Projector points:", projector_points)
    
    # Step 3: Compute homography
    try:
        homography = compute_homography(camera_points, projector_points)
        return homography
    except cv2.error as e:
        print(f"Error computing homography: {e}")
        return None

# Load balloons
base_dir = os.path.dirname(os.path.abspath(__file__))
balloon_paths = [
    os.path.join(base_dir, 'assets', 'balloon1.png'),
    os.path.join(base_dir, 'assets', 'balloon2.png'),
    os.path.join(base_dir, 'assets', 'balloon3.png')
]
balloons = []
for path in balloon_paths:
    if not os.path.exists(path):
        print(f"Warning: Image file {path} does not exist. Skipping.")
        continue
    try:
        balloons.append(Balloon(path, SCREEN_WIDTH, SCREEN_HEIGHT))
        print(f"Loaded balloon image: {path}")
    except ValueError as e:
        print(f"Warning: {e}. Skipping image {path}.")
if not balloons:
    print("Error: No valid balloon images loaded. Exiting.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Initialize hit detector
hit_detector = HitDetector()

def hit_callback(x, y, r, balloon):
    balloon.pop()

# Create windows
cv2.namedWindow("Game Display", cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
cv2.moveWindow("Game Display", external_origin[0], external_origin[1])
cv2.moveWindow("Camera Feed", main_origin[0], main_origin[1])
cv2.resizeWindow("Game Display", SCREEN_WIDTH, SCREEN_HEIGHT)
cv2.resizeWindow("Camera Feed", SCREEN_WIDTH, SCREEN_HEIGHT)

# Run calibration
homography = calibrate_system()
if homography is None:
    print("Calibration failed. Exiting.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Game loop
frame_time = 1.0 / TARGET_FPS
while True:
    start_time = time.time()
    
    # Create white background for game display
    game_frame = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) * 255
    
    # Read camera frame
    ret, camera_frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    
    # Flip camera frame for consistency
    camera_frame = cv2.resize(camera_frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
    camera_frame = cv2.flip(camera_frame, 1)  # Re-enable flip for consistency
    
    # Move and draw balloons
    for balloon in balloons:
        balloon.update()
        balloon.move()
        balloon.draw(game_frame, show_bbox=False)
    
    # Detect circles in camera frame
    circles, processed_frame = detect_circles_in_frame(
        camera_frame,
        min_radius=30,
        max_radius=50,
        min_dist=80,
        param1=120,
        param2=30,
        intensity_threshold=100
    )
    
    # Transform and visualize circles
    debug_visualization = True
    if circles is not None and isinstance(circles, np.ndarray) and circles.size > 0:
        circles = np.round(circles[0, :]).astype("int")
        if circles.ndim == 1:
            circles = circles.reshape(1, -1)
        # Transform circles to projector coordinates
        circles_transformed = []
        for (x, y, r) in circles:
            point = np.array([[x, y]], dtype='float32')
            projected_point = cv2.perspectiveTransform(point[None, :, :], homography)[0][0]
            x_p, y_p = int(projected_point[0]), int(projected_point[1])
            circles_transformed.append([x_p, y_p, r])
            if debug_visualization:
                cv2.circle(game_frame, (x_p, y_p), r, (0, 255, 255), 3)  # Yellow outline
                cv2.circle(game_frame, (x_p, y_p), 2, (0, 0, 255), 3)    # Red center
        circles_transformed = np.array(circles_transformed, dtype="int")
        hit_detector.process_detected_circles(circles_transformed, balloons, callback=hit_callback)
    
    processed_frame = cv2.resize(processed_frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
    # Show frames
    cv2.imshow("Game Display", game_frame)
    cv2.imshow("Camera Feed", processed_frame)
    
    # Frame rate control
    elapsed = time.time() - start_time
    sleep_time = max(1, int((frame_time - elapsed) * 1000))
    key = cv2.waitKey(sleep_time) & 0xFF
    if key in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()