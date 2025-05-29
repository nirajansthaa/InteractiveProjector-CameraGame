import cv2, os
import numpy as np
from modules.game_visual import Balloon
from modules.circle_detection import detect_circles_in_frame
from modules.hit_detection import HitDetector
from screeninfo import get_monitors
from calibration import Calibration  # Make sure your calibration file is named calibration.py

# Initialize Calibration
calibration = Calibration(camera_id=1, save_path="calibration_points.json")

# Simulated resolution for both windows
screen_width, screen_height = 1800, 1000

# Get screen info
monitors = get_monitors()
main_screen = monitors[0]  # Usually primary
external_screen = monitors[1] if len(monitors) > 1 else monitors[0]  # Fallback

# Get top-left coordinates
main_origin = (main_screen.x, main_screen.y)
external_origin = (external_screen.x, external_screen.y)

# Balloon image paths (relative to project root)
base_dir = os.path.dirname(os.path.abspath(__file__))
balloon_paths = [
        os.path.join(base_dir, 'assets', 'balloon1.png'),
        os.path.join(base_dir,  'assets', 'balloon2.png'),
        os.path.join(base_dir,  'assets', 'balloon3.png')
    ]
    
    # Initialize balloons, skipping invalid images
balloons = []
    
for path in balloon_paths:
    try:
        balloons.append(Balloon(path, screen_width, screen_height))
        print(f"Loaded balloon image: {path}")
    except ValueError as e:
        print(f"Warning: {e}. Skipping this image.")

if not balloons:
    print("Error: No valid balloon images loaded. Exiting.")



# Open camera
cap = cv2.VideoCapture(1)  # Use external camera (adjust if needed)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Initialize hit detector
hit_detector = HitDetector()


def hit_callback(x, y, r, balloon):
    balloon.pop()

# Create and move the windows
cv2.namedWindow("Game Display", cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

# Move to different screens
# cv2.moveWindow("Game Display", 1920, 0)  # Adjust to position on external monitor
cv2.moveWindow("Camera Feed", main_origin[0], main_origin[1])
cv2.resizeWindow("Game Display", screen_width, screen_height)

# cv2.moveWindow("Camera Feed", 0, 0)  # On main screen
cv2.moveWindow("Game Display", external_origin[0], external_origin[1])
cv2.resizeWindow("Camera Feed", screen_width, screen_height)

# Try to load existing calibration points
src_points = calibration.load_points()

# If not found, run calibration
if src_points is None:
    print("No calibration found. Starting manual calibration...")
    src_points = calibration.capture_and_select_points()
    if src_points is None:
        print("Calibration failed or canceled. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Define fixed destination points (based on game screen resolution)
dst_points = np.array([
    [0, 0],
    [screen_width - 1, 0],
    [screen_width - 1, screen_height - 1],
    [0, screen_height - 1]
], dtype=np.float32)

# Compute perspective transformation matrix
transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

while True:
    # Create a white background for the game window
    game_frame = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255

    ret, camera_frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    for balloon in balloons:
            balloon.update()
            balloon.move()
            balloon.draw(game_frame, show_bbox=False)

    # # Resize camera feed to match display resolution
    # camera_frame = cv2.resize(camera_frame, (screen_width, screen_height))

    # Apply perspective transform
    camera_frame = cv2.warpPerspective(camera_frame, transform_matrix, (screen_width, screen_height))
    
    # Detect circles and draw them on the same frame
    circles = detect_circles_in_frame(
            camera_frame,
            min_radius=30,
            max_radius=50,
            min_dist=80,
            param1=120,
            param2=20,
            intensity_threshold=100
        )
    
    # Process detected circles for hits
    if circles is not None:
        hit_detector.process_detected_circles(circles, balloons, callback=hit_callback)
         
    # Show both windows
    cv2.imshow("Game Display", game_frame)
    cv2.imshow("Camera Feed", camera_frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
