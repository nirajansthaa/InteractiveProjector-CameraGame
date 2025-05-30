import cv2, os, time
import numpy as np
from modules.game_visual import Balloon
from modules.circle_detection import detect_circles_in_frame
from modules.hit_detection import HitDetector
from screeninfo import get_monitors

# Configuration
screen_width, screen_height = 1800, 1000
target_fps = 30
circle_detection_params = {
    "min_radius": 30, "max_radius": 50, "min_dist": 80,
    "param1": 120, "param2": 20, "intensity_threshold": 100
}

# Monitor setup
monitors = get_monitors()
if not monitors:
    print("Error: No monitors detected")
    exit()
main_screen = monitors[0]
external_screen = monitors[1] if len(monitors) > 1 else monitors[0]
main_origin = (main_screen.x, main_screen.y)
external_origin = (external_screen.x, external_screen.y)

# Camera setup
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Window setup
cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
cv2.namedWindow("Game Display", cv2.WINDOW_NORMAL)
cv2.moveWindow("Camera Feed", main_origin[0], main_origin[1])
cv2.moveWindow("Game Display", external_origin[0], external_origin[1])
cv2.resizeWindow("Camera Feed", screen_width, screen_height)
cv2.resizeWindow("Game Display", screen_width, screen_height)

# # Calibration
# calibration = Calibration(camera_id=1, save_path="calibration_points.json")
# src_points = calibration.load_points()
# if src_points is None:
#     print("No calibration found. Starting manual calibration...")
#     ret, calib_frame = cap.read()
#     if not ret:
#         print("Error: Failed to grab calibration frame")
#         cap.release()
#         cv2.destroyAllWindows()
#         exit()
#     calib_frame = cv2.flip(calib_frame, 1)
#     cv2.imshow("Camera Feed", calib_frame)
#     src_points = calibration.capture_and_select_points()
#     if src_points is None:
#         print("Calibration failed or canceled. Exiting.")
#         cap.release()
#         cv2.destroyAllWindows()
#         exit()

# dst_points = np.array([[0, 0], [screen_width-1, 0], [screen_width-1, screen_height-1], [0, screen_height-1]], dtype=np.float32)
# try:
#     transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
# except cv2.error as e:
#     print(f"Error computing perspective transform: {e}")
#     cap.release()
#     cv2.destroyAllWindows()
#     exit()

# Balloon setup
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
        balloons.append(Balloon(path, screen_width, screen_height))
    except ValueError as e:
        print(f"Warning: {e}. Skipping image {path}.")
if not balloons:
    print("Error: No valid balloon images loaded. Exiting.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Hit detector
hit_detector = HitDetector()
def hit_callback(x, y, r, balloon):
    balloon.pop()

# Game loop
frame_time = 1.0 / target_fps
while True:
    start_time = time.time()
    game_frame = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255
    game_frame = cv2.flip(game_frame, 1)
    ret, camera_frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    camera_frame = cv2.flip(camera_frame, 1)  # Consistent with calibration
    # camera_frame = cv2.warpPerspective(camera_frame, transform_matrix, (screen_width, screen_height))

    for balloon in balloons:
        balloon.update()
        balloon.move()
        balloon.draw(game_frame, show_bbox=False)

    circles, _ = detect_circles_in_frame(
        camera_frame,
        min_radius=30,
        max_radius=50,
        min_dist=80,
        param1=120,
        param2=20,
        intensity_threshold=100
    )

    # Visualize and process detected circles
    debug_visualization = True
    if circles is not None and isinstance(circles, np.ndarray) and circles.size > 0:
        circles = np.round(circles[0, :]).astype("int")
        if circles.ndim == 1:  # Single circle case
            circles = circles.reshape(1, -1)
        for (x, y, r) in circles:
            if debug_visualization:
                cv2.circle(game_frame, (x, y), r, (0, 255, 255), 3)  # Yellow outline
                cv2.circle(game_frame, (x, y), 2, (0, 0, 255), 3)    # Red center
            hit_detector.process_detected_circles(circles, balloons, callback=hit_callback)

    cv2.imshow("Game Display", game_frame)
    cv2.imshow("Camera Feed", camera_frame)

    elapsed = time.time() - start_time
    sleep_time = max(1, int((frame_time - elapsed) * 1000))
    key = cv2.waitKey(sleep_time) & 0xFF
    if key in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()