import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.projector_display import show_calibration_pattern
from modules.camera_capture import capture_camera_points
from modules.calibrate import compute_homography, test_homography

projector_points = show_calibration_pattern()
camera_points = capture_camera_points()

H = compute_homography(camera_points, projector_points)
print("Homography Matrix:\n", H)

# Test point
test_cam_point = camera_points[0]
mapped = test_homography(H, test_cam_point)
print(f"Camera Point: {test_cam_point} → Projector Point: {mapped}")
