import cv2
import numpy as np
import time
import json
import pygame

# Constants
SCREEN_WIDTH = 1360
SCREEN_HEIGHT = 768
CALIBRATION_FILE = "calibration.json"
PROJECTOR_PATTERN_SIZE = (260, 150)  # Size for projector pattern (circle size)

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top_points = sorted_by_y[:2]
    bottom_points = sorted_by_y[2:]
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    bottom_right, bottom_left = bottom_points[np.argsort(bottom_points[:, 0])[::-1]]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def validate_quadrilateral(points):
    if len(points) != 4:
        return False, "Need exactly 4 points"
    min_distance = 50
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if calculate_distance(points[i], points[j]) < min_distance:
                return False, f"Points {i+1} and {j+1} are too close together"
    ordered_points = order_points(points)
    top_width = calculate_distance(ordered_points[0], ordered_points[1])
    bottom_width = calculate_distance(ordered_points[3], ordered_points[2])
    left_height = calculate_distance(ordered_points[0], ordered_points[3])
    right_height = calculate_distance(ordered_points[1], ordered_points[2])
    avg_width = (top_width + bottom_width) / 2
    avg_height = (left_height + right_height) / 2
    if avg_width == 0 or avg_height == 0:
        return False, "Invalid dimensions"
    aspect_ratio = avg_width / avg_height
    expected_ratio = SCREEN_WIDTH / SCREEN_HEIGHT
    if not (expected_ratio * 0.7 <= aspect_ratio <= expected_ratio * 1.3):
        return False, f"Aspect ratio {aspect_ratio:.2f} doesn't match expected {expected_ratio:.2f}"
    return True, "Valid quadrilateral"

def get_calibration_points(cap, screen, window_name="Calibration"):
    points_camera = []
    points_projector = []
    selected_point = -1
    dragging = False
    calibration_phase = "camera"  # Track whether calibrating camera or projector

    def mouse_callback(event, x, y, flags, param):
        nonlocal points_camera, points_projector, selected_point, dragging, calibration_phase
        points = points_camera if calibration_phase == "camera" else points_projector
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, pt in enumerate(points):
                if calculate_distance([x, y], pt) < 15:
                    selected_point = i
                    dragging = True
                    return
            if len(points) < 4:
                points.append([x, y])
                selected_point = len(points) - 1
                print(f"Point {len(points)} selected: ({x}, {y})")
        elif event == cv2.EVENT_MOUSEMOVE and dragging and selected_point >= 0:
            points[selected_point] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            selected_point = -1

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Camera calibration instructions
    print("=== CAMERA CALIBRATION INSTRUCTIONS ===")
    print("1. Click EXACTLY on the four corners of the projector screen in the camera view")
    print("2. Use the grid to align points precisely")
    print("3. Drag points to adjust")
    print("4. Points will be ordered: top-left, top-right, bottom-right, bottom-left")
    print("5. Press 'r' to reset, 'c' to confirm, 'q' to quit")

    while calibration_phase == "camera":
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame during camera calibration")
            return None, None
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        for i in range(0, width, 50):
            cv2.line(display_frame, (i, 0), (i, height), (50, 50, 50), 1)
        for i in range(0, height, 50):
            cv2.line(display_frame, (0, i), (width, i), (50, 50, 50), 1)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        labels = ["TL", "TR", "BR", "BL"]
        for i, pt in enumerate(points_camera):
            color = colors[i % 4]
            cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 8, color, -1)
            cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 12, (255, 255, 255), 2)
            cv2.putText(display_frame, labels[i], (int(pt[0]) - 8, int(pt[1]) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if len(points_camera) >= 2:
            ordered_points = order_points(points_camera) if len(points_camera) == 4 else points_camera
            for i in range(len(ordered_points)):
                if i < len(ordered_points) - 1:
                    cv2.line(display_frame, (int(ordered_points[i][0]), int(ordered_points[i][1])),
                             (int(ordered_points[i+1][0]), int(ordered_points[i+1][1])), (0, 255, 255), 2)
            if len(points_camera) == 4:
                cv2.line(display_frame, (int(ordered_points[-1][0]), int(ordered_points[-1][1])),
                         (int(ordered_points[0][0]), int(ordered_points[0][1])), (0, 255, 255), 2)
        status_text = f"Camera Points: {len(points_camera)}/4"
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if len(points_camera) == 4:
            is_valid, message = validate_quadrilateral(points_camera)
            color = (0, 255, 0) if is_valid else (0, 0, 255)
            cv2.putText(display_frame, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if is_valid:
                cv2.putText(display_frame, "Press 'c' to confirm camera points", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return None, None
        elif key == ord('r'):
            points_camera = []
            print("Camera points reset")
        elif key == ord('c') and len(points_camera) == 4:
            is_valid, message = validate_quadrilateral(points_camera)
            if is_valid:
                calibration_phase = "projector"
                print("Camera calibration complete. Starting projector calibration...")

    # Projector calibration: Display pattern and capture points
    marge_x, marge_y = 400, 300
    projector_points = [
        [marge_x, marge_y],
        [SCREEN_WIDTH - marge_x, marge_y],
        [SCREEN_WIDTH - marge_x, SCREEN_HEIGHT - marge_y],
        [marge_x, SCREEN_HEIGHT - marge_y]
    ]
    calibration_pattern = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), np.uint8)
    for i, p1 in enumerate(projector_points):
        p1 = tuple(map(int, p1))
        for p2 in projector_points[i+1:]:
            p2 = tuple(map(int, p2))
            cv2.line(calibration_pattern, p1, p2, (255, 255, 255), 3)
        cv2.circle(calibration_pattern, p1, 40, (255, 255, 255), -1)
    pattern_surface = pygame.surfarray.make_surface(calibration_pattern.swapaxes(0, 1))
    screen.blit(pattern_surface, (0, 0))
    pygame.display.flip()
    time.sleep(0.08)  # Ensure pattern is visible

    print("=== PROJECTOR CALIBRATION INSTRUCTIONS ===")
    print("1. Click EXACTLY on the four white circles projected on the screen in the camera view")
    print("2. Use the grid to align points precisely")
    print("3. Drag points to adjust")
    print("4. Points will be ordered: top-left, top-right, bottom-right, bottom-left")
    print("5. Press 'r' to reset, 'c' to confirm, 'q' to quit")

    while calibration_phase == "projector":
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame during projector calibration")
            return None, None
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        for i in range(0, width, 50):
            cv2.line(display_frame, (i, 0), (i, height), (50, 50, 50), 1)
        for i in range(0, height, 50):
            cv2.line(display_frame, (0, i), (width, i), (50, 50, 50), 1)
        for i, pt in enumerate(points_projector):
            color = colors[i % 4]
            cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 8, color, -1)
            cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 12, (255, 255, 255), 2)
            cv2.putText(display_frame, labels[i], (int(pt[0]) - 8, int(pt[1]) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if len(points_projector) >= 2:
            ordered_points = order_points(points_projector) if len(points_projector) == 4 else points_projector
            for i in range(len(ordered_points)):
                if i < len(ordered_points) - 1:
                    cv2.line(display_frame, (int(ordered_points[i][0]), int(ordered_points[i][1])),
                             (int(ordered_points[i+1][0]), int(ordered_points[i+1][1])), (0, 255, 255), 2)
            if len(points_projector) == 4:
                cv2.line(display_frame, (int(ordered_points[-1][0]), int(ordered_points[-1][1])),
                         (int(ordered_points[0][0]), int(ordered_points[0][1])), (0, 255, 255), 2)
        status_text = f"Projector Points: {len(points_projector)}/4"
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if len(points_projector) == 4:
            is_valid, message = validate_quadrilateral(points_projector)
            color = (0, 255, 0) if is_valid else (0, 0, 255)
            cv2.putText(display_frame, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if is_valid:
                cv2.putText(display_frame, "Press 'c' to confirm projector points", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return None, None
        elif key == ord('r'):
            points_projector = []
            print("Projector points reset")
        elif key == ord('c') and len(points_projector) == 4:
            is_valid, message = validate_quadrilateral(points_projector)
            if is_valid:
                cv2.destroyWindow(window_name)
                return order_points(points_camera).tolist(), order_points(points_projector).tolist()
    return None, None

def save_calibration_points(points_camera, points_projector, offset_x=0, offset_y=0, debug_offset_x=0, debug_offset_y=0, filename=CALIBRATION_FILE):
    try:
        calibration_data = {
            'points_camera': points_camera,
            'points_projector': points_projector,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'debug_offset_x': debug_offset_x,
            'debug_offset_y': debug_offset_y,
            'screen_width': SCREEN_WIDTH,
            'screen_height': SCREEN_HEIGHT,
            'timestamp': time.time(),
            'version': '2.2'
        }
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"Calibration saved to {filename} with homography offset ({offset_x}, {offset_y}) and debug offset ({debug_offset_x}, {debug_offset_y})")
    except Exception as e:
        print(f"Error saving calibration points: {e}")

def load_calibration_points(filename=CALIBRATION_FILE):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data, None, 0, 0, 0, 0  # Backward compatibility
        if isinstance(data, dict) and 'points_camera' in data and 'points_projector' in data:
            if data.get('screen_width') != SCREEN_WIDTH or data.get('screen_height') != SCREEN_HEIGHT:
                print("Warning: Calibration was done for different screen dimensions")
                return None, None, 0, 0, 0, 0
            offset_x = data.get('offset_x', 0)
            offset_y = data.get('offset_y', 0)
            debug_offset_x = data.get('debug_offset_x', 0)
            debug_offset_y = data.get('debug_offset_y', 0)
            return data['points_camera'], data['points_projector'], offset_x, offset_y, debug_offset_x, debug_offset_y
        return None, None, 0, 0, 0, 0
    except Exception as e:
        print(f"Error loading calibration points: {e}")
        return None, None, 0, 0, 0, 0

def get_perspective_transforms(points_camera, points_projector):
    screen_points = np.float32([
        [0, 0],
        [SCREEN_WIDTH - 1, 0],
        [SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1],
        [0, SCREEN_HEIGHT - 1]
    ])
    points_camera = np.float32(points_camera)
    points_projector = np.float32(points_projector)
    m_camera2screen = cv2.getPerspectiveTransform(points_camera, screen_points)
    m_projector2camera = cv2.getPerspectiveTransform(points_projector, points_camera)
    print("=== CAMERA-TO-SCREEN CALIBRATION QUALITY ===")
    for i, (src, dst) in enumerate(zip(points_camera, screen_points)):
        transformed = cv2.perspectiveTransform(np.array([[src]], dtype=np.float32), m_camera2screen)[0][0]
        error = np.linalg.norm(transformed - dst)
        print(f"Corner {i+1}: Error = {error:.2f} pixels")
    print("=== PROJECTOR-TO-CAMERA CALIBRATION QUALITY ===")
    for i, (src, dst) in enumerate(zip(points_projector, points_camera)):
        transformed = cv2.perspectiveTransform(np.array([[src]], dtype=np.float32), m_projector2camera)[0][0]
        error = np.linalg.norm(transformed - dst)
        print(f"Corner {i+1}: Error = {error:.2f} pixels")
    return m_camera2screen, m_projector2camera

def test_calibration_accuracy(transform_matrix, points, expected_points, transform_name="Calibration"):
    print(f"\n=== TESTING {transform_name.upper()} ACCURACY ===")
    src_points = np.float32(points).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(src_points, transform_matrix)
    expected_points = np.float32(expected_points).reshape(-1, 2)
    total_error = 0
    for i, (expected, actual) in enumerate(zip(expected_points, transformed_points.reshape(-1, 2))):
        error = np.linalg.norm(actual - expected)
        total_error += error
        print(f"Corner {i+1}: Expected {expected}, Got {actual}, Error: {error:.2f}px")
    avg_error = total_error / 4
    print(f"Average error: {avg_error:.2f} pixels")
    if avg_error < 5: print("✓ Excellent calibration accuracy")
    elif avg_error < 15: print("✓ Good calibration accuracy")
    elif avg_error < 30: print("⚠ Moderate calibration accuracy - consider recalibrating")
    else: print("✗ Poor calibration accuracy - recalibration recommended")
    return avg_error