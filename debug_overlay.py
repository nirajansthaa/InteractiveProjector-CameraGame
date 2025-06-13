import pygame
import cv2
import numpy as np
import sys
import json
import os
import time
import logging

# Suppress Ultralytics logging
os.environ["YOLO_VERBOSE"] = "False"
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

# Suppress OpenCV logging
cv2.setLogLevel(0)

from ultralytics import YOLO
from screeninfo import get_monitors

# Constants
SCREEN_WIDTH = 1360
SCREEN_HEIGHT = 768
FPS = 60
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.7
CLICK_COOLDOWN = 0.5
MODEL_PATH = "best.onnx"
CRACK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "boom1.png")
CALIBRATION_FILE = "calibration.json"

# Initialize Pygame
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
pygame.init()
monitors = get_monitors()
if len(monitors) < 2:
    print("Error: External monitor (projector) not detected")
    sys.exit(1)
external_screen = monitors[1]
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{external_screen.x},{external_screen.y}"
screen = pygame.display.set_mode((external_screen.width, external_screen.height))
pygame.display.set_caption("Green Ball Crack Effect")
clock = pygame.time.Clock()

# Load crack image
try:
    crack_img = pygame.image.load(CRACK_PATH).convert_alpha()
    crack_img = pygame.transform.scale(crack_img, (260, 150))
except pygame.error as e:
    print(f"Error loading boom1.png: {e}")
    sys.exit(1)

# Initialize YOLO model with suppressed output
try:
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    model = YOLO(MODEL_PATH, task="detect", verbose=False)
    sys.stdout = original_stdout
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    sys.exit(1)

# Initialize camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera")
    pygame.quit()
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Calibration functions
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

def get_calibration_points(cap, window_name="Calibration"):
    points = []
    selected_point = -1
    dragging = False
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, selected_point, dragging
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
    print("=== CALIBRATION INSTRUCTIONS ===")
    print("1. Click EXACTLY on the four corners of the projector screen in the camera view")
    print("2. Use the grid to align points precisely")
    print("3. Drag points to adjust")
    print("4. Points will be ordered: top-left, top-right, bottom-right, bottom-left")
    print("5. Press 'r' to reset, 'c' to confirm, 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame during calibration")
            return None
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        for i in range(0, width, 50):
            cv2.line(display_frame, (i, 0), (i, height), (50, 50, 50), 1)
        for i in range(0, height, 50):
            cv2.line(display_frame, (0, i), (width, i), (50, 50, 50), 1)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        labels = ["TL", "TR", "BR", "BL"]
        for i, pt in enumerate(points):
            color = colors[i % 4]
            cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 8, color, -1)
            cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 12, (255, 255, 255), 2)
            cv2.putText(display_frame, labels[i], (int(pt[0]) - 8, int(pt[1]) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if len(points) >= 2:
            ordered_points = order_points(points) if len(points) == 4 else points
            for i in range(len(ordered_points)):
                if i < len(ordered_points) - 1:
                    cv2.line(display_frame, (int(ordered_points[i][0]), int(ordered_points[i][1])),
                             (int(ordered_points[i+1][0]), int(ordered_points[i+1][1])), (0, 255, 255), 2)
            if len(points) == 4:
                cv2.line(display_frame, (int(ordered_points[-1][0]), int(ordered_points[-1][1])),
                         (int(ordered_points[0][0]), int(ordered_points[0][1])), (0, 255, 255), 2)
        status_text = f"Points: {len(points)}/4"
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if len(points) == 4:
            is_valid, message = validate_quadrilateral(points)
            color = (0, 255, 0) if is_valid else (0, 0, 255)
            cv2.putText(display_frame, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if is_valid:
                cv2.putText(display_frame, "Press 'c' to confirm", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return None
        elif key == ord('r'):
            points = []
            print("Points reset")
        elif key == ord('c'):
            if len(points) == 4:
                is_valid, message = validate_quadrilateral(points)
                if is_valid:
                    cv2.destroyWindow(window_name)
                    return order_points(points).tolist()
                else:
                    print(f"Cannot confirm: {message}")

def save_calibration_points(points, offset_x=0, offset_y=0, debug_offset_x=0, debug_offset_y=0, filename=CALIBRATION_FILE):
    try:
        calibration_data = {
            'points': points,
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
            return data, 0, 0, 0, 0
        if isinstance(data, dict) and 'points' in data:
            if data.get('screen_width') != SCREEN_WIDTH or data.get('screen_height') != SCREEN_HEIGHT:
                print("Warning: Calibration was done for different screen dimensions")
                return None, 0, 0, 0, 0
            offset_x = data.get('offset_x', 0)
            offset_y = data.get('offset_y', 0)
            debug_offset_x = data.get('debug_offset_x', 0)
            debug_offset_y = data.get('debug_offset_y', 0)
            return data['points'], offset_x, offset_y, debug_offset_x, debug_offset_y
        return None, 0, 0, 0, 0
    except Exception as e:
        print(f"Error loading calibration points: {e}")
        return None, 0, 0, 0, 0

def get_perspective_transform(src_points, offset_x=0, offset_y=0):
    dst_points = np.float32([
        [0 + offset_x, 0 + offset_y],
        [SCREEN_WIDTH-1 + offset_x, 0 + offset_y],
        [SCREEN_WIDTH-1 + offset_x, SCREEN_HEIGHT-1 + offset_y],
        [0 + offset_x, SCREEN_HEIGHT-1 + offset_y]
    ])
    src_points = np.float32(src_points)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    print("=== CALIBRATION QUALITY ===")
    for i, (src, dst) in enumerate(zip(src_points, dst_points)):
        transformed = cv2.perspectiveTransform(np.array([[src]], dtype=np.float32), matrix)[0][0]
        error = np.linalg.norm(transformed - dst)
        print(f"Corner {i+1}: Error = {error:.2f} pixels")
    return matrix

def test_calibration_accuracy(transform_matrix, calibration_points):
    print("\n=== TESTING CALIBRATION ACCURACY ===")
    src_points = np.float32(calibration_points).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(src_points, transform_matrix)
    expected_points = np.float32([[0, 0], [SCREEN_WIDTH-1, 0], [SCREEN_WIDTH-1, SCREEN_HEIGHT-1], [0, SCREEN_HEIGHT-1]])
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

# Load or perform calibration
calibration_points, offset_x, offset_y, debug_offset_x, debug_offset_y = load_calibration_points()
transform_matrix = None
if calibration_points and len(calibration_points) == 4:
    print(f"Loading existing calibration with homography offset ({offset_x}, {offset_y}) and debug offset ({debug_offset_x}, {debug_offset_y})...")
    transform_matrix = get_perspective_transform(calibration_points, offset_x, offset_y)
    test_calibration_accuracy(transform_matrix, calibration_points)
else:
    print("Performing camera calibration...")
    calibration_points = get_calibration_points(cap)
    offset_x, offset_y = 0, 0
    debug_offset_x, debug_offset_y = 0, 0
    if calibration_points and len(calibration_points) == 4:
        save_calibration_points(calibration_points, offset_x, offset_y, debug_offset_x, debug_offset_y)
        transform_matrix = get_perspective_transform(calibration_points, offset_x, offset_y)
        test_calibration_accuracy(transform_matrix, calibration_points)
    else:
        print("Error: Calibration failed")
        cap.release()
        pygame.quit()
        sys.exit(1)

# Crack effect class
class CrackEffect:
    def __init__(self, x, y, duration=0.3):
        self.x = x
        self.y = y
        self.start_time = time.time()
        self.duration = duration

    def draw(self, win):
        if time.time() - self.start_time < self.duration:
            win.blit(crack_img, (self.x, self.y))
            return True
        return False

# Main loop variables
cracks = []
last_click_time = 0
running = True
show_debug_overlay = False
font = pygame.font.SysFont(None, 36)

# Main loop
while running:
    clock.tick(FPS)
    
    # Read camera frame
    ret, frame = cap.read()
    if not ret:
        print("Warning: Could not read frame")
        continue
    
    # Apply perspective transform
    warped_frame = cv2.warpPerspective(frame, transform_matrix, (SCREEN_WIDTH, SCREEN_HEIGHT))
    
    # Run YOLO detection
    results = model.predict(warped_frame, imgsz=640, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device="cpu", verbose=False)
    
    # Process detections
    current_time = time.time()
    for result in results:
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if (0 <= cx <= SCREEN_WIDTH and 0 <= cy <= SCREEN_HEIGHT and 
                    current_time - last_click_time >= CLICK_COOLDOWN):
                    crack_x = int(cx - crack_img.get_width() / 2)
                    crack_y = int(cy - crack_img.get_height() / 2)
                    cracks.append(CrackEffect(crack_x, crack_y))
                    last_click_time = current_time
    
    # Debug view
    debug_view = warped_frame.copy()
    for result in results:
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                confidence = float(box.conf[0])
                cv2.rectangle(debug_view, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(debug_view, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.putText(debug_view, f"Green Ball: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.putText(debug_view, f"Screen: {SCREEN_WIDTH}x{SCREEN_HEIGHT}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_view, f"Homography Offset: ({offset_x}, {offset_y})", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_view, f"Debug Offset: ({debug_offset_x}, {debug_offset_y})", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Camera Feed", debug_view)
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_c:
                print("Starting recalibration...")
                calibration_points = get_calibration_points(cap)
                if calibration_points and len(calibration_points) == 4:
                    offset_x, offset_y = 0, 0
                    debug_offset_x, debug_offset_y = 0, 0
                    save_calibration_points(calibration_points, offset_x, offset_y, debug_offset_x, debug_offset_y)
                    transform_matrix = get_perspective_transform(calibration_points, offset_x, offset_y)
                    test_calibration_accuracy(transform_matrix, calibration_points)
            elif event.key == pygame.K_f:
                show_debug_overlay = not show_debug_overlay
                print(f"Debug overlay: {'ON' if show_debug_overlay else 'OFF'}")
            elif event.key == pygame.K_LEFT:
                offset_x -= 5
                transform_matrix = get_perspective_transform(calibration_points, offset_x, offset_y)
                print(f"Adjusted homography offset to ({offset_x}, {offset_y})")
            elif event.key == pygame.K_RIGHT:
                offset_x += 5
                transform_matrix = get_perspective_transform(calibration_points, offset_x, offset_y)
                print(f"Adjusted homography offset to ({offset_x}, {offset_y})")
            elif event.key == pygame.K_UP:
                offset_y -= 5
                transform_matrix = get_perspective_transform(calibration_points, offset_x, offset_y)
                print(f"Adjusted homography offset to ({offset_x}, {offset_y})")
            elif event.key == pygame.K_DOWN:
                offset_y += 5
                transform_matrix = get_perspective_transform(calibration_points, offset_x, offset_y)
                print(f"Adjusted homography offset to ({offset_x}, {offset_y})")
            elif event.key == pygame.K_w:
                debug_offset_y -= 5
                print(f"Adjusted debug offset to ({debug_offset_x}, {debug_offset_y})")
            elif event.key == pygame.K_s:
                debug_offset_y += 5
                print(f"Adjusted debug offset to ({debug_offset_x}, {debug_offset_y})")
            elif event.key == pygame.K_a:
                debug_offset_x -= 5
                print(f"Adjusted debug offset to ({debug_offset_x}, {debug_offset_y})")
            elif event.key == pygame.K_d:
                debug_offset_x += 5
                print(f"Adjusted debug offset to ({debug_offset_x}, {debug_offset_y})")
            elif event.key == pygame.K_p:
                save_calibration_points(calibration_points, offset_x, offset_y, debug_offset_x, debug_offset_y)
    
    # Render screen
    screen.fill((0, 0, 0))
    for crack in cracks:
        crack.draw(screen)
    cracks = [c for c in cracks if c.draw(screen)]
    if show_debug_overlay:
        # Draw detected points with debug offset
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) / 2 + debug_offset_x
                    cy = (y1 + y2) / 2 + debug_offset_y
                    pygame.draw.circle(screen, (255, 0, 0), (int(cx), int(cy)), 5)
                    pygame.draw.rect(screen, (0, 255, 0),
                                     (x1 + debug_offset_x, y1 + debug_offset_y,
                                      x2 - x1, y2 - y1), 2)
        # Draw calibration corners with debug offset
        dst_points = np.float32([[0, 0], [SCREEN_WIDTH-1, 0], [SCREEN_WIDTH-1, SCREEN_HEIGHT-1], [0, SCREEN_HEIGHT-1]])
        for i, pt in enumerate(dst_points):
            pygame.draw.circle(screen, (0, 255, 255),
                               (int(pt[0] + debug_offset_x), int(pt[1] + debug_offset_y)), 10)
            pygame.draw.circle(screen, (255, 255, 255),
                               (int(pt[0] + debug_offset_x), int(pt[1] + debug_offset_y)), 10, 2)
    # Display offsets
    offset_text = font.render(f"Homography Offset: ({offset_x}, {offset_y})", True, (255, 255, 255))
    debug_offset_text = font.render(f"Debug Offset: ({debug_offset_x}, {debug_offset_y})", True, (255, 255, 255))
    screen.blit(offset_text, (10, 10))
    screen.blit(debug_offset_text, (10, 40))
    pygame.display.flip()
    
    # Check for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()