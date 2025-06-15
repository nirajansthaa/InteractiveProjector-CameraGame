import pygame
import cv2
import numpy as np
import sys
import json
import os
import time
import logging
from collections import deque
from modules.calibration import *

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
MODEL_PATH = "best1.onnx"
CRACK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "boom1.png")
CALIBRATION_FILE = "calibration.json"
HIT_ZONE_Y = SCREEN_HEIGHT * 0.9  # Bottom 10% of screen as hit zone
VELOCITY_THRESHOLD = 50  # Pixels per second for velocity reversal
TRACKING_FRAMES = 5  # Number of frames to track ball position

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
    offset_x, offset_y = 280, 125
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
ball_positions = deque(maxlen=TRACKING_FRAMES)  # Store (x, y, timestamp) for tracking
last_hit_time = 0  # Track last hit to debounce
hit_debounce = 1.0  # Seconds to wait before allowing another hit

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
    detected_ball = False
    for result in results:
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if 0 <= cx <= SCREEN_WIDTH and 0 <= cy <= SCREEN_HEIGHT:
                    detected_ball = True
                    # Store position and timestamp
                    ball_positions.append((cx, cy, current_time))
                    # Check for hit condition
                    if (len(ball_positions) >= TRACKING_FRAMES and 
                        cy > HIT_ZONE_Y and 
                        current_time - last_hit_time >= hit_debounce):
                        # Calculate vertical velocity (y difference over time)
                        prev_pos = ball_positions[-TRACKING_FRAMES]
                        dy = cy - prev_pos[1]
                        dt = current_time - prev_pos[2]
                        if dt > 0:
                            velocity_y = dy / dt  # Pixels per second
                            # Check for velocity reversal (downward to upward)
                            if velocity_y < -VELOCITY_THRESHOLD:  # Negative indicates upward
                                # Verify previous velocity was downward
                                mid_pos = ball_positions[-TRACKING_FRAMES//2]
                                mid_dy = mid_pos[1] - prev_pos[1]
                                mid_dt = mid_pos[2] - prev_pos[2]
                                if mid_dt > 0 and mid_dy / mid_dt > 0:  # Downward
                                    crack_x = int(cx - crack_img.get_width() / 2)
                                    crack_y = int(cy - crack_img.get_height() / 2)
                                    cracks.append(CrackEffect(crack_x, crack_y))
                                    last_hit_time = current_time
                                    print(f"Hit detected at ({cx:.1f}, {cy:.1f}) with velocity {velocity_y:.1f} px/s")
                    break  # Process only the first detected ball
    
    # Clear positions if no ball detected to avoid stale data
    if not detected_ball:
        ball_positions.clear()
    
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
    
    # Draw hit zone in debug view
    cv2.line(debug_view, (0, int(HIT_ZONE_Y)), (SCREEN_WIDTH, int(HIT_ZONE_Y)), (255, 0, 0), 2)
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
            elif event.key == pygame.K_d:
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
        dst_points = np.float32([[0, 0], [SCREEN_WIDTH-1, 0], [SCREEN_WIDTH-1, SCREEN_HEIGHT-1], [0, SCREEN_HEIGHT-1]])
        for i, pt in enumerate(dst_points):
            pygame.draw.circle(screen, (0, 255, 255),
                               (int(pt[0] + debug_offset_x), int(pt[1] + debug_offset_y)), 10)
            pygame.draw.circle(screen, (255, 255, 255),
                               (int(pt[0] + debug_offset_x), int(pt[1] + debug_offset_y)), 10, 2)
    offset_text = font.render(f"Homography Offset: ({offset_x}, {offset_y})", True, (255, 255, 255))
    debug_offset_text = font.render(f"Debug Offset: ({debug_offset_x}, {debug_offset_y})", True, (255, 255, 255))
    screen.blit(offset_text, (10, 10))
    screen.blit(debug_offset_text, (10, 40))
    pygame.display.flip()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()