import pygame
import pyautogui
import cv2
import numpy as np
import sys
import json
import os
import time
import logging
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

# Load or perform calibration
calibration_points, offset_x, offset_y, debug_offset_x, debug_offset_y = load_calibration_points()
transform_matrix = None
if calibration_points and len(calibration_points) == 4:
    print(f"Loading existing calibration with homography offset ({offset_x}, {offset_y}) and debug offset ({debug_offset_x}, {debug_offset_y})...")
    transform_matrix = get_perspective_transform(calibration_points, 0, 0)
else:
    print("Performing camera calibration...")
    calibration_points = get_calibration_points(cap)
    offset_x, offset_y = 0, 0  # Set homography offset to 0,0
    debug_offset_x, debug_offset_y = 0, 0  
    if calibration_points and len(calibration_points) == 4:
        save_calibration_points(calibration_points, offset_x, offset_y, debug_offset_x, debug_offset_y)
        transform_matrix = get_perspective_transform(calibration_points, 0, 0)
    else:
        print("Error: Calibration failed")
        cap.release()
        pygame.quit()
        sys.exit()

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

# Compute inverse transform for manual clicks
inv_transform_matrix = np.linalg.inv(transform_matrix)

# Main loop
while running:
    clock.tick(FPS)
    
    # Read camera frame
    ret, frame = cap.read()
    if not ret:
        print("Warning: Could not read frame")
        continue
    
    # Apply perspective transform
    warped_frame = cv2.warpPerspective(frame, transform_matrix, (external_screen.width, external_screen.height))
    
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
                if (0 <= cx <= external_screen.width and 0 <= cy <= external_screen.height and 
                    current_time - last_click_time >= CLICK_COOLDOWN):
                    screen_x = int(cx + debug_offset_x + external_screen.x)
                    screen_y = int(cy + debug_offset_y + external_screen.y)

                    # Move the mouse and click
                    pyautogui.moveTo(screen_x, screen_y)
                    pyautogui.click(button='left')

                    # Still show the visual crack effect for feedback
                    crack_x = int(cx - crack_img.get_width() / 2 + debug_offset_x)
                    crack_y = int(cy - crack_img.get_height() / 2 + debug_offset_y)
                    cracks.append(CrackEffect(crack_x, crack_y))
                    last_click_time = current_time
    
    # Debug view
    debug_view = warped_frame.copy()
    debug_view = cv2.resize(warped_frame, (SCREEN_WIDTH, SCREEN_HEIGHT))

    # Draw ROI boundary
    roi_points = np.float32([[0, 0], [external_screen.width-1, 0], [external_screen.width-1, external_screen.height-1], [0, external_screen.height-1]])
    roi_points = roi_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(debug_view, [roi_points], True, (255, 0, 0), 2)
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

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
            if current_time - last_click_time >= CLICK_COOLDOWN:
                mx, my = event.pos
                point = np.float32([[[mx, my]]])
                warped_point = cv2.perspectiveTransform(point, inv_transform_matrix)[0][0]
                cx, cy = warped_point
                if 0 <= cx <= external_screen.width and 0 <= cy <= external_screen.height:
                    crack_x = int(mx - crack_img.get_width() / 2)
                    crack_y = int(my - crack_img.get_height() / 2)
                    cracks.append(CrackEffect(crack_x, crack_y))
                    last_click_time = current_time
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
                    pygame.draw.rect(screen, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)
    
    pygame.display.flip()
    
    # Check for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()