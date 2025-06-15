import pygame
import cv2
import numpy as np
import random
import sys
import json
import os
import time
import logging
from screeninfo import get_monitors
from modules.background import draw_text_with_bg
from modules.edge_detection import EdgeDetector
from modules.calibration import *

# Suppress Ultralytics logging
os.environ["YOLO_VERBOSE"] = "False"
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

# Suppress OpenCV logging
cv2.setLogLevel(0)

from ultralytics import YOLO

# Constants
SCREEN_WIDTH = 1360
SCREEN_HEIGHT = 768
FPS = 60
GAME_DURATION = 120  # Seconds
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.7
MODEL_PATH = "best1.onnx"
CALIBRATION_FILE = "calibration.json"

# Get monitor info for projector display
monitors = get_monitors()
if not monitors:
    print("Error: No monitors detected")
    sys.exit(1)

print("Detected monitors:")
for i, monitor in enumerate(monitors):
    print(f"Monitor {i}: {monitor.width}x{monitor.height} at ({monitor.x}, {monitor.y})")
main_screen = monitors[0]
external_screen = monitors[1] if len(monitors) > 1 else monitors[0]
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{external_screen.x},{external_screen.y}"

# Initialize Pygame
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
pygame.init()
WIN = pygame.display.set_mode((external_screen.width, external_screen.height))
pygame.display.set_caption("Balloon Pop Game")
CLOCK = pygame.time.Clock()
pygame.mouse.set_visible(True)

actual_width, actual_height = WIN.get_size()

# Load YOLO model
try:
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    model = YOLO(MODEL_PATH, task="detect", verbose=False)
    sys.stdout = original_stdout
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    pygame.quit()
    sys.exit(1)

# Load assets
base_dir = os.path.dirname(os.path.abspath(__file__))
BALLOON_IMAGES = []
balloon_files = ["balloon1.png", "balloon2.png", "balloon3.png", "balloon4.png"]
for file in balloon_files:
    path = os.path.join(base_dir, "assets", file)
    if not os.path.exists(path):
        print(f"Warning: Balloon image {path} not found. Skipping.")
        continue
    try:
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.scale(img, (320, 360))
        BALLOON_IMAGES.append(img)
    except pygame.error as e:
        print(f"Warning: Failed to load {path}: {e}. Skipping.")
if len(BALLOON_IMAGES) < 4:
    print(f"Error: Only {len(BALLOON_IMAGES)} balloon images loaded, expected 4. Exiting.")
    pygame.quit()
    sys.exit(1)

try:
    pop_sound = pygame.mixer.Sound(os.path.join(base_dir, "assets", "pop.wav"))
except pygame.error as e:
    print(f"Warning: Failed to load pop.wav: {e}. No pop sound.")
    pop_sound = None

try:
    boom_path = os.path.join(base_dir, "assets", "boom.png")
    boom_img = pygame.image.load(boom_path).convert_alpha()
    boom_img = pygame.transform.scale(boom_img, (260, 150))
except pygame.error as e:
    print(f"Warning: Failed to load boom.png: {e}. No boom image.")
    boom_img = None

# Colors and fonts
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
FONT = pygame.font.SysFont("arial", 32)
BIG_FONT = pygame.font.SysFont("arial", 48)

class Balloon:
    def __init__(self, image):
        self.image = image  # Use specific image passed to constructor
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.popped = False
        self.pop_time = None
        # Initialize position attributes without calling reset
        self.x = 0
        self.y = 0
        self.speed = 0

    def reset(self):
        max_attempts = 10  # Prevent infinite loops
        for _ in range(max_attempts):
            proposed_x = random.randint(0, actual_width - self.width)
            proposed_y = actual_height + random.randint(0, 300)
            overlap = False
            for other in balloons:
                if other is not self and not other.popped:
                    distance_x = abs(proposed_x - other.x)
                    distance_y = abs(proposed_y - other.y)
                    min_distance_x = (self.width + other.width) // 2
                    min_distance_y = (self.height + other.height) // 2
                    if distance_x < min_distance_x and distance_y < min_distance_y:
                        overlap = True
                        break
            if not overlap:
                self.x = proposed_x
                self.y = proposed_y
                self.speed = random.uniform(3.75, 6.0)
                self.popped = False
                return
        # Fallback to random position if no non-overlapping spot found
        self.x = random.randint(0, actual_width - self.width)
        self.y = actual_height + random.randint(0, 300)
        self.speed = random.uniform(3.75, 6.0)
        self.popped = False

    def update(self):
        if self.popped and self.pop_time and time.time() - self.pop_time >= 0.3:
            self.reset()
        elif not self.popped:
            self.y -= self.speed
            if self.y + self.height < 0:
                self.reset()

    def draw(self, win):
        if not self.popped:
            win.blit(self.image, (self.x, self.y))

    def check_object_occlusion(self, object_mask, detected_boxes, detector, warped_frame):
        if self.popped or object_mask is None:
            return False
        x1, y1 = int(self.x), int(self.y)
        x2, y2 = x1 + self.width, y1 + self.height
        if x1 < 0 or y1 < 0 or x2 > SCREEN_WIDTH or y2 > SCREEN_HEIGHT:
            return False
        balloon_crop = warped_frame[y1:y2, x1:x2]
        if balloon_crop.size == 0:
            return False
        gray_crop = cv2.cvtColor(balloon_crop, cv2.COLOR_BGR2GRAY)
        balloon_mask = detector.detect_edges_from_array(gray_crop)
        object_crop = object_mask[y1:y2, x1:x2]
        if object_crop.shape != balloon_mask.shape:
            object_crop = cv2.resize(object_crop, (balloon_mask.shape[1], balloon_mask.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        intersection = cv2.bitwise_and(object_crop, balloon_mask)
        overlap = np.count_nonzero(intersection)
        balloon_area = np.count_nonzero(balloon_mask)
        if balloon_area == 0:
            return False
        overlap_ratio = overlap / balloon_area
        balloon_center_x = self.x + self.width // 2
        balloon_center_y = self.y + self.height // 2
        object_nearby = False
        for box in detected_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            distance = np.sqrt((cx - balloon_center_x)**2 + (cy - balloon_center_y)**2)
            if distance < (min(self.width, self.height) // 2 + min(x2-x1, y2-y1) // 2):
                object_nearby = True
                break
        if overlap_ratio > 0.05 and object_nearby:
            self.popped = True
            self.pop_time = time.time()
            return True
        return False

class CrackEffect:
    def __init__(self, x, y, duration=0.3):
        self.x = x
        self.y = y
        self.start_time = time.time()
        self.duration = duration

    def draw(self, win):
        if boom_img and time.time() - self.start_time < self.duration:
            win.blit(boom_img, (self.x, self.y))
            return True
        return False

# Initialize camera
cap = cv2.VideoCapture(0)
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

# Initialize game variables
NUM_BALLOONS = 3  # Number of balloons to display
balloon_images_shuffled = BALLOON_IMAGES.copy()  # Copy to avoid modifying original
random.shuffle(balloon_images_shuffled)  # Randomize balloon types
balloons = [Balloon(balloon_images_shuffled[i % len(balloon_images_shuffled)]) for i in range(NUM_BALLOONS)]
# Explicitly reset each balloon after list creation
for balloon in balloons:
    balloon.reset()
cracks = []
score = 0
start_time = pygame.time.get_ticks()
game_over = False
in_calibration = False
detector = EdgeDetector(low_threshold=30, high_threshold=100)  # Adjusted for better sensitivity
show_debug_overlay = False

# Main loop
running = True
while running:
    CLOCK.tick(FPS)
    ret, frame = cap.read()
    if not ret:
        print("Warning: Could not read frame")
        continue
    warped_frame = cv2.warpPerspective(frame, transform_matrix, (SCREEN_WIDTH, SCREEN_HEIGHT))
    warped_frame = cv2.GaussianBlur(warped_frame, (5, 5), 0)

    # Run YOLO detection
    results = model.predict(warped_frame, imgsz=640, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device="cpu", verbose=False)
    
    # Create object mask from YOLO detections
    object_mask = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)
    detected_boxes = []
    for result in results:
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                radius = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2)
                cv2.circle(object_mask, (cx, cy), radius, 255, -1)
                detected_boxes.append(box)

    # Create debug view
    debug_view = warped_frame.copy()
    for box in detected_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        radius = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2)
        confidence = float(box.conf[0])
        cv2.circle(debug_view, (cx, cy), radius, (0, 255, 0), 2)
        cv2.putText(debug_view, f"Green Ball: {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(debug_view, f"Screen: {SCREEN_WIDTH}x{SCREEN_HEIGHT}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Camera Feed", debug_view)

    # Create edge debug view
    debug_edge_view = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    for b in balloons:
        if not b.popped:
            balloon_crop = warped_frame[int(b.y):int(b.y + b.height), int(b.x):int(b.x + b.width)]
            if balloon_crop.size == 0:
                continue
            gray_crop = cv2.cvtColor(balloon_crop, cv2.COLOR_BGR2GRAY)
            balloon_mask = detector.detect_edges_from_array(gray_crop)
            contours, _ = cv2.findContours(balloon_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            offset_contours = [cnt + [int(b.x), int(b.y)] for cnt in contours]
            cv2.drawContours(debug_edge_view, offset_contours, -1, (255, 255, 255), 2)
    if object_mask is not None:
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug_edge_view, contours, -1, (0, 0, 255), 2)
    cv2.imshow("Edge Debug View", debug_edge_view)

    elapsed_time = (pygame.time.get_ticks() - start_time) / 1000
    time_left = max(0, GAME_DURATION - int(elapsed_time))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_c:
                in_calibration = True
                calibration_points = get_calibration_points(cap)
                if calibration_points and len(calibration_points) == 4:
                    offset_x, offset_y = 0, 0
                    debug_offset_x, debug_offset_y = 0, 0
                    save_calibration_points(calibration_points, offset_x, offset_y, debug_offset_x, debug_offset_y)
                    transform_matrix = get_perspective_transform(calibration_points, offset_x, offset_y)
                    test_calibration_accuracy(transform_matrix, calibration_points)
                    in_calibration = False
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
            elif event.key == pygame.K_p:
                save_calibration_points(calibration_points, offset_x, offset_y, debug_offset_x, debug_offset_y)
            elif game_over:
                if event.key == pygame.K_r:
                    score = 0
                    start_time = pygame.time.get_ticks()
                    game_over = False
                    random.shuffle(balloon_images_shuffled)  # Re-shuffle balloons on restart
                    for i, b in enumerate(balloons):
                        b.image = balloon_images_shuffled[i % len(balloon_images_shuffled)]
                        b.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False

    if not game_over and not in_calibration:
        WIN.fill(WHITE)
        for b in balloons:
            if b.check_object_occlusion(object_mask, detected_boxes, detector, warped_frame):
                if pop_sound:
                    pop_sound.play()
                cracks.append(CrackEffect(b.x, b.y))
                score += 1
            b.update()
            b.draw(WIN)
        cracks = [c for c in cracks if c.draw(WIN)]
        if show_debug_overlay:
            for box in detected_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                radius = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2)
                pygame.draw.circle(WIN, (0, 255, 0), (cx, cy), radius, 2)
        timer_text = FONT.render(f"Time Left: {time_left}s", True, BLACK)
        score_text = FONT.render(f"Score: {score}", True, BLACK)
        offset_text = FONT.render(f"Homography Offset: ({offset_x}, {offset_y})", True, BLACK)
        draw_text_with_bg(WIN, timer_text, 20, 20)
        draw_text_with_bg(WIN, score_text, 20, 60)
        draw_text_with_bg(WIN, offset_text, 20, 100)
        if elapsed_time >= GAME_DURATION:
            game_over = True
    else:
        final_text = BIG_FONT.render("Game Over!", True, RED)
        final_score = FONT.render(f"Your Final Score: {score}", True, BLACK)
        restart_hint = FONT.render("Press R to replay or ESC to exit", True, BLACK)
        draw_text_with_bg(WIN, final_text, SCREEN_WIDTH//2 - final_text.get_width()//2, SCREEN_HEIGHT//2 - 80)
        draw_text_with_bg(WIN, final_score, SCREEN_WIDTH//2 - final_score.get_width()//2, SCREEN_HEIGHT//2 - 20)
        draw_text_with_bg(WIN, restart_hint, SCREEN_WIDTH//2 - restart_hint.get_width()//2, SCREEN_HEIGHT//2 + 30)

    pygame.display.update()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()