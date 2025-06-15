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
from modules.calibration import *

# Suppress logging
os.environ["YOLO_VERBOSE"] = "False"
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
cv2.setLogLevel(0)

from ultralytics import YOLO

# Constants
SCREEN_WIDTH = 1360
SCREEN_HEIGHT = 768
FPS = 60
GAME_DURATION = 120  # Seconds
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.7
CLICK_COOLDOWN = 0.5
MODEL_PATH = "best.onnx"
CALIBRATION_FILE = "calibration.json"
NUM_BALLOONS = 3

# Get monitor info for projector display
monitors = get_monitors()
if len(monitors) < 2:
    print("Error: External monitor (projector) not detected")
    sys.exit(1)
external_screen = monitors[1]
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{external_screen.x},{external_screen.y}"

# Initialize Pygame
pygame.init()
WIN = pygame.display.set_mode((external_screen.width, external_screen.height))  # Enforce 1360x768
pygame.display.set_caption("Balloon Popping Game")
CLOCK = pygame.time.Clock()
pygame.mouse.set_visible(True)

actual_width, actual_height = WIN.get_size()
print(f"Window size: {actual_width}x{actual_height}")

# Load YOLO model
try:
    model = YOLO(MODEL_PATH, task="detect", verbose=False)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    pygame.quit()
    sys.exit(1)

# Load assets
base_dir = os.path.dirname(os.path.abspath(__file__))
BALLOON_IMAGES = []
for file in ["balloon1.png", "balloon2.png", "balloon3.png"]:
    path = os.path.join(base_dir, "assets", file)
    if os.path.exists(path):
        try:
            img = pygame.image.load(path).convert_alpha()
            img = pygame.transform.scale(img, (320, 360))
            BALLOON_IMAGES.append(img)
        except pygame.error as e:
            print(f"Warning: Failed to load {path}: {e}")
    else:
        print(f"Warning: Balloon image {path} not found")
if not BALLOON_IMAGES:
    print("Error: No valid balloon images loaded. Exiting.")
    pygame.quit()
    sys.exit(1)

pop_sound = pygame.mixer.Sound(os.path.join(base_dir, "assets", "pop.wav")) if os.path.exists(os.path.join(base_dir, "assets", "pop.wav")) else None
boom_img = pygame.image.load(os.path.join(base_dir, "assets", "boom1.png")).convert_alpha() if os.path.exists(os.path.join(base_dir, "assets", "boom1.png")) else None
if boom_img:
    boom_img = pygame.transform.scale(boom_img, (290, 180))

# Colors and fonts
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
FONT = pygame.font.SysFont("arial", 32)
BIG_FONT = pygame.font.SysFont("arial", 48)

class Balloon:
    def __init__(self, image):
        self.image = image
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.popped = False
        self.pop_time = None
        self.x = 0
        self.y = 0
        self.speed = 0
        self.reset()

    def reset(self, existing_balloons=None):
        max_attempts = 10
        for _ in range(max_attempts):
            proposed_x = random.randint(0, SCREEN_WIDTH - self.width)
            proposed_y = -self.height + random.randint(0, 50)  # Start just above screen
            overlap = False
            if existing_balloons:
                for other in existing_balloons:
                    if other is not self and not other.popped:
                        distance_x = abs(proposed_x - other.x)
                        distance_y = abs(proposed_y - other.y)
                        min_distance = (self.width + other.width) // 4
                        if distance_x < min_distance and distance_y < min_distance:
                            overlap = True
                            break
            if not overlap:
                self.x = proposed_x
                self.y = proposed_y
                self.speed = random.uniform(3.75, 6.0)
                self.popped = False
                print(f"Balloon reset to ({self.x}, {self.y}) with speed {self.speed}")
                return
        self.x = random.randint(0, SCREEN_WIDTH - self.width)
        self.y = -self.height + random.randint(0, 50)
        self.speed = random.uniform(3.75, 6.0)
        self.popped = False
        print(f"Balloon fallback reset to ({self.x}, {self.y}) with speed {self.speed}")

    def update(self):
        if self.popped and self.pop_time and time.time() - self.pop_time >= 0.3:
            self.reset()
        elif not self.popped:
            self.y += self.speed  # Move downward
            if self.y > SCREEN_HEIGHT:
                self.reset()

    def draw(self, win):
        if not self.popped and 0 <= self.x < SCREEN_WIDTH and 0 <= self.y < SCREEN_HEIGHT:
            win.blit(self.image, (self.x, self.y))

    def is_detected(self, box):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_cx = (x1 + x2) // 2
        box_cy = (y1 + y2) // 2
        balloon_cx = self.x + self.width // 2 + debug_offset_x
        balloon_cy = self.y + self.height // 2 + debug_offset_y
        dx = box_cx - balloon_cx
        dy = box_cy - balloon_cy
        return dx * dx + dy * dy <= (self.width // 2) ** 2

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
    print(f"Loading existing calibration with homography offset ({offset_x}, {offset_y}) and debug offset ({debug_offset_x}, {debug_offset_y})")
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

# Initialize game variables
balloon_images_shuffled = BALLOON_IMAGES.copy()
random.shuffle(balloon_images_shuffled)
balloons = [Balloon(balloon_images_shuffled[i % len(balloon_images_shuffled)]) for i in range(NUM_BALLOONS)]
cracks = []
score = 0
start_time = pygame.time.get_ticks()
game_over = False
last_click_time = 0  # Initialize click cooldown
show_debug_overlay = False

# Main loop
running = True
while running:
    CLOCK.tick(FPS)
    
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
    detected_boxes = []
    for result in results:
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if 0 <= cx <= SCREEN_WIDTH and 0 <= cy <= SCREEN_HEIGHT and current_time - last_click_time >= CLICK_COOLDOWN:
                    detected_boxes.append(box)
    
    # Handle balloon pops
    for balloon in balloons[:]:
        for box in detected_boxes:
            if balloon.is_detected(box):
                balloon.popped = True
                if pop_sound:
                    pop_sound.play()
                cracks.append(CrackEffect(balloon.x, balloon.y))
                balloons.remove(balloon)
                score += 1
                last_click_time = current_time
                break

    # Update and draw balloons
    WIN.fill(WHITE)
    for balloon in balloons:
        balloon.update()
        balloon.draw(WIN)

    # Add new balloons if needed
    if len(balloons) < NUM_BALLOONS:
        new_balloon = Balloon(random.choice(BALLOON_IMAGES))
        new_balloon.reset(balloons)
        balloons.append(new_balloon)

    # Draw crack effects
    cracks = [c for c in cracks if c.draw(WIN)]

    # Debug view
    debug_view = warped_frame.copy()
    roi_points = np.float32([[0, 0], [SCREEN_WIDTH-1, 0], [SCREEN_WIDTH-1, SCREEN_HEIGHT-1], [0, SCREEN_HEIGHT-1]])
    roi_points = roi_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(debug_view, [roi_points], True, (255, 0, 0), 2)
    for result in results:
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2 + debug_offset_x
                cy = (y1 + y2) // 2 + debug_offset_y
                confidence = float(box.conf[0])
                cv2.rectangle(debug_view, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(debug_view, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.putText(debug_view, f"Confidence: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.putText(debug_view, f"Screen: {SCREEN_WIDTH}x{SCREEN_HEIGHT}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_view, f"Offset: ({offset_x}, {offset_y})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_view, f"Debug Offset: ({debug_offset_x}, {debug_offset_y})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Camera Feed", debug_view)

    # Handle events
    elapsed_time = (pygame.time.get_ticks() - start_time) / 1000
    time_left = max(0, GAME_DURATION - int(elapsed_time))
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
                    offset_x, offset_y = 280, 125
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
                print(f"Adjusted offset to ({offset_x}, {offset_y})")
            elif event.key == pygame.K_RIGHT:
                offset_x += 5
                transform_matrix = get_perspective_transform(calibration_points, offset_x, offset_y)
                print(f"Adjusted offset to ({offset_x}, {offset_y})")
            elif event.key == pygame.K_UP:
                offset_y -= 5
                transform_matrix = get_perspective_transform(calibration_points, offset_x, offset_y)
                print(f"Adjusted offset to ({offset_x}, {offset_y})")
            elif event.key == pygame.K_DOWN:
                offset_y += 5
                transform_matrix = get_perspective_transform(calibration_points, offset_x, offset_y)
                print(f"Adjusted offset to ({offset_x}, {offset_y})")
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
            elif game_over and event.key == pygame.K_r:
                score = 0
                start_time = pygame.time.get_ticks()
                game_over = False
                for b in balloons:
                    b.reset()

    # Render screen
    if not game_over:
        for crack in cracks:
            crack.draw(WIN)
        if show_debug_overlay:
            for box in detected_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2 + debug_offset_x
                cy = (y1 + y2) // 2 + debug_offset_y
                pygame.draw.circle(WIN, (255, 0, 0), (int(cx), int(cy)), 5)
                pygame.draw.rect(WIN, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)

        # Display game info
        timer_text = FONT.render(f"Time Left: {time_left}s", True, BLACK)
        score_text = FONT.render(f"Score: {score}", True, BLACK)
        offset_text = FONT.render(f"Offset: ({offset_x}, {offset_y})", True, BLACK)
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

    pygame.display.flip()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()