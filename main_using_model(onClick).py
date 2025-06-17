import pygame
import pyautogui
import cv2
import numpy as np
import sys
import json
import os
import random
import time
import logging
from modules.background import draw_text_with_bg
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
BACKGROUND_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "background.jpg")
CALIBRATION_FILE = "calibration.json"
CAMERA_INDEX = 1  # Configurable camera index
BALLOON_FILES = ["balloon1.png", "balloon2.png", "balloon3.png", 'balloon4.png']
POP_SOUND_PATH = "pop.wav"

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
pygame.display.set_caption("Balloon Popping Game")
clock = pygame.time.Clock()

actual_width, actual_height = screen.get_size()

# Load assets
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
BALLOON_IMAGES = []

# Load balloon images
for file in BALLOON_FILES:
    path = os.path.join(base_dir, "assets", file)
    if not os.path.exists(path):
        print(f"Warning: Balloon image {path} not found. Skipping.")
        continue
    try:
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.scale(img, (520, 560))
        BALLOON_IMAGES.append(img)
    except pygame.error as e:
        print(f"Warning: Failed to load {path}: {e}. Skipping.")

if not BALLOON_IMAGES:
    print("Error: No valid balloon images loaded. Exiting.")
    pygame.quit()
    sys.exit(1)

# Load crack image
try:
    crack_img = pygame.image.load(CRACK_PATH).convert_alpha()
    crack_img = pygame.transform.scale(crack_img, (260, 150))
except pygame.error as e:
    print(f"Error loading boom1.png: {e}")
    sys.exit(1)

# Load pop sound
def load_sound(filename):
    path = os.path.join(base_dir, "assets", filename)
    if os.path.exists(path):
        try:
            return pygame.mixer.Sound(path)
        except pygame.error as e:
            print(f"Failed to load sound {filename}: {e}")
    return None

pop_sound = load_sound(POP_SOUND_PATH)

# Load background image
try:
    background_image = pygame.image.load(BACKGROUND_IMAGE_PATH).convert()
    background_image = pygame.transform.scale(background_image, (external_screen.width, external_screen.height))
except pygame.error as e:
    print(f"Error loading background image {BACKGROUND_IMAGE_PATH}: {e}")
    pygame.quit()
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
cap = cv2.VideoCapture(CAMERA_INDEX)
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

# Colors and fonts
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
FONT = pygame.font.SysFont("arial", 32)
BIG_FONT = pygame.font.SysFont("arial", 48)

# Balloon class
class Balloon:
    def __init__(self):
        self.image = random.choice(BALLOON_IMAGES)
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.radius = min(self.width, self.height) // 2
        self.popped = False
        self.reset()

    def reset(self):
        self.x = random.randint(0, actual_width - self.width)
        self.y = actual_height + random.randint(0, 300)
        self.speed = random.uniform(8.0, 9.0)
        self.popped = False

    def update(self):
        if not self.popped:
            self.y -= self.speed
            if self.y + self.height < 0:
                self.reset()

    def draw(self, win):
        if not self.popped:
            win.blit(self.image, (self.x, self.y))

    def is_clicked(self, pos):
        cx = self.x + self.width // 2
        cy = self.y + self.height // 2
        dx = pos[0] - cx
        dy = pos[1] - cy
        return dx * dx + dy * dy <= self.radius * self.radius

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
balloons = [Balloon() for _ in range(1)]
cracks = []
last_click_time = 0
running = True
show_debug_overlay = False
font = pygame.font.SysFont(None, 36)
score = 0
GAME_DURATION = 120
start_time = pygame.time.get_ticks()
game_over = False

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

    if not game_over:
        # Process detections
        current_time = time.time()
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    # Apply inverse perspective transform to convert to screen coordinates
                    point = np.float32([[[cx, cy]]])
                    warped_point = cv2.perspectiveTransform(point, inv_transform_matrix)[0][0]
                    screen_x, screen_y = warped_point
        
                    if (0 <= cx <= external_screen.width and 0 <= cy <= external_screen.height and 
                        current_time - last_click_time >= CLICK_COOLDOWN):
                        screen_x = int(cx + debug_offset_x + external_screen.x)
                        screen_y = int(cy + debug_offset_y + external_screen.y)

                        # Move the mouse and click
                        pyautogui.moveTo(screen_x, screen_y)
                        pyautogui.click(button='left')

                        for balloon in balloons[:]:
                            if balloon.is_clicked((screen_x, screen_y)):
                                balloon.popped = True
                                if pop_sound:
                                    pop_sound.play()
                                crack_x = int(screen_x - crack_img.get_width() / 2)
                                crack_y = int(screen_y - crack_img.get_height() / 2)
                                cracks.append(CrackEffect(crack_x, crack_y))
                                balloons.remove(balloon)
                                score += 1
                                last_click_time = current_time
                                break
        # Add new balloons if needed
        if len(balloons) < 3:
            balloons.append(Balloon())
    
    # Update and draw balloons
    for balloon in balloons:
        balloon.update()    
    elapsed_time = (pygame.time.get_ticks() - start_time) / 1000
    time_left = max(0, GAME_DURATION - int(elapsed_time))
    if elapsed_time >= GAME_DURATION and not game_over:
        game_over = True

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
                    for balloon in balloons[:]:
                        if balloon.is_clicked((mx, my)):
                            balloon.popped = True
                            if pop_sound:
                                pop_sound.play()
                            balloons.remove(balloon)
                            score += 1
                            break
                    last_click_time = current_time
    # Render screen
    screen.blit(background_image, (0, 0))
    if not game_over:
        for balloon in balloons:
            balloon.draw(screen)
        cracks = [c for c in cracks if c.draw(screen)]
        timer_text = FONT.render(f"Time Left: {time_left}s", True, (0,0,0))
        score_text = FONT.render(f"Score: {score}", True, (0,0,0))
        draw_text_with_bg(screen, timer_text, 20, 20)
        draw_text_with_bg(screen, score_text, 20, 60)
    else:
        final_text = BIG_FONT.render("Game Over!", True, RED)
        final_score = FONT.render(f"Your Final Score: {score}", True, BLACK)
        restart_hint = FONT.render("Press R to replay or ESC to exit", True, BLACK)
        draw_text_with_bg(screen, final_text, SCREEN_WIDTH//2 - final_text.get_width()//2, SCREEN_HEIGHT//2 - 80)
        draw_text_with_bg(screen, final_score, SCREEN_WIDTH//2 - final_score.get_width()//2, SCREEN_HEIGHT//2 - 20)
        draw_text_with_bg(screen, restart_hint, SCREEN_WIDTH//2 - restart_hint.get_width()//2, SCREEN_HEIGHT//2 + 30)
    
    pygame.display.flip()
    
    # Check for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()