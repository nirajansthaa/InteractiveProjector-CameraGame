import pygame
import cv2
import numpy as np
import random
import sys
import json
import os
import time
from screeninfo import get_monitors
from modules.circle_mask_detection import get_circle_mask
from modules.background import draw_text_with_bg
from modules.edge_detection import EdgeDetector

# Constants
SCREEN_WIDTH = 1360  # External monitor width
SCREEN_HEIGHT = 768  # External monitor height
FPS = 60
GAME_DURATION = 120  # Seconds

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
external_origin = (1920, 0)

os.environ['SDL_VIDEO_WINDOW_POS'] = f"{external_origin[0]},{external_origin[1]}"

# Calibration functions
def get_calibration_points(cap, window_name="Calibration"):
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            print(f"Point {len(points)} selected: ({x}, {y})")
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("Click the four corners of the monitor in the camera feed (top-left, top-right, bottom-right, bottom-left). Press 'c' to confirm.")
    
    while len(points) < 4:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera during calibration")
            return None
        for pt in points:
            cv2.circle(frame, (pt[0], pt[1]), 5, (0, 255, 0), -1)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('c') and len(points) == 4:
            break
    
    cv2.destroyWindow(window_name)
    return points

def save_calibration_points(points, filename="calibration.json"):
    try:
        with open(filename, 'w') as f:
            json.dump(points, f)
    except Exception as e:
        print(f"Error saving calibration points: {e}")

def load_calibration_points(filename="calibration.json"):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading calibration points: {e}")
        return None

def get_perspective_transform(src_points, screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT):
    dst_points = np.float32([
        [0, 0],
        [screen_width, 0],
        [screen_width, screen_height],
        [0, screen_height]
    ])
    src_points = np.float32(src_points)
    return cv2.getPerspectiveTransform(src_points, dst_points)

# Initialize Pygame
pygame.init()
WIN = pygame.display.set_mode((external_screen.width, external_screen.height))
pygame.display.set_caption("Balloon Pop Game")
CLOCK = pygame.time.Clock()
pygame.mouse.set_visible(True)

actual_width, actual_height = WIN.get_size()

# Load assets
base_dir = os.path.dirname(os.path.abspath(__file__))
BALLOON_IMAGES = []
balloon_files = ["balloon1.png", "balloon2.png", "balloon3.png"]
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
if not BALLOON_IMAGES:
    print("Error: No valid balloon images loaded. Exiting.")
    pygame.quit()
    sys.exit(1)

try:
    pop_sound = pygame.mixer.Sound(os.path.join(base_dir, "assets", "pop.wav"))
except pygame.error as e:
    print(f"Warning: Failed to load pop.wav: {e}. No pop sound.")
    pop_sound = None

try:
    boom_path = os.path.join(base_dir, "assets", 'boom.png')
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
    def __init__(self):
        self.image = random.choice(BALLOON_IMAGES)
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.popped = False
        self.pop_time = None
        self.reset()

    def reset(self):
        self.x = random.randint(0, actual_width - self.width)
        self.y = actual_height + random.randint(0, 300)
        self.speed = random.uniform(3.75, 6.0)
        self.popped = False

    def update(self):
        if self.popped:
            if time.time() - self.pop_time >= 0.3:
                self.reset()
        else:
            self.y -= self.speed
            if self.y + self.height < 0:
                self.reset()

    def draw(self, win):
        if not self.popped:
            win.blit(self.image, (self.x, self.y))

    def check_circle_occlusion(self, circle_mask, detected_circles, detector, warped_frame):
        if self.popped or circle_mask is None:
            return False
        x1, y1 = int(self.x), int(self.y)
        x2, y2 = x1 + self.width, y1 + self.height
        if x1 < 0 or y1 < 0 or x2 > SCREEN_WIDTH or y2 > SCREEN_HEIGHT:
            return False
        # Extract balloon region
        balloon_crop = warped_frame[y1:y2, x1:x2]
        if balloon_crop.size == 0:
            return False
        # Convert to grayscale and detect edges
        gray_crop = cv2.cvtColor(balloon_crop, cv2.COLOR_BGR2GRAY)
        balloon_mask = detector.detect_edges_from_array(gray_crop)
        # Resize circle_mask to match balloon_crop
        circle_crop = circle_mask[y1:y2, x1:x2]
        if circle_crop.shape != balloon_mask.shape:
            circle_crop = cv2.resize(circle_crop, (balloon_mask.shape[1], balloon_mask.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        # Compute overlap
        intersection = cv2.bitwise_and(circle_crop, balloon_mask)
        overlap = np.count_nonzero(intersection)
        balloon_area = np.count_nonzero(balloon_mask)
        if balloon_area == 0:
            return False
        overlap_ratio = overlap / balloon_area
        # Check for nearby circle
        balloon_center_x = self.x + self.width // 2
        balloon_center_y = self.y + self.height // 2
        circle_nearby = False
        for cx, cy, cr in detected_circles:
            distance = np.sqrt((cx - balloon_center_x)**2 + (cy - balloon_center_y)**2)
            if distance < (cr + min(self.width, self.height) // 2):
                circle_nearby = True
                break
        # Pop if significant overlap and circle nearby
        if overlap_ratio > 0.05 and circle_nearby:  # Adjusted threshold
            self.popped = True
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
    
def perform_calibration(cap):
    calibration_points = get_calibration_points(cap)
    if calibration_points and len(calibration_points) == 4:
        save_calibration_points(calibration_points)
        return get_perspective_transform(calibration_points)
    print("Error: Calibration failed.")
    return None

# Initialize camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera")
    pygame.quit()
    sys.exit(1)

# Load or perform calibration
calibration_file = os.path.join(base_dir, "calibration.json")
calibration_points = load_calibration_points(calibration_file)
transform_matrix = None
if calibration_points and len(calibration_points) == 4:
    transform_matrix = get_perspective_transform(calibration_points)
else:
    print("Performing camera calibration...")
    transform_matrix = perform_calibration(cap)
    if transform_matrix is None:
        cap.release()
        pygame.quit()
        sys.exit(1)

# Initialize game variables
balloons = [Balloon() for _ in range(4)]
cracks = []
score = 0
start_time = pygame.time.get_ticks()
game_over = False
background_frame = None
in_calibration = False
detector = EdgeDetector(low_threshold=100, high_threshold=200)

# Main loop
running = True
while running:
    CLOCK.tick(FPS)
    ret, frame = cap.read()
    if not ret:
        print("Warning: Could not read frame")
        continue
    # frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
    warped_frame = cv2.warpPerspective(frame, transform_matrix, (SCREEN_WIDTH, SCREEN_HEIGHT))
    # Apply light Gaussian blur to reduce noise
    warped_frame = cv2.GaussianBlur(warped_frame, (5, 5), 0)
    
    if background_frame is None:
        background_frame = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        background_frame = cv2.GaussianBlur(background_frame, (21, 21), 0)
        continue
    
    object_mask, detected_circles = get_circle_mask(warped_frame, background_frame)
    
    # Create debug view
    debug_view = warped_frame.copy()
    if detected_circles is not None:
        for cx, cy, cr in detected_circles:
            cv2.circle(debug_view, (int(cx), int(cy)), int(cr), (0, 255, 0), 2)
    if object_mask is not None:
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug_view, contours, -1, (0, 0, 255), 2)
    # debug_view = cv2.resize(debug_view, (640, 480))
    cv2.imshow("Camera Feed", debug_view)

    # Create debug view (single window showing balloon and occlusion edges)
    debug_view = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)  # Black background
    for b in balloons:
        if not b.popped:
            balloon_crop = warped_frame[int(b.y):int(b.y + b.height), int(b.x):int(b.x + b.width)]
            if balloon_crop.size == 0:
                continue
            gray_crop = cv2.cvtColor(balloon_crop, cv2.COLOR_BGR2GRAY)
            balloon_mask = detector.detect_edges_from_array(gray_crop)
            contours, _ = cv2.findContours(balloon_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            offset_contours = [cnt + [int(b.x), int(b.y)] for cnt in contours]
            cv2.drawContours(debug_view, offset_contours, -1, (255, 255, 255), 2)  # Blue for balloon edges
    if object_mask is not None:
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug_view, contours, -1, (0, 0, 255), 2)  # Red for occlusion edges
    if detected_circles is not None:
        for cx, cy, cr in detected_circles:
            cv2.circle(debug_view, (int(cx), int(cy)), int(cr), (0, 255, 0), 2)  # Green for detected circles (optional)
    # debug_view = cv2.resize(debug_view, (640, 480))
    cv2.imshow("Edge Debug View", debug_view)

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
                transform_matrix = perform_calibration(cap)
                if transform_matrix is None:
                    running = False
            elif game_over:
                if event.key == pygame.K_r:
                    score = 0
                    start_time = pygame.time.get_ticks()
                    game_over = False
                    for b in balloons:
                        b.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False

    if not game_over and not in_calibration:
        WIN.fill(WHITE)
        for b in balloons:
            if b.check_circle_occlusion(object_mask, detected_circles, detector, warped_frame):
                if pop_sound:
                    pop_sound.play()
                cracks.append(CrackEffect(b.x, b.y))
                score += 1
            b.update()
            b.draw(WIN)
        cracks = [c for c in cracks if c.draw(WIN)]
        timer_text = FONT.render(f"Time Left: {time_left}s", True, BLACK)
        score_text = FONT.render(f"Score: {score}", True, BLACK)
        draw_text_with_bg(WIN, timer_text, 20, 20)
        draw_text_with_bg(WIN, score_text, 20, 60)
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