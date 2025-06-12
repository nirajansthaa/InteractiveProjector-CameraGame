import pygame
import cv2
import numpy as np
import sys
import json
import os
import time
from screeninfo import get_monitors
from modules.circle_mask_detection import get_circle_mask

# Constants
SCREEN_WIDTH = 1360  # External monitor width
SCREEN_HEIGHT = 768  # External monitor height
FPS = 60

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
pygame.display.set_caption("Click Detection")
CLOCK = pygame.time.Clock()
pygame.mouse.set_visible(True)

actual_width, actual_height = WIN.get_size()

# Load assets
base_dir = os.path.dirname(os.path.abspath(__file__))
try:
    crack = os.path.join(base_dir, "assets", 'crack.png')
    crack_img = pygame.image.load(crack).convert_alpha()
    crack_img = pygame.transform.scale(crack_img, (260, 150))
except pygame.error as e:
    print(f"Warning: Failed to load boom.png: {e}. No boom image.")
    boom_img = None

# Colors and fonts
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)

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
cracks = []
background_frame = None
in_calibration = False
last_click_time = 0
click_cooldown = 0.5  # Seconds between detected clicks

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
            # Simulate click at circle center if within screen bounds
            current_time = time.time()
            if (0 <= cx <= SCREEN_WIDTH and 0 <= cy <= SCREEN_HEIGHT and 
                current_time - last_click_time >= click_cooldown):
                cracks.append(CrackEffect(cx - crack_img.get_width() // 2, 
                                        cy - crack_img.get_height() // 2))
                last_click_time = current_time
    if object_mask is not None:
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug_view, contours, -1, (0, 0, 255), 2)
    cv2.imshow("Camera Feed", debug_view)

    # Create debug view for edges
    debug_view = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    
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

    if not in_calibration:
        WIN.fill(WHITE)
        cracks = [c for c in cracks if c.draw(WIN)]

    pygame.display.update()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()