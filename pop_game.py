import pygame
import cv2
import numpy as np
import random
import math
import sys
import os
import time
from screeninfo import get_monitors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.camera_capture import capture_camera_points
from modules.projector_display import show_calibration_pattern
from modules.calibrate import compute_homography

# Configuration
CAMERA_ID = 1
SCREEN_WIDTH, SCREEN_HEIGHT = 1900, 1000  # Projector resolution
FPS = 60
GAME_DURATION = 120  # seconds

# Get monitor info for projector display
monitors = get_monitors()
if not monitors:
    print("Error: No monitors detected")
    sys.exit(1)

main_screen = monitors[0]
external_screen = monitors[1] if len(monitors) > 1 else monitors[0]
external_origin = (external_screen.x, external_screen.y)

# Set Pygame window position to external screen
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{external_origin[0]},{external_origin[1]}"

# Initialize camera
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print(f"Error: Could not open camera with ID {CAMERA_ID}")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

# Calibration
def calibrate_system():
    # Step 1: Show calibration pattern on projector
    projector_points = show_calibration_pattern(screen_resolution=(SCREEN_WIDTH, SCREEN_HEIGHT))
    
    # Step 2: Capture camera points
    global clicked_points
    clicked_points = []  # Reset global variable from camera_capture
    camera_points = capture_camera_points(resolution=(SCREEN_WIDTH, SCREEN_HEIGHT))
    if len(camera_points) != 4:
        print("Error: Exactly 4 points required for calibration")
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    # Debug: Print points to verify
    print("Camera points:", camera_points)
    print("Projector points:", projector_points)
    
    # Step 3: Compute homography
    try:
        homography = compute_homography(camera_points, projector_points)
        return homography
    except cv2.error as e:
        print(f"Error computing homography: {e}")
        return None
    
cv2.namedWindow("Camera Feed (Press Q to Quit)", cv2.WINDOW_NORMAL)
cv2.moveWindow("Camera Feed (Press Q to Quit)", main_screen.x, main_screen.y)

# Initialize Pygame
pygame.init()
WIN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Balloon Pop Game")
CLOCK = pygame.time.Clock()
pygame.mouse.set_visible(False)

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
    cap.release()
    pygame.quit()
    sys.exit(1)

try:
    pop_sound = pygame.mixer.Sound(os.path.join(base_dir, "assets", "pop.wav"))
except pygame.error as e:
    print(f"Warning: Failed to load pop.wav: {e}. No pop sound.")
    pop_sound = None

background_img = None

# Colors and fonts
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = pygame.font.SysFont("arial", 32)
BIG_FONT = pygame.font.SysFont("arial", 48)

def draw_text_with_bg(win, text_surface, x, y, padding=10, bg_color=(255, 255, 255, 180)):
    rect = text_surface.get_rect(topleft=(x, y))
    background = pygame.Surface((rect.width + padding*2, rect.height + padding*2), pygame.SRCALPHA)
    background.fill(bg_color)
    win.blit(background, (x - padding, y - padding))
    win.blit(text_surface, (x, y))

def get_circle_mask(frame, background, threshold=30, min_radius=20, max_radius=150):
    """
    Enhanced object mask that specifically detects circular objects
    """
    # Standard motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if background is None:
        return None, []
    
    # Background subtraction
    diff = cv2.absdiff(background, blur)
    _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Clean up the motion mask
    kernel = np.ones((5,5), np.uint8)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    
    # Detect circles using HoughCircles on the current frame
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,  # Minimum distance between circle centers
        param1=50,   # Upper threshold for edge detection
        param2=30,   # Accumulator threshold for center detection
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    # Create circle mask
    circle_mask = np.zeros_like(motion_mask)
    detected_circles = []
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Only consider circles that overlap with motion areas
            circle_roi = motion_mask[max(0, y-r):min(motion_mask.shape[0], y+r+1),
                                   max(0, x-r):min(motion_mask.shape[1], x+r+1)]
            
            if np.count_nonzero(circle_roi) > 50:  # Minimum motion pixels in circle area
                cv2.circle(circle_mask, (x, y), r, 255, -1)
                detected_circles.append((x, y, r))
    
    # Combine motion mask with circle detection
    combined_mask = cv2.bitwise_and(motion_mask, circle_mask)
    
    return combined_mask, detected_circles

def get_enhanced_circle_mask(frame, background, threshold=30, circle_threshold=0.7):
    """
    Advanced version using contour analysis for better circular object detection
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if background is None:
        return None, []
    
    # Background subtraction
    diff = cv2.absdiff(background, blur)
    _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circle_mask = np.zeros_like(motion_mask)
    circular_objects = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # Skip small contours
            continue
            
        # Calculate circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Check if object is circular enough
        if circularity > circle_threshold:
            # Get bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Draw filled circle on mask
            cv2.circle(circle_mask, center, radius, 255, -1)
            circular_objects.append((center[0], center[1], radius))
    
    return circle_mask, circular_objects

# def get_color_circle_mask(frame, background, color_range, threshold=30):
#     """
#     Detect circles of specific colors (e.g., colored balls)
#     color_range should be [(lower_hsv1, upper_hsv1), (lower_hsv2, upper_hsv2), ...]
#     """
#     # Convert to HSV for better color detection
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
#     # Create color mask
#     color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
#     for lower, upper in color_range:
#         mask = cv2.inRange(hsv, lower, upper)
#         color_mask = cv2.bitwise_or(color_mask, mask)
    
#     # Clean up color mask
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
#     color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    
#     # Find circular contours in color mask
#     contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     circle_mask = np.zeros_like(color_mask)
#     detected_objects = []
    
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area < 300:
#             continue
            
#         # Check circularity
#         perimeter = cv2.arcLength(contour, True)
#         if perimeter == 0:
#             continue
            
#         circularity = 4 * np.pi * area / (perimeter * perimeter)
        
#         if circularity > 0.6:  # Reasonably circular
#             (x, y), radius = cv2.minEnclosingCircle(contour)
#             center = (int(x), int(y))
#             radius = int(radius)
            
#             cv2.circle(circle_mask, center, radius, 255, -1)
#             detected_objects.append((center[0], center[1], radius))
    
#     return circle_mask, detected_objects

class Balloon:
    def check_circle_occlusion(self, circle_mask, detected_circles):
        """
        Enhanced occlusion check specifically for circular objects
        """
        if self.popped or circle_mask is None:
            return
            
        x1, y1 = int(self.x), int(self.y)
        x2, y2 = x1 + self.width, y1 + self.height
        
        if x1 < 0 or y1 < 0 or x2 > SCREEN_WIDTH or y2 > SCREEN_HEIGHT:
            return
        
        # Check overlap with circle mask
        circle_crop = circle_mask[y1:y2, x1:x2]
        balloon_alpha = pygame.surfarray.array_alpha(self.image)
        balloon_alpha = np.rot90(balloon_alpha, k=3)
        balloon_mask = (balloon_alpha > 10).astype(np.uint8) * 255
        
        if circle_crop.shape != balloon_mask.shape:
            balloon_mask = cv2.resize(balloon_mask, 
                                    (circle_crop.shape[1], circle_crop.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        
        intersection = cv2.bitwise_and(circle_crop, balloon_mask)
        overlap = np.count_nonzero(intersection)
        balloon_area = np.count_nonzero(balloon_mask)
        
        if balloon_area == 0:
            return
            
        overlap_ratio = overlap / balloon_area
        
        # Additional check: is there a detected circle center near balloon center?
        balloon_center_x = self.x + self.width // 2
        balloon_center_y = self.y + self.height // 2
        
        circle_nearby = False
        for cx, cy, cr in detected_circles:
            distance = np.sqrt((cx - balloon_center_x)**2 + (cy - balloon_center_y)**2)
            if distance < (cr + min(self.width, self.height) // 2):
                circle_nearby = True
                break
        
        # Pop if significant overlap AND a circle is detected nearby
        if overlap_ratio > 0.02 and circle_nearby:
            self.popped = True
            global score
            score += 1
            if pop_sound:
                pop_sound.play()

    def __init__(self):
        self.image = random.choice(BALLOON_IMAGES)
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.reset()

    def reset(self):
        self.x = random.randint(0, SCREEN_WIDTH - self.width)
        self.y = SCREEN_HEIGHT + random.randint(0, 300)
        self.speed = random.uniform(3.75, 6.0)
        self.popped = False

    def update(self):
        if not self.popped:
            self.y -= self.speed
            if self.y + self.height < 0:
                self.reset()

    def draw(self, win):
        if not self.popped:
            win.blit(self.image, (self.x, self.y))

# Initialize game variables
balloons = [Balloon() for _ in range(5)]
score = 0
start_time = pygame.time.get_ticks()
game_over = False
background_frame = None

# Main loop
running = True
while running:
    CLOCK.tick(FPS)
    ret, frame = cap.read()
    if not ret:
        print("Warning: Could not read frame")
        continue
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
    if background_frame is None:
        background_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background_frame = cv2.GaussianBlur(background_frame, (21, 21), 0)
        continue
    object_mask, detected_circles = get_circle_mask(frame, background_frame)
    
    if object_mask is not None:
        debug_view = cv2.cvtColor(object_mask, cv2.COLOR_GRAY2BGR)
        debug_combined = np.hstack((frame, debug_view))
        debug_combined = cv2.resize(debug_combined, (SCREEN_WIDTH, SCREEN_HEIGHT))
        cv2.imshow("Camera Feed (Press Q to Quit)", debug_combined)

    elapsed_time = (pygame.time.get_ticks() - start_time) / 1000
    time_left = max(0, GAME_DURATION - int(elapsed_time))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif game_over and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                score = 0
                start_time = pygame.time.get_ticks()
                game_over = False
                for b in balloons:
                    b.reset()
            elif event.key == pygame.K_ESCAPE:
                running = False

    if not game_over:
        # Draw background
        if background_img:
            WIN.blit(background_img, (0, 0))
        else:
            WIN.fill(WHITE)
        # Update and draw balloons
        for b in balloons:
            b.check_circle_occlusion(object_mask, detected_circles)
        for b in balloons:
            if b.popped:
                b.reset()
            b.update()
            b.draw(WIN)
        # Draw timer and score
        timer_text = FONT.render(f"Time Left: {time_left}s", True, BLACK)
        score_text = FONT.render(f"Score: {score}", True, BLACK)
        draw_text_with_bg(WIN, timer_text, 20, 20)
        draw_text_with_bg(WIN, score_text, 20, 60)
        if elapsed_time >= GAME_DURATION:
            game_over = True
    else:
        # Game over screen
        final_text = BIG_FONT.render("Game Over!", True, (200, 0, 0))
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
pygame.quit()
sys.exit()