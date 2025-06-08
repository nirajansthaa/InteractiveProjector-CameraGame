import pygame
import cv2
import sys
import os
import time
import numpy as np
import pyautogui
import json
import random
from screeninfo import get_monitors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.edge_detection import EdgeDetector

# Get monitor info for projector display
monitors = get_monitors()
if not monitors:
    print("Error: No monitors detected")
    sys.exit(1)

# Debug monitor info
print("Detected monitors:")
for i, monitor in enumerate(monitors):
    print(f"Monitor {i}: {monitor.width}x{monitor.height} at ({monitor.x}, {monitor.y})")
# Monitor 0: 1920x1080 at (0, 0)
# Monitor 1: 2040x1152 at (2880, 0)    
main_screen = monitors[0]
external_screen = monitors[1] if len(monitors) > 1 else monitors[0]
external_origin = (2880, 0)

os.environ['SDL_VIDEO_WINDOW_POS'] = f"{external_origin[0]},{external_origin[1]}"

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
    with open(filename, 'w') as f:
        json.dump(points, f)

def load_calibration_points(filename="calibration.json"):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def get_perspective_transform(src_points, screen_width=800, screen_height=600):
    dst_points = np.float32([
        [0, 0],
        [screen_width, 0],
        [screen_width, screen_height],
        [0, screen_height]
    ])
    src_points = np.float32(src_points)
    return cv2.getPerspectiveTransform(src_points, dst_points)

class Balloon:
    def __init__(self, image_path, screen_width, screen_height, other_balloons=None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.other_balloons = other_balloons or []
        self.image = pygame.image.load(image_path).convert_alpha()
        self.image = pygame.transform.scale(self.image, (420, 460))
        self.rect = self.image.get_rect()
        self.speed = random.randint(2, 5)
        self.alive = True
        self.reset()

    def reset(self):
        max_attempts = 50
        for _ in range(max_attempts):
            self.rect.x = random.randint(0, self.screen_width - self.rect.width)
            self.rect.y = self.screen_height + random.randint(50, 300)
            if not self.is_overlapping():
                return
        print("Warning: Could not find non-overlapping position")

    def is_overlapping(self):
        for other in self.other_balloons:
            if other is self:
                continue
            if self.rect.colliderect(other.rect):
                return True
        return False

    def move(self):
        if not self.alive:
            return
        self.rect.y -= self.speed
        if self.rect.bottom < 0:
            self.reset()

def main():
    pygame.init()
    # Game settings
    # screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((external_screen.width, external_screen.height))

    pygame.display.set_caption("Balloon Popping Game")
    
    
    # Initialize edge detector
    detector = EdgeDetector(low_threshold=100, high_threshold=200)
    
    # Load images
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    balloon_files = ["balloon1.png", "balloon2.png", "balloon3.png"]
    image_paths = [os.path.join(base_dir, "assets", file) for file in balloon_files]
    
    # Validate paths
    valid_paths = []
    for path in image_paths:
        if os.path.exists(path) and os.path.isfile(path):
            valid_paths.append(path)
        else:
            print(f"Error: Image file does not exist or is not a file: {path}")
    
    if not valid_paths:
        print("Error: No valid image files found. Exiting.")
        pygame.quit()
        return
    
    # Initialize balloons
    balloons = []
    actual_width, actual_height = screen.get_size()

    for path in valid_paths:
        balloon = Balloon(path, actual_width, actual_height, balloons)
        balloons.append(balloon)
        balloons[-1].other_balloons = balloons
    
    # Initialize camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera")
        pygame.quit()
        return
    
    # Load or perform calibration
    calibration_file = os.path.join(base_dir, "calibration.json")
    calibration_points = load_calibration_points(calibration_file)
    if calibration_points is None or len(calibration_points) != 4:
        print("Performing camera calibration...")
        calibration_points = get_calibration_points(cap)
        if calibration_points is None or len(calibration_points) != 4:
            print("Error: Calibration failed. Exiting.")
            cap.release()
            pygame.quit()
            return
        save_calibration_points(calibration_points, calibration_file)
        print(f"Calibration points saved to {calibration_file}")
    
    transform_matrix = get_perspective_transform(calibration_points, actual_width, actual_height)
    
    # Create OpenCV window
    cv2.namedWindow("Camera Edges", cv2.WINDOW_NORMAL)
    
    # Game loop
    clock = pygame.time.Clock()
    score = 0
    start_time = time.time()
    running = True
    
    while running and time.time() - start_time < 120:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    print("Recalibrating...")
                    calibration_points = get_calibration_points(cap)
                    if calibration_points and len(calibration_points) == 4:
                        save_calibration_points(calibration_points, calibration_file)
                        transform_matrix = get_perspective_transform(calibration_points, actual_width, actual_height)
                        print(f"New calibration points saved to {calibration_file}")
        
        # Update balloons
        for balloon in balloons:
            balloon.move()
        
        # Process camera feed
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera")
            break
        
        warped_frame = cv2.warpPerspective(frame, transform_matrix, (actual_width, actual_height))
        gray_frame = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        # Downscale to 800x600 for edge detection
        gray_frame = cv2.resize(gray_frame, (720, 480), interpolation=cv2.INTER_AREA)
        # Apply Gaussian blur to reduce noise
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        edges = detector.detect_edges_from_array(gray_frame)        
        
        cv2.imshow("Camera Edges", edges)
        
        # Draw game screen
        screen.fill((255, 255, 255))
        for balloon in balloons:
            if balloon.alive:
                screen.blit(balloon.image, balloon.rect)
        pygame.display.flip()
        
        cv2.waitKey(1)
        clock.tick(60)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print(f"Final score: {score}")

if __name__ == "__main__":
    main()