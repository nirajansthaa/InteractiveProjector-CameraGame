import pygame
import cv2
import sys
import os
import time
import numpy as np
import json
import random
from screeninfo import get_monitors
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
# Monitor 1: 1360x768 at (1920, 0)    
main_screen = monitors[0]
external_screen = monitors[1] if len(monitors) > 1 else monitors[0]
external_origin = (1920, 0)

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

class OcclusionDetector:
    def __init__(self, sensitivity=30, min_contour_area=500):
        self.background = None
        self.sensitivity = sensitivity
        self.min_contour_area = min_contour_area
        self.frame_count = 0
        self.background_update_rate = 30  # Update background every N frames
        
    def set_background(self, frame):
        """Set the background frame for comparison"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.background = cv2.GaussianBlur(gray, (21, 21), 0).astype(np.float32)
        
    def update_background(self, frame, learning_rate=0.01):
        """Gradually update background to handle lighting changes"""
        if self.background is None:
            self.set_background(frame)
            return
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0).astype(np.float32)
        
        # Gradually update background
        cv2.accumulateWeighted(gray_blur, self.background, learning_rate)
    
    def detect_occlusions(self, frame):
        """Detect areas where occlusion occurs"""
        if self.background is None:
            self.set_background(frame)
            return []
            
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Compute absolute difference between background and current frame
        diff = cv2.absdiff(self.background.astype(np.uint8), gray_blur)
        
        # Threshold the difference
        thresh = cv2.threshold(diff, self.sensitivity, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate to fill gaps
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and return centers
        occlusion_points = []
        for contour in contours:
            if cv2.contourArea(contour) > self.min_contour_area:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if radius > 5:  # Ignore very small circles
                    occlusion_points.append((int(x), int(y), int(radius)))
                    
        return occlusion_points, thresh

class Balloon:
    def __init__(self, image_path, screen_width, screen_height, other_balloons=None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.other_balloons = other_balloons or []
        self.image = pygame.image.load(image_path).convert_alpha()
        self.image = pygame.transform.scale(self.image, (420, 460))
        self.rect = self.image.get_rect()
        self.speed = random.randint(1, 3)
        self.alive = True
        self.hit_cooldown = 0 # Prevent multiple hits
        self.reset()

    def reset(self):
        max_attempts = 50
        for _ in range(max_attempts):
            self.rect.x = random.randint(0, self.screen_width - self.rect.width)
            self.rect.y = self.screen_height + random.randint(50, 300)
            if not self.is_overlapping():
                self.alive = True
                self.hit_cooldown = 60 # Set invulnerability for 1 second
                return
        # If we can't find space, place it anyway
        self.rect.x = random.randint(0, self.screen_width - self.rect.width)
        self.rect.y = self.screen_height + random.randint(50, 300)
        self.alive = True
        self.hit_cooldown = 60

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
            
        if self.hit_cooldown > 0:
            self.hit_cooldown -= 1
            
        self.rect.y -= self.speed
        if self.rect.bottom < 0:
            self.reset()

    def check_occlusion_hit(self, occlusion_points):
        """Check if any occlusion point overlaps with this balloon"""
        if not self.alive or self.hit_cooldown > 0:
            return False
            
        balloon_center = self.rect.center
        balloon_radius = min(self.rect.width, self.rect.height) // 2
        
        for occ_x, occ_y, occ_radius in occlusion_points:
            # Check if occlusion overlaps with balloon
            distance = ((balloon_center[0] - occ_x) ** 2 + (balloon_center[1] - occ_y) ** 2) ** 0.5
            if distance < balloon_radius + occ_radius:
                return True
        return False
    
def main():
    pygame.init()
    # Game settings
    # screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((external_screen.width, external_screen.height))

    pygame.display.set_caption("Balloon Popping Game")
    
    
    # Initialize edge detector
    detector = EdgeDetector(low_threshold=100, high_threshold=200)
    
    # Initialize occlusion detector
    occlusion_detector = OcclusionDetector(sensitivity=25, min_contour_area=300)

    # Load images
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
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

    for i in range(3):  # 3 balloons total
        path = valid_paths[i % len(valid_paths)]
        balloon = Balloon(path, actual_width, actual_height, balloons)
        balloons.append(balloon)
        balloon.other_balloons = balloons
    
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
    background_set = False
    frame_count = 0

    # Font for score display
    font = pygame.font.Font(None, 72)
    
    print("Game starting! Wait 3 seconds for background calibration...")
    print("Controls: Q to quit, R to recalibrate, SPACE to reset background")

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

        frame_count += 1

        # Set background after a few frames to let camera adjust
        if not background_set and frame_count > 90:  # 3 seconds at 30fps
            occlusion_detector.set_background(warped_frame)
            background_set = True
            print("Background set! Start playing!")
        
        # Update background periodically
        if background_set and frame_count % 30 == 0:
            occlusion_detector.update_background(warped_frame, learning_rate=0.005)
        
        # Detect occlusions
        occlusion_points = []
        diff_image = None
        if background_set:
            occlusion_points, diff_image = occlusion_detector.detect_occlusions(warped_frame)
            
            # Check for balloon hits
            for balloon in balloons:
                if balloon.check_occlusion_hit(occlusion_points):
                    balloon.alive = False
                    balloon.hit_cooldown = 30  # 0.5 second cooldown
                    score += 1
                    print(f"Balloon popped! Score: {score}")
                if not balloon.alive and balloon.hit_cooldown <= 0:
                    balloon.reset()

        # Draw occlusion points on camera feed for debugging
        debug_frame = warped_frame.copy()
        # for occ_x, occ_y, occ_w, occ_h in occlusion_points:
        #     cv2.rectangle(debug_frame, (occ_x - occ_w//2, occ_y - occ_h//2), 
        #                  (occ_x + occ_w//2, occ_y + occ_h//2), (0, 255, 0), 2)
        #     cv2.circle(debug_frame, (occ_x, occ_y), 5, (0, 0, 255), -1)

        # Show camera feeds
        cv2.imshow("Camera Feed", debug_frame)
        # if diff_image is not None:
        #     cv2.imshow("Occlusion Detection", diff_image)

        # Draw game screen
        screen.fill((255, 255, 255))  # White background

        # Draw balloons
        for balloon in balloons:
            if balloon.alive:
                screen.blit(balloon.image, balloon.rect)
        
        # Draw score
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_text, (50, 50))
        
        # Draw time remaining
        time_left = max(0, 120 - (time.time() - start_time))
        time_text = font.render(f"Time: {int(time_left)}s", True, (255, 255, 255))
        screen.blit(time_text, (50, 130))

        # Show status
        if not background_set:
            status_text = font.render("Calibrating background...", True, (255, 0, 0))
            screen.blit(status_text, (actual_width//2 - 200, actual_height//2))
        
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