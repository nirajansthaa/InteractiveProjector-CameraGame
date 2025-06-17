import pygame
import pyautogui
import cv2
import numpy as np
import sys
import json
import os
import time
import logging
from modules.calibration_grid import *

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

# Load calibration data
calibration_points, segment_offsets = load_calibration_points()

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

# Create a 3x3 grid of segments to adjust offsets
GRID_ROWS = 3
GRID_COLS = 3
segment_width = SCREEN_WIDTH // GRID_COLS
segment_height = SCREEN_HEIGHT // GRID_ROWS

# Function to adjust offsets for a grid segment
def adjust_offset_for_segment(segment_x, segment_y, dx, dy):
    global segment_offsets
    offset_x, offset_y = segment_offsets[segment_y][segment_x]
    segment_offsets[segment_y][segment_x] = (offset_x + dx, offset_y + dy)

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

# Function to display the 3x3 grid with instructions
def display_grid_with_instructions():
    screen.fill((0, 0, 0))  # Clear screen
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            pygame.draw.rect(screen, (255, 255, 255), (col * segment_width, row * segment_height, segment_width, segment_height), 2)
    
    # Display current segment offset information
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            offset_text = f"Offset: {segment_offsets[row][col]}"
            offset_surface = font.render(offset_text, True, (255, 255, 255))
            screen.blit(offset_surface, (col * segment_width + 5, row * segment_height + 5))

    # Instructions
    instructions = [
        "Press arrow keys to adjust offsets for selected segment",
        "Press 'c' to confirm and save adjustments",
        "Press 'q' to quit"
    ]
    for i, line in enumerate(instructions):
        text_surface = font.render(line, True, (255, 255, 255))
        screen.blit(text_surface, (10, SCREEN_HEIGHT - 50 + (i * 30)))
    
    pygame.display.flip()

# Main loop
while running:
    clock.tick(FPS)

    # Display the 3x3 grid with instructions
    display_grid_with_instructions()

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
    for result in results:
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if (0 <= cx <= SCREEN_WIDTH and 0 <= cy <= SCREEN_HEIGHT and 
                    current_time - last_click_time >= CLICK_COOLDOWN):

                    # Determine which grid segment the detected point belongs to
                    segment_x = cx // segment_width
                    segment_y = cy // segment_height
                    
                    # Retrieve the offset for the specific grid segment
                    offset_x, offset_y = segment_offsets[segment_y][segment_x]
                    
                    # Apply the segment offset
                    screen_x = int(cx + offset_x + external_screen.x)
                    screen_y = int(cy + offset_y + external_screen.y)

                    # Move the mouse and click
                    pyautogui.moveTo(screen_x, screen_y)
                    pyautogui.click(button='right')

                    # Show visual feedback with the crack effect
                    crack_x = int(cx - crack_img.get_width() / 2 + offset_x)
                    crack_y = int(cy - crack_img.get_height() / 2 + offset_y)
                    cracks.append(CrackEffect(crack_x, crack_y))
                    last_click_time = current_time
    
    # Debug view
    debug_view = warped_frame.copy()
    cv2.imshow("Camera Feed", debug_view)

    # Handle events for user interaction
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_LEFT:
                adjust_offset_for_segment(0, 0, -5, 0)  # Adjust for segment (0,0)
            elif event.key == pygame.K_RIGHT:
                adjust_offset_for_segment(1, 0, 5, 0)  # Adjust for segment (1,0)
            elif event.key == pygame.K_UP:
                adjust_offset_for_segment(0, 1, 0, -5)  # Adjust for segment (0,1)
            elif event.key == pygame.K_DOWN:
                adjust_offset_for_segment(0, 2, 0, 5)  # Adjust for segment (0,2)
            elif event.key == pygame.K_c:
                save_calibration_points(calibration_points, segment_offsets)
                print("Calibration saved!")
            elif event.key == pygame.K_f:
                show_debug_overlay = not show_debug_overlay

    # Render the offsets and debug overlays
    if show_debug_overlay:
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    pygame.draw.circle(screen, (255, 0, 0), (int(cx), int(cy)), 5)
                    pygame.draw.rect(screen, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)
    
    # Final screen update
    pygame.display.flip()

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()
