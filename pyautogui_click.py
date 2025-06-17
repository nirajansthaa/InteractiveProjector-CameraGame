import pygame
import pyautogui
import cv2
import numpy as np
import sys
import time
import logging
import os
from modules.config import *
from modules.utils import validate_quadrilateral
from modules.calibration_grid import get_calibration_points, save_calibration_points, load_calibration_points, get_perspective_transform, test_calibration_accuracy
from ultralytics import YOLO
from screeninfo import get_monitors

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress Ultralytics logging
os.environ["YOLO_VERBOSE"] = "False"
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

# Suppress OpenCV logging
cv2.setLogLevel(0)

# Initialize Pygame
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
pygame.init()
monitors = get_monitors()
if len(monitors) < 2:
    logger.error("External monitor (projector) not detected")
    pygame.quit()
    sys.exit(1)
external_screen = monitors[1]
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{external_screen.x},{external_screen.y}"
screen = pygame.display.set_mode((external_screen.width, external_screen.height))
pygame.display.set_caption("Green Ball Crack Effect")
clock = pygame.time.Clock()

# Load crack image
try:
    crack_img = pygame.image.load(CRACK_IMAGE_PATH).convert_alpha()
    crack_img = pygame.transform.scale(crack_img, (260, 150))
except pygame.error as e:
    logger.error(f"Error loading crack image: {e}")
    pygame.quit()
    sys.exit(1)

# Initialize YOLO model
try:
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    model = YOLO(MODEL_PATH, task="detect", verbose=False)
    sys.stdout = original_stdout
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    pygame.quit()
    sys.exit(1)

# Initialize camera
try:
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise Exception("Could not open camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
except Exception as e:
    logger.error(f"Camera initialization failed: {e}")
    pygame.quit()
    sys.exit(1)

# Load or perform calibration
calibration_points, segment_offsets, offset_x, offset_y, debug_offset_x, debug_offset_y = load_calibration_points()
transform_matrix = None
if calibration_points and len(calibration_points) == 4:
    is_valid, message = validate_quadrilateral(calibration_points)
    if is_valid:
        logger.info(f"Loading existing calibration with homography offset ({offset_x}, {offset_y}) and debug offset ({debug_offset_x}, {debug_offset_y})")
        transform_matrix = get_perspective_transform(calibration_points, segment_offsets, offset_x, offset_y)
        test_calibration_accuracy(transform_matrix, calibration_points)
    else:
        logger.warning(f"Invalid calibration points: {message}")
        calibration_points = None
if not calibration_points or len(calibration_points) != 4:
    logger.info("Performing camera calibration...")
    calibration_points = get_calibration_points(cap)
    offset_x, offset_y = 0, 0
    debug_offset_x, debug_offset_y = 280, 125
    if calibration_points and len(calibration_points) == 4:
        save_calibration_points(calibration_points, segment_offsets, offset_x, offset_y, debug_offset_x, debug_offset_y)
        transform_matrix = get_perspective_transform(calibration_points, segment_offsets, offset_x, offset_y)
        test_calibration_accuracy(transform_matrix, calibration_points)
    else:
        logger.error("Calibration failed")
        cap.release()
        pygame.quit()
        sys.exit(1)

# Initialize inverse transform matrix
inv_transform_matrix = np.linalg.inv(transform_matrix)

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
selected_segment = [0, 0]
font = pygame.font.SysFont(None, 36)

def adjust_offset_for_segment(segment_x, segment_y, dx, dy):
    global segment_offsets
    offset_x, offset_y = segment_offsets[segment_y][segment_x]
    segment_offsets[segment_y][segment_x] = (offset_x + dx, offset_y + dy)
    logger.info(f"Adjusted segment ({segment_x}, {segment_y}) offset to {segment_offsets[segment_y][segment_x]}")

def update_transform_matrix():
    global transform_matrix, inv_transform_matrix
    transform_matrix = get_perspective_transform(calibration_points, segment_offsets, offset_x, offset_y)
    inv_transform_matrix = np.linalg.inv(transform_matrix)
    logger.info("Transform matrix updated")

def display_grid_with_instructions():
    screen.fill((0, 0, 0))
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            color = (255, 255, 0) if [col, row] == selected_segment else (255, 255, 255)
            pygame.draw.rect(screen, color, (col * SEGMENT_WIDTH, row * SEGMENT_HEIGHT, SEGMENT_WIDTH, SEGMENT_HEIGHT), 2)
            offset_text = f"Offset: {segment_offsets[row][col]}"
            offset_surface = font.render(offset_text, True, (255, 255, 255))
            screen.blit(offset_surface, (col * SEGMENT_WIDTH + 5, row * SEGMENT_HEIGHT + 5))
    instructions = [
        "Use 1/2 to move left/right, 3/4 to move up/down",
        "Arrow keys to adjust offsets",
        "Press 'c' to confirm, 'q' to quit"
    ]
    for i, line in enumerate(instructions):
        text_surface = font.render(line, True, (255, 255, 255))
        screen.blit(text_surface, (10, SCREEN_HEIGHT - 50 + i * 30))
    pygame.display.flip()

# Main loop
while running:
    clock.tick(FPS)
    ret, frame = cap.read()
    if not ret:
        logger.warning("Could not read frame, retrying...")
        time.sleep(0.1)
        continue
    warped_frame = cv2.warpPerspective(frame, transform_matrix, (SCREEN_WIDTH, SCREEN_HEIGHT))
    results = model.predict(warped_frame, imgsz=640, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device="cpu", verbose=False)
    current_time = time.time()
    for result in results:
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if (0 <= cx <= SCREEN_WIDTH and 0 <= cy <= SCREEN_HEIGHT and 
                    current_time - last_click_time >= CLICK_COOLDOWN):
                    segment_x = int(cx // SEGMENT_WIDTH)
                    segment_y = int(cy // SEGMENT_HEIGHT)
                    segment_offset_x, segment_offset_y = segment_offsets[segment_y][segment_x]
                    screen_x = int(cx + debug_offset_x + segment_offset_x + external_screen.x)
                    screen_y = int(cy + debug_offset_y + segment_offset_y + external_screen.y)
                    try:
                        pyautogui.moveTo(screen_x, screen_y)
                        pyautogui.click(button='right')
                    except Exception as e:
                        logger.error(f"Mouse click failed: {e}")
                    crack_x = int(cx - crack_img.get_width() / 2 + debug_offset_x + segment_offset_x)
                    crack_y = int(cy - crack_img.get_height() / 2 + debug_offset_y + segment_offset_y)
                    cracks.append(CrackEffect(crack_x, crack_y))
                    last_click_time = current_time
    debug_view = warped_frame.copy()
    roi_points = np.float32([[0, 0], [SCREEN_WIDTH-1, 0], [SCREEN_WIDTH-1, SCREEN_HEIGHT-1], [0, SCREEN_HEIGHT-1]])
    roi_points = roi_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(debug_view, [roi_points], True, (255, 0, 0), 2)
    for result in results:
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) / 2 + debug_offset_x
                cy = (y1 + y2) / 2 + debug_offset_y
                confidence = float(box.conf[0])
                cv2.rectangle(debug_view, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(debug_view, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.putText(debug_view, f"Green Ball: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(debug_view, f"Screen: {SCREEN_WIDTH}x{SCREEN_HEIGHT}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_view, f"Homography Offset: ({offset_x}, {offset_y})", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_view, f"Debug Offset: ({debug_offset_x}, {debug_offset_y})", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Camera Feed", debug_view)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_c:
                logger.info("Starting recalibration...")
                calibration_points = get_calibration_points(cap)
                if calibration_points and len(calibration_points) == 4:
                    offset_x, offset_y = 0, 0
                    debug_offset_x, debug_offset_y = 280, 125
                    save_calibration_points(calibration_points, segment_offsets, offset_x, offset_y, debug_offset_x, debug_offset_y)
                    transform_matrix = get_perspective_transform(calibration_points, segment_offsets, offset_x, offset_y)
                    inv_transform_matrix = np.linalg.inv(transform_matrix)
                    test_calibration_accuracy(transform_matrix, calibration_points)
            elif event.key == pygame.K_f:
                show_debug_overlay = not show_debug_overlay
                if show_debug_overlay:
                    selected_segment = [0, 0]
                    display_grid_with_instructions()
                logger.info(f"Debug overlay: {'ON' if show_debug_overlay else 'OFF'}")
            elif show_debug_overlay:
                if event.key == pygame.K_1:
                    selected_segment[0] = (selected_segment[0] - 1) % GRID_COLS
                    display_grid_with_instructions()
                elif event.key == pygame.K_2:
                    selected_segment[0] = (selected_segment[0] + 1) % GRID_COLS
                    display_grid_with_instructions()
                elif event.key == pygame.K_3:
                    selected_segment[1] = (selected_segment[1] - 1) % GRID_ROWS
                    display_grid_with_instructions()
                elif event.key == pygame.K_4:
                    selected_segment[1] = (selected_segment[1] + 1) % GRID_ROWS
                    display_grid_with_instructions()
                elif event.key == pygame.K_LEFT:
                    adjust_offset_for_segment(selected_segment[0], selected_segment[1], -5, 0)
                    update_transform_matrix()
                    display_grid_with_instructions()
                elif event.key == pygame.K_RIGHT:
                    adjust_offset_for_segment(selected_segment[0], selected_segment[1], 5, 0)
                    update_transform_matrix()
                    display_grid_with_instructions()
                elif event.key == pygame.K_UP:
                    adjust_offset_for_segment(selected_segment[0], selected_segment[1], 0, -5)
                    update_transform_matrix()
                    display_grid_with_instructions()
                elif event.key == pygame.K_DOWN:
                    adjust_offset_for_segment(selected_segment[0], selected_segment[1], 0, 5)
                    update_transform_matrix()
                    display_grid_with_instructions()
            elif event.key == pygame.K_w:
                debug_offset_y -= 5
                logger.info(f"Adjusted debug offset to ({debug_offset_x}, {debug_offset_y})")
            elif event.key == pygame.K_s:
                debug_offset_y += 5
                logger.info(f"Adjusted debug offset to ({debug_offset_x}, {debug_offset_y})")
            elif event.key == pygame.K_a:
                debug_offset_x -= 5
                logger.info(f"Adjusted debug offset to ({debug_offset_x}, {debug_offset_y})")
            elif event.key == pygame.K_d:
                debug_offset_x += 5
                logger.info(f"Adjusted debug offset to ({debug_offset_x}, {debug_offset_y})")
            elif event.key == pygame.K_p:
                save_calibration_points(calibration_points, segment_offsets, offset_x, offset_y, debug_offset_x, debug_offset_y)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if current_time - last_click_time >= CLICK_COOLDOWN:
                mx, my = event.pos
                point = np.float32([[[mx, my]]])
                warped_point = cv2.perspectiveTransform(point, inv_transform_matrix)[0][0]
                cx, cy = warped_point
                if 0 <= cx <= SCREEN_WIDTH and 0 <= cy <= SCREEN_HEIGHT:
                    crack_x = int(cx - crack_img.get_width() / 2)
                    crack_y = int(cy - crack_img.get_height() / 2)
                    cracks.append(CrackEffect(crack_x, crack_y))
                    last_click_time = current_time
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
    offset_text = font.render(f"Homography Offset: ({offset_x}, {offset_y})", True, (255, 255, 255))
    debug_offset_text = font.render(f"Debug Offset: ({debug_offset_x}, {debug_offset_y})", True, (255, 255, 255))
    screen.blit(offset_text, (10, 10))
    screen.blit(debug_offset_text, (10, 40))
    pygame.display.flip()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()