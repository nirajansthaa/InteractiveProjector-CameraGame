import cv2
import numpy as np
import time
import json
import os
import logging
from modules.config import *
from modules.utils import order_points, calculate_distance, validate_quadrilateral

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_calibration_points(cap, window_name="Calibration"):
    if not cap.isOpened():
        logger.error("Camera not accessible")
        return None
    points = []
    selected_point = -1
    dragging = False
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, selected_point, dragging
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, pt in enumerate(points):
                if calculate_distance([x, y], pt) < 15:
                    selected_point = i
                    dragging = True
                    return
            if len(points) < 4:
                points.append([x, y])
                selected_point = len(points) - 1
                logger.info(f"Point {len(points)} selected: ({x}, {y})")
        elif event == cv2.EVENT_MOUSEMOVE and dragging and selected_point >= 0:
            points[selected_point] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            selected_point = -1
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CAMERA_WIDTH, CAMERA_HEIGHT)
    cv2.setMouseCallback(window_name, mouse_callback)
    logger.info("=== CALIBRATION INSTRUCTIONS ===")
    logger.info("1. Click EXACTLY on the four corners of the projector screen in the camera view")
    logger.info("2. Use the grid to align points precisely")
    logger.info("3. Drag points to adjust")
    logger.info("4. Points will be ordered: top-left, top-right, bottom-right, bottom-left")
    logger.info("5. Press 'r' to reset, 'c' to confirm, 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Could not read frame during calibration")
            return None
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        for i in range(0, width, 50):
            cv2.line(display_frame, (i, 0), (i, height), (50, 50, 50), 1)
        for i in range(0, height, 50):
            cv2.line(display_frame, (0, i), (width, i), (50, 50, 50), 1)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        labels = ["TL", "TR", "BR", "BL"]
        for i, pt in enumerate(points):
            color = colors[i % 4]
            cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 8, color, -1)
            cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 12, (255, 255, 255), 2)
            cv2.putText(display_frame, labels[i], (int(pt[0]) - 8, int(pt[1]) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if len(points) >= 2:
            ordered_points = order_points(points) if len(points) == 4 else points
            for i in range(len(ordered_points)):
                if i < len(ordered_points) - 1:
                    cv2.line(display_frame, (int(ordered_points[i][0]), int(ordered_points[i][1])),
                             (int(ordered_points[i+1][0]), int(ordered_points[i+1][1])), (0, 255, 255), 2)
            if len(points) == 4:
                cv2.line(display_frame, (int(ordered_points[-1][0]), int(ordered_points[-1][1])),
                         (int(ordered_points[0][0]), int(ordered_points[0][1])), (0, 255, 255), 2)
        status_text = f"Points: {len(points)}/4"
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if len(points) == 4:
            is_valid, message = validate_quadrilateral(points)
            color = (0, 255, 0) if is_valid else (0, 0, 255)
            cv2.putText(display_frame, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if is_valid:
                cv2.putText(display_frame, "Press 'c' to confirm", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return None
        elif key == ord('r'):
            points = []
            logger.info("Points reset")
        elif key == ord('c'):
            if len(points) == 4:
                is_valid, message = validate_quadrilateral(points)
                if is_valid:
                    cv2.destroyWindow(window_name)
                    return order_points(points).tolist()

def save_calibration_points(points, segment_offsets, offset_x=0, offset_y=0, debug_offset_x=0, debug_offset_y=0, filename=CALIBRATION_FILE):
    try:
        calibration_data = {
            'points': points,
            'segment_offsets': segment_offsets,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'debug_offset_x': debug_offset_x,
            'debug_offset_y': debug_offset_y,
            'screen_width': SCREEN_WIDTH,
            'screen_height': SCREEN_HEIGHT,
            'timestamp': time.time(),
            'version': '2.2'
        }
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        logger.info(f"Calibration saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving calibration points: {e}")

def load_calibration_points(filename=CALIBRATION_FILE):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'points' in data:
            version = data.get('version', '1.0')
            if version != '2.2':
                logger.warning(f"Calibration file version {version} may be incompatible with version 2.2")
            if data.get('screen_width') != SCREEN_WIDTH or data.get('screen_height') != SCREEN_HEIGHT:
                logger.warning("Calibration was done for different screen dimensions")
                return None, [[(0, 0)] * GRID_COLS for _ in range(GRID_ROWS)], 0, 0, 0, 0
            segment_offsets = data.get('segment_offsets', [[(0, 0)] * GRID_COLS for _ in range(GRID_ROWS)])
            offset_x = data.get('offset_x', 0)
            offset_y = data.get('offset_y', 0)
            debug_offset_x = data.get('debug_offset_x', 0)
            debug_offset_y = data.get('debug_offset_y', 0)
            return data['points'], segment_offsets, offset_x, offset_y, debug_offset_x, debug_offset_y
        return None, [[(0, 0)] * GRID_COLS for _ in range(GRID_ROWS)], 0, 0, 0, 0
    except Exception as e:
        logger.error(f"Error loading calibration points: {e}")
        return None, [[(0, 0)] * GRID_COLS for _ in range(GRID_ROWS)], 0, 0, 0, 0

def get_perspective_transform(src_points, segment_offsets, offset_x=0, offset_y=0):
    dst_points = np.float32([
        [0 + offset_x, 0 + offset_y],
        [SCREEN_WIDTH-1 + offset_x, 0 + offset_y],
        [SCREEN_WIDTH-1 + offset_x, SCREEN_HEIGHT-1 + offset_y],
        [0 + offset_x, SCREEN_HEIGHT-1 + offset_y]
    ])
    src_points = np.float32(src_points)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    logger.info("=== CALIBRATION QUALITY ===")
    for i, (src, dst) in enumerate(zip(src_points, dst_points)):
        transformed = cv2.perspectiveTransform(np.array([[src]], dtype=np.float32), matrix)[0][0]
        error = np.linalg.norm(transformed - dst)
        logger.info(f"Corner {i+1}: Error = {error:.2f} pixels")
    return matrix

def test_calibration_accuracy(transform_matrix, points):
    dst_points = np.float32([
        [0, 0],
        [SCREEN_WIDTH-1, 0],
        [SCREEN_WIDTH-1, SCREEN_HEIGHT-1],
        [0, SCREEN_HEIGHT-1]
    ])
    logger.info("=== CALIBRATION ACCURACY TEST ===")
    for i, (src, dst) in enumerate(zip(points, dst_points)):
        transformed = cv2.perspectiveTransform(np.array([[src]], dtype=np.float32), transform_matrix)[0][0]
        error = np.linalg.norm(transformed - dst)
        logger.info(f"Corner {i+1}: Transformed = {transformed}, Expected = {dst}, Error = {error:.2f} pixels")