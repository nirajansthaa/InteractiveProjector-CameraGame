import numpy as np
from modules.config import SCREEN_WIDTH, SCREEN_HEIGHT

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top_points = sorted_by_y[:2]
    bottom_points = sorted_by_y[2:]
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    bottom_right, bottom_left = bottom_points[np.argsort(bottom_points[:, 0])[::-1]]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def validate_quadrilateral(points):
    if len(points) != 4:
        return False, "Need exactly 4 points"
    min_distance = 50
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if calculate_distance(points[i], points[j]) < min_distance:
                return False, f"Points {i+1} and {j+1} are too close together"
    ordered_points = order_points(points)
    top_width = calculate_distance(ordered_points[0], ordered_points[1])
    bottom_width = calculate_distance(ordered_points[3], ordered_points[2])
    left_height = calculate_distance(ordered_points[0], ordered_points[3])
    right_height = calculate_distance(ordered_points[1], ordered_points[2])
    avg_width = (top_width + bottom_width) / 2
    avg_height = (left_height + right_height) / 2
    if avg_width == 0 or avg_height == 0:
        return False, "Invalid dimensions"
    aspect_ratio = avg_width / avg_height
    expected_ratio = SCREEN_WIDTH / SCREEN_HEIGHT
    if not (expected_ratio * 0.7 <= aspect_ratio <= expected_ratio * 1.3):
        return False, f"Aspect ratio {aspect_ratio:.2f} doesn't match expected {expected_ratio:.2f}"
    return True, "Valid quadrilateral"