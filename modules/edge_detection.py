# detection/edge_detection.py
import cv2
import numpy as np

def load_and_detect_edges(image_paths, resize=(300, 300)):
    edge_results = []

    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Error: Could not load {path}")
            continue

        # Resize image
        image = cv2.resize(image, resize)

        # Extract RGB and alpha if present
        if image.shape[2] == 4:
            bgr = image[:, :, :3]
            alpha = image[:, :, 3]
        else:
            bgr = image
            alpha = None

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Mask with alpha to suppress transparent regions
        if alpha is not None:
            gray = cv2.bitwise_and(gray, gray, mask=alpha)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        edge_results.append({
            'original': bgr,
            'gray': gray,
            'edges': edges,
            'path': path
        })

    return edge_results