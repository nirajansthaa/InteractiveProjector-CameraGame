import cv2
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.edge_detection import load_and_detect_edges

def display_results():
    # Sample image paths (replace with actual image paths)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_paths = [
        os.path.join(base_dir, '..', 'assets', 'balloon1.png'),
        os.path.join(base_dir, '..', 'assets', 'balloon2.png'),
        os.path.join(base_dir, '..', 'assets', 'balloon3.png')
    ]

    # Call the edge detection function
    results = load_and_detect_edges(image_paths, resize=(300, 300))

    # Display results
    for result in results:
        cv2.imshow('Original Image', result['original'])
        cv2.imshow('Grayscale Image', result['gray'])
        cv2.imshow('Edges', result['edges'])
        print(f"Processed: {result['path']}")
        cv2.waitKey(0)  # Wait for key press to move to next image

    cv2.destroyAllWindows()

if __name__ == '__main__':
    display_results()