import cv2
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.edge_detection import EdgeDetector

def main():
    # Initialize edge detector
    detector = EdgeDetector(low_threshold=100, high_threshold=200)
    
    # Input paths
    input_image = "input.jpg"  # Replace with your image path
    # Load assets
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    balloon_files = ["balloon1.png", "balloon2.png", "balloon3.png", 'balloon4.png']
    for file in balloon_files:
        path = os.path.join(base_dir, "assets", file)
        try:
            # Detect edges
            edges = detector.detect_edges(path)
            
            # Display results
            original = cv2.imread(path)
            cv2.imshow("Original Image", original)
            cv2.imshow("Edges", edges)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            
            # Test threshold adjustment
            detector.adjust_thresholds(50, 150)
            edges_new = detector.detect_edges(path)
            cv2.imshow("Edges with Adjusted Thresholds", edges_new)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()