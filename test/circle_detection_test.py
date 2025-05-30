import cv2
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.circle_detection import detect_circles_in_frame

def test_circle_detection():
    """
    Test circle detection on camera feed.
    Press 'q' or ESC to quit.
    """
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    def sample_callback(x, y, r):
        """Callback to print detected circle parameters."""
        # print(f"Detected circle: center=({x}, {y}), radius={r}")
        pass

    # Test loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Detect circles in the frame
        _, frame = detect_circles_in_frame(
            frame,
            callback=sample_callback,
            min_radius=30,
            max_radius=50,
            min_dist=80,
            param1=120,
            param2=20,
            intensity_threshold=100
        )

        # Display the frame
        cv2.imshow("Test Circle Detection", frame)

        # Exit on 'q' or ESC
        key = cv2.waitKey(30) & 0xFF
        if key in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_circle_detection()