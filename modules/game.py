import cv2
import os
import numpy as np
from game_visual import Balloon
from circle_detection import detect_circles_in_frame
from hit_detection import HitDetector

def run_balloon_pop_game():
    """
    Run a game where balloons pop when hit by detected circles from the camera.
    Press 'q' or ESC to quit.
    """
    # Screen dimensions
    screen_width, screen_height = 800, 600

    # Balloon image paths (relative to project root)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    balloon_paths = [
        os.path.join(base_dir, '..', 'assets', 'balloon1.png'),
        os.path.join(base_dir, '..', 'assets', 'balloon2.png'),
        os.path.join(base_dir, '..', 'assets', 'balloon3.png')
    ]
    
    # Initialize balloons, skipping invalid images
    balloons = []
    
    for path in balloon_paths:
        try:
            balloons.append(Balloon(path, screen_width, screen_height))
            print(f"Loaded balloon image: {path}")
        except ValueError as e:
            print(f"Warning: {e}. Skipping this image.")

    if not balloons:
        print("Error: No valid balloon images loaded. Exiting.")
        return

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        cap.release()
        return
    
    # Initialize hit detector
    hit_detector = HitDetector()


    def hit_callback(x, y, r, balloon):
        # print(f"Balloon popped at ({x}, {y}) with radius {r}")
        # balloon.alive = False
        balloon.pop()

    # Game loop
    while True:
        ret, camera_frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Create game frame (camera feed as background)
        frame = camera_frame.copy()
        frame = cv2.resize(frame, (screen_width, screen_height))
        frame = cv2.flip(frame, 1)

        # Move and draw balloons
        for balloon in balloons:
            balloon.update()
            balloon.move()
            balloon.draw(frame, show_bbox=False)
        
        # Detect circles and draw them on the same frame
        circles, frame = detect_circles_in_frame(
            frame,
            min_radius=30,
            max_radius=50,
            min_dist=80,
            param1=120,
            param2=20,
            intensity_threshold=100
        )
        
        # Process detected circles for hits
        if circles is not None:
            hit_detector.process_detected_circles(circles, balloons, callback=hit_callback)
                
        # Show the combined frame
        cv2.imshow("Balloon Pop Game", frame)

        # Exit on 'q' or ESC
        key = cv2.waitKey(30) & 0xFF
        if key in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_balloon_pop_game()