import cv2
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.game_visual import Balloon

def run_game():
    """
    Run the balloon game with a single window displaying moving balloons.
    Press 'q' or ESC to quit.
    """
    # Balloon image paths (relative to project root)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    balloon_paths = [
        os.path.join(base_dir, '..', 'assets', 'balloon1.png'),
        os.path.join(base_dir, '..', 'assets', 'balloon2.png'),
        os.path.join(base_dir, '..', 'assets', 'balloon3.png')
    ]
    screen_width, screen_height = 800, 600

    # Initialize balloons, skipping invalid images
    balloons = []
    for path in balloon_paths:
        try:
            balloons.append(Balloon(path, screen_width, screen_height))
            # print(f"Loaded balloon image: {path}")
        except ValueError as e:
            print(f"Warning: {e}. Skipping this image.")

    if not balloons:
        print("Error: No valid balloon images loaded. Exiting.")
        return

    # Game loop
    while True:
        frame = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255  # White background

        for balloon in balloons:
            balloon.move()
            balloon.draw(frame, show_bbox=False)  # Draw without bounding box

        cv2.imshow("Balloon Game", frame)

        # Exit on 'q' or ESC
        key = cv2.waitKey(30) & 0xFF
        if key in [ord('q'), 27]:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_game()