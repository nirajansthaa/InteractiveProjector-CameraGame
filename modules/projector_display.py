import cv2
import numpy as np

def show_calibration_pattern(screen_resolution=(1280, 720), points=4):
    img = np.zeros((screen_resolution[1], screen_resolution[0], 3), dtype=np.uint8)
    positions = [
        (100, 100),
        (screen_resolution[0]-100, 100),
        (screen_resolution[0]-100, screen_resolution[1]-100),
        (100, screen_resolution[1]-100)
    ]
    for i, pos in enumerate(positions):
        cv2.circle(img, pos, 10, (0, 255, 0), -1)
        cv2.putText(img, f"{i+1}", (pos[0]+10, pos[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Projector Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return positions
