# detection/circle_detection.py
import cv2
import numpy as np

def detect_circles_from_camera(callback=None):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=100,
            param2=30,
            minRadius=10,
            maxRadius=60
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 3)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                
                if callback:
                    callback(x, y, r)  # e.g., check_collision(x, y)

        cv2.imshow("Circle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
