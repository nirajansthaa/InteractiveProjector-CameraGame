import cv2
import numpy as np
import json
import os

class Calibration:
    def __init__(self, camera_id=1, save_path="calibration_points.json"):
        self.camera_id = camera_id
        self.save_path = save_path
        self.points = []
        self.frame = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"Point {len(self.points)} selected: ({x}, {y})")

    def capture_and_select_points(self):
        cap = cv2.VideoCapture(self.camera_id)
        print("Press 'c' to capture a calibration frame.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                cap.release()
                return None
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "Press 'c' to capture, 'q' to quit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Live Feed - Press 'c' to capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                self.frame = frame.copy()
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None

        cap.release()
        cv2.destroyWindow("Live Feed - Press 'c' to capture")

        if self.frame is None:
            print("Failed to capture frame.")
            return None

        cv2.namedWindow("Select 4 Corners")
        cv2.setMouseCallback("Select 4 Corners", self.mouse_callback)

        while True:
            disp = self.frame.copy()
            for i, p in enumerate(self.points):
                cv2.circle(disp, p, 5, (0, 0, 255), -1)
                cv2.putText(disp, f"P{i+1}", (p[0] + 10, p[1] + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if len(self.points) == 4:
                cv2.polylines(disp, [np.array(self.points)], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(disp, "Press 'q' to save or 'r' to reset", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Select 4 Corners", disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                self.points = []
                print("Reset points.")
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

        if len(self.points) == 4:
            self.save_points()
            return np.array(self.points, dtype=np.float32)
        else:
            print("4 points were not selected.")
            return None

    def save_points(self):
        with open(self.save_path, 'w') as f:
            json.dump(self.points, f)
        print(f"Saved points to {self.save_path}")

    def load_points(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                self.points = json.load(f)
            return np.array(self.points, dtype=np.float32)
        else:
            print("No saved calibration found.")
            return None