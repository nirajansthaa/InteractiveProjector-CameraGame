import cv2

def test_cameras(max_test=5):
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i} works. Press ESC to skip.")
                cv2.imshow(f"Camera {i}", frame)
                key = cv2.waitKey(0)
                if key == 27:  # ESC
                    cv2.destroyAllWindows()
                    cap.release()
                    return i
        cap.release()
    print("No valid camera found.")
    return -1

if __name__ == "__main__":
    working_id = test_cameras()
    print(f"Use CAMERA_ID = {working_id}")
