import cv2
import sys

def test_camera_indices(max_index=10):
    """
    Test camera indices from 0 to max_index-1 and display feed for accessible cameras.
    Prints status for each index and shows index on the video feed.
    Press 'q' to close each camera window or 'Esc' to exit the program.
    """
    print("Testing camera indices...")
    print("Press 'q' to close a camera window or 'Esc' to exit the program.")

    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"Camera index {index}: Not accessible")
            cap.release()
            continue
        
        print(f"Camera index {index}: Accessible")
        print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        
        window_name = f"Camera Index {index}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Camera index {index}: Failed to read frame")
                break
            
            # Overlay index number on the frame
            cv2.putText(frame, f"Index: {index}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"Closed camera index {index}")
                break
            elif key == 27:  # Esc key
                print("Exiting program")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
        
        cap.release()
        cv2.destroyWindow(window_name)

    print("Finished testing camera indices")

if __name__ == "__main__":
    try:
        test_camera_indices(max_index=10)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        cv2.destroyAllWindows()
        print("All camera windows closed")