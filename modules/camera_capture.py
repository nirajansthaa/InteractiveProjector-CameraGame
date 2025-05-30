# import cv2

# clicked_points = []

# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         clicked_points.append((x, y))
#         print(f"Clicked: {x}, {y}")

# def capture_camera_points():
#     cap = cv2.VideoCapture(1)
#     cv2.namedWindow("Camera View")
#     cv2.setMouseCallback("Camera View", mouse_callback)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         for point in clicked_points:
#             cv2.circle(frame, point, 5, (0, 0, 255), -1)
#         cv2.imshow("Camera View", frame)
#         if cv2.waitKey(1) & 0xFF == 27 or len(clicked_points) >= 4:
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     return clicked_points

import cv2

clicked_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Clicked: {x}, {y}")

def capture_camera_points(resolution=(1900, 1000)):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return []
    cv2.namedWindow("Camera View")
    cv2.setMouseCallback("Camera View", mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        # Resize and flip frame to match game loop
        frame = cv2.resize(frame, resolution)
        frame = cv2.flip(frame, 1)
        # Draw clicked points with numbers
        for i, point in enumerate(clicked_points):
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{i+1}", (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Camera View", frame)
        if cv2.waitKey(1) & 0xFF == 27 or len(clicked_points) >= 4:
            break
    cap.release()
    cv2.destroyAllWindows()
    return clicked_points