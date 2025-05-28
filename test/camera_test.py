import cv2

# Set the camera index to 1 (external camera)
camera_index = 1

# Initialize the camera
cap = cv2.VideoCapture(camera_index)

# Set optional resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if camera opened successfully
if not cap.isOpened():
    print(f"❌ Could not open camera with index {camera_index}")
    exit()

print(f"✅ Camera {camera_index} opened successfully. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break
    
    
    cv2.imshow(f"Camera {camera_index} Feed", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        print("🔚 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
