import cv2
import os

# === SETUP ===
video_path = r'C:\Users\ACER\gitClones\InteractiveProjector-CameraGame\datapreprocessing\data.mp4'  
output_folder = r'C:\Users\ACER\gitClones\InteractiveProjector-CameraGame\datapreprocessing\extracted_frames'       # Folder to save extracted frames

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video
cap = cv2.VideoCapture(video_path)

frame_count = 0

# === FRAME EXTRACTION LOOP ===
while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # No more frames, exit loop
    
    # Save each frame as a jpg image
    frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, frame)
    
    frame_count += 1

cap.release()
print(f"[INFO] Extracted {frame_count} frames to '{output_folder}'")