import cv2
import numpy as np
import os

input_video_path = 'rain.mp4'

output_folder = 'output_videos'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Extract the base name of the input video
base_name = os.path.splitext(os.path.basename(input_video_path))[0]

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to compress the frames
output_video_path = os.path.join(output_folder, f"{base_name}_output.mp4")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Prepare a frame accumulator
accumulator = np.zeros_like(prev_frame, dtype=np.float32)

# Frame blending factor
alpha = 0.5

while True:

    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow or motion between frames
    flow = cv2.absdiff(prev_gray, gray)
    prev_gray = gray
    
    # Threshold to detect significant changes
    _, motion_mask = cv2.threshold(flow, 25, 255, cv2.THRESH_BINARY)
    
    # Create a motion mask for blending (make it 3-channel)
    motion_mask_3ch = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

    cv2.accumulateWeighted(frame, accumulator, alpha, mask=motion_mask)
    
    # Normalize the accumulator to uint8 image
    lagged_frame = cv2.convertScaleAbs(accumulator)
    
    # Blend the lagged frame with the original frame based on the motion mask
    blended_frame = np.where(motion_mask_3ch == 255, lagged_frame, frame)

    # Write the processed frame into the file
    out.write(blended_frame)

    # Show the result
    cv2.imshow('Vision Lag Effect', blended_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Save the output video
cv2.destroyAllWindows()
