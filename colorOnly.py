import cv2
import numpy as np
import os

input_video_path = 'nono.mp4'
output_folder = 'output_videos'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

base_name = os.path.splitext(os.path.basename(input_video_path))[0]
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = os.path.join(output_folder, f"{base_name}_coloronly_output.mp4")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
accumulator = np.zeros_like(prev_frame, dtype=np.float32)

decay = 0.2  # Decay factor for the accumulator

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.absdiff(prev_gray, gray)
    prev_gray = gray

    accumulator *= (1 - decay)
    threshold_value = 50
    _, motion_mask = cv2.threshold(flow, threshold_value, 255, cv2.THRESH_BINARY)
    motion_mask_3ch = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
    
    np.add(accumulator, frame, out=accumulator, where=(motion_mask_3ch == 255))

    lagged_frame = np.zeros_like(frame)
    lagged_frame[motion_mask_3ch == 255] = cv2.convertScaleAbs(accumulator)[motion_mask_3ch == 255]

    out.write(lagged_frame)
    cv2.imshow('Vision Lag Effect with Extended Trail', lagged_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
