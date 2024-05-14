import cv2
import numpy as np
import os

input_video_path = 'car.mp4'
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
output_video_path = os.path.join(output_folder, f"{base_name}_sparse_output.mp4")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=15,
                      blockSize=7)

# Take the first frame and find corners in it
ret, old_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

mask = np.zeros_like(old_frame)
trail_history = np.zeros_like(old_frame)
refresh_counter = 0
refresh_rate = 30  
movement_threshold = 2 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    if p1 is None or len(p1) == 0:
        continue

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Filter points by movement threshold
    motion_vectors = np.linalg.norm(good_new - good_old, axis=1)
    significant_movement_indices = motion_vectors > movement_threshold
    good_new = good_new[significant_movement_indices]
    good_old = good_old[significant_movement_indices]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)  # Convert to integers
        trail_history = cv2.line(trail_history, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)
    
    # Apply decay to the trail
    trail_history = cv2.addWeighted(trail_history, 0.9, np.zeros_like(trail_history), 0.1, 0)

    img = cv2.add(frame, trail_history)

    out.write(img)
    cv2.imshow('Sparse Optical Flow with Trail Effect', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    
    # Periodically refresh the feature points
    refresh_counter += 1
    if refresh_counter % refresh_rate == 0 or len(p0) < 10:  
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        trail_history = np.zeros_like(old_frame)  

cap.release()
out.release()
cv2.destroyAllWindows()
