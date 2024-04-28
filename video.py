import cv2
import numpy as np

cap = cv2.VideoCapture('cow.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

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
    
    # Update the accumulator using the motion mask as is without converting it to BGR
    cv2.accumulateWeighted(frame, accumulator, alpha, mask=motion_mask)
    
    # Normalize the accumulator to uint8 image
    lagged_frame = cv2.convertScaleAbs(accumulator)
    
    cv2.imshow('Vision Lag', lagged_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
