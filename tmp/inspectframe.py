import cv2, numpy as np

VIDEO = "/home/ted/so101-sim-project/data/episodes/episode_7.mp4"  # adjust path
OFFSET = 130.4

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(OFFSET * fps))
ret, frame = cap.read()
cap.release()

# Save the raw frame
cv2.imwrite("/tmp/ep7_frame.png", frame)

# Check what green blobs exist (LEFT half only)
h, w = frame.shape[:2]
left = frame[:, :w//2]
hsv = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)

# Current green range
mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Green blobs found (HSV 35-85): {len(cnts)}")
for c in cnts:
    M = cv2.moments(c)
    if M["m00"] > 0:
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        print(f"  blob area={cv2.contourArea(c):.0f} center=({cx},{cy})")

# Sample the actual HSV at the center of frame to see what's there
print(f"\nFrame shape: {frame.shape}")
print(f"Center pixel BGR: {frame[h//2, w//4]}")  # center of left half
print(f"Center pixel HSV: {cv2.cvtColor(frame[h//2:h//2+1, w//4:w//4+1], cv2.COLOR_BGR2HSV)[0,0]}")