#!/usr/bin/env python3
"""Detect cylinder and target positions from pre-extracted episode frame PNGs."""
import cv2
import numpy as np
import os
import json

CAMERA_X  = 0.20
CAMERA_Y  = 0.005
CAMERA_Z  = 0.602
TABLE_Z   = 0.032
FOV_H_DEG = 69.0
IMG_W, IMG_H = 640, 480

dist_to_table = CAMERA_Z - TABLE_Z
pixel_size = (2 * dist_to_table * np.tan(np.deg2rad(FOV_H_DEG / 2))) / IMG_W
print(f"pixel_size = {pixel_size*1000:.3f} mm/px")

CYLINDER_Z = TABLE_Z - 0.012 + 0.06 / 2
TARGET_Z   = TABLE_Z - 0.012 + 0.05 / 2

# Map episode -> pre-extracted frame PNG
EP_FRAMES = {
    0:  os.path.expanduser('~/so101-sim-project/ep_frames/ep0_frame92.png'),
    1:  os.path.expanduser('~/so101-sim-project/ep_frames/ep1_frame1084.png'),
    2:  os.path.expanduser('~/so101-sim-project/ep_frames/ep2_frame1518.png'),
    3:  os.path.expanduser('~/so101-sim-project/ep_frames/ep3_frame1988.png'),
    4:  os.path.expanduser('~/so101-sim-project/ep_frames/ep4_frame2488.png'),
}

def pixel_to_world(px, py):
    world_x = CAMERA_X + (IMG_H / 2 - py) * pixel_size
    world_y = CAMERA_Y + (px - IMG_W / 2) * pixel_size
    return round(world_x, 4), round(world_y, 4)

def detect_blob(frame_bgr, lower_hsv, upper_hsv, min_area=300):
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    mask = cv2.erode(mask,  None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    m = cv2.moments(mask)
    if m['m00'] > min_area:
        return int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])
    return None

def detect_green(frame_bgr):
    return detect_blob(frame_bgr, [40, 60, 60], [90, 255, 255])

def detect_yellow(frame_bgr):
    return detect_blob(frame_bgr, [15, 100, 100], [40, 255, 255])

positions = {}
debug_dir = os.path.expanduser('~/so101-sim-project/ep_frames')

for ep, png_path in sorted(EP_FRAMES.items()):
    if not os.path.exists(png_path):
        print(f'[SKIP] ep {ep}: {png_path} not found')
        continue

    frame = cv2.imread(png_path)
    if frame is None:
        print(f'[ERROR] ep {ep}: could not read {png_path}')
        continue

    debug = frame.copy()
    cyl_px = detect_green(frame)
    tgt_px = detect_yellow(frame)
    cyl_world = tgt_world = None

    if cyl_px:
        wx, wy = pixel_to_world(*cyl_px)
        cyl_world = [wx, wy, round(CYLINDER_Z, 4)]
        cv2.circle(debug, cyl_px, 10, (0, 255, 0), 2)
        cv2.putText(debug, f"cyl ({wx:.3f},{wy:.3f})", (cyl_px[0]+12, cyl_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        print(f'ep {ep}  cylinder  px={cyl_px}  world=({wx:.4f}, {wy:.4f}, {CYLINDER_Z:.4f})')
    else:
        print(f'ep {ep}  [WARN] cylinder not detected')

    if tgt_px:
        wx, wy = pixel_to_world(*tgt_px)
        tgt_world = [wx, wy, round(TARGET_Z, 4)]
        cv2.circle(debug, tgt_px, 10, (0, 200, 255), 2)
        cv2.putText(debug, f"tgt ({wx:.3f},{wy:.3f})", (tgt_px[0]+12, tgt_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)
        print(f'ep {ep}  target    px={tgt_px}  world=({wx:.4f}, {wy:.4f}, {TARGET_Z:.4f})')
    else:
        print(f'ep {ep}  [WARN] target not detected')

    out_img = os.path.join(debug_dir, f'ep{ep}_detect.png')
    cv2.imwrite(out_img, debug)
    print(f'       debug image -> {out_img}')
    positions[ep] = {'cylinder': cyl_world, 'target': tgt_world}

out_json = os.path.expanduser('~/so101-sim-project/episode_positions.json')
with open(out_json, 'w') as f:
    json.dump(positions, f, indent=2)
print(f'\nSaved -> {out_json}')
print(json.dumps(positions, indent=2))
