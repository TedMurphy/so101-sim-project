"""
isaac_grid_overlay.py
=====================
Take a SIM camera frame and overlay the ISAAC world coordinate grid
projected directly through the pinhole camera model — no H_SIM involved.

This answers: "where does Isaac *actually* project each (X,Y) world
point onto the SIM camera image?"

If that grid matches H_SIM, H_SIM is correctly calibrated.
If it doesn't, H_SIM itself is the source of the offset.

Usage:
    python isaac_grid_overlay.py <path_to_sim_frame.png> [output.png]

The SIM frame can be any PNG saved by the diagnostic scripts, e.g.
    ~/so101-sim-project/diag_cup_frames/frame_r0_c0_on.png
"""

import sys
import math
import numpy as np
import cv2

# ── Camera parameters (from replay_interactive_v2.py set_viewport_camera) ─────
CAM_POS   = np.array([-0.03, 0.32, 0.70])   # Isaac world XYZ metres
TILT_DEG  = 5.0                              # tilt toward robot (+Y tilt)
FOCAL_MM  = 18.14756
H_APERTURE_MM = 20.995   # horizontal sensor size
V_APERTURE_MM = 16.594   # vertical sensor size

# Image resolution (must match what the SIM camera outputs)
IMG_W = 640
IMG_H = 480

# ── Build camera intrinsics ────────────────────────────────────────────────────
# fx = focal_mm / sensor_width_mm * pixel_width
fx = FOCAL_MM / H_APERTURE_MM * IMG_W   # pixels
fy = FOCAL_MM / V_APERTURE_MM * IMG_H
cx = IMG_W / 2.0
cy = IMG_H / 2.0
K  = np.array([[fx, 0, cx],
               [0, fy, cy],
               [0,  0,  1]], dtype=np.float64)

# ── Build camera rotation ──────────────────────────────────────────────────────
# Same quaternion construction as replay_interactive_v2.py
tilt_rad = math.radians(TILT_DEG)
_cT, _sT = math.cos(tilt_rad / 2), math.sin(tilt_rad / 2)
# q_tilt: Ry(+tilt) — tilts toward robot
_w1, _x1, _y1, _z1 = _cT, 0.0, +_sT, 0.0
# q_base: Rz(180°) — flips image so image-right = Isaac +X
_w2, _x2, _y2, _z2 = 0.0, 0.0, 0.0, 1.0
# Combined quaternion q = q_tilt * q_base
qw = _w1*_w2 - _x1*_x2 - _y1*_y2 - _z1*_z2
qx = _w1*_x2 + _x1*_w2 + _y1*_z2 - _z1*_y2
qy = _w1*_y2 - _x1*_z2 + _y1*_w2 + _z1*_x2
qz = _w1*_z2 + _x1*_y2 - _y1*_x2 + _z1*_w2
q  = np.array([qw, qx, qy, qz])

def quat_to_rot(q):
    """Quaternion (w,x,y,z) → 3×3 rotation matrix (world→camera convention)."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])

# R_world_to_cam: rotates Isaac world vector into camera-frame vector
R_wc = quat_to_rot(q)
# But Isaac Sim uses a different camera convention than OpenCV.
# Isaac camera: +X right, +Y up, -Z forward (RHS, looking into scene)
# OpenCV camera: +X right, +Y down, +Z forward
# Adjust: flip Y and Z
R_isaac_to_cv = np.diag([1, -1, -1])  # OpenCV ↔ Isaac camera flip

R = R_isaac_to_cv @ R_wc   # final rotation: Isaac world → OpenCV camera

# ── World → pixel projection ───────────────────────────────────────────────────
def world_to_pixel(X, Y, Z=0.0):
    """Project Isaac world point (X,Y,Z) → (u,v) in image pixels.
    Returns None if point is behind the camera."""
    p_world  = np.array([X, Y, Z]) - CAM_POS
    p_cam    = R @ p_world             # camera frame (OpenCV convention)
    if p_cam[2] <= 0:
        return None                    # behind camera
    u = K[0,0] * p_cam[0] / p_cam[2] + cx
    v = K[1,1] * p_cam[1] / p_cam[2] + cy
    return float(u), float(v)

# ── H_SIM homography (for comparison grid) ────────────────────────────────────
_H_SIM = np.array([
    [ 1.26659037e-03, -1.15176118e-06, -4.16389850e-01],
    [-2.38682034e-06,  1.25586847e-03, -7.64706966e-02],
    [ 2.29476315e-04, -8.30288972e-06,  1.00000000e+00],
], dtype=np.float64)
_H_SIM_INV = np.linalg.inv(_H_SIM)

def world_to_pixel_hsim(wx, wy):
    """Project via H_SIM (the calibrated homography)."""
    p = _H_SIM_INV @ np.array([wx, wy, 1.0])
    return float(p[0]/p[2]), float(p[1]/p[2])

# ── Draw grid ─────────────────────────────────────────────────────────────────
def draw_grid(img, grid_x, grid_y, proj_fn, color, dot_r=3, line_thick=1,
              label=True, label_color=None):
    lc = label_color or color
    pts = {}
    for gy in grid_y:
        row = []
        for gx in grid_x:
            pix = proj_fn(gx, gy)
            pts[(gx, gy)] = pix
            if pix:
                row.append((int(round(pix[0])), int(round(pix[1]))))
        if line_thick > 0 and len(row) >= 2:
            for a, b in zip(row, row[1:]):
                cv2.line(img, a, b, color, line_thick, cv2.LINE_AA)
    for gx in grid_x:
        col = []
        for gy in grid_y:
            pix = pts.get((gx, gy))
            if pix:
                col.append((int(round(pix[0])), int(round(pix[1]))))
        if line_thick > 0 and len(col) >= 2:
            for a, b in zip(col, col[1:]):
                cv2.line(img, a, b, color, line_thick, cv2.LINE_AA)
    for (gx, gy), pix in pts.items():
        if pix:
            u, v = int(round(pix[0])), int(round(pix[1]))
            cv2.circle(img, (u, v), dot_r, color, -1, cv2.LINE_AA)
            if label and abs(gy - grid_y[0]) < 1e-6:
                cv2.putText(img, f"{gx:.2f}", (u-10, v-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, lc, 1, cv2.LINE_AA)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path.replace(".png", "_isaac_grid.png")

    img = cv2.imread(in_path)
    if img is None:
        print(f"Cannot read {in_path}"); sys.exit(1)
    h, w = img.shape[:2]
    print(f"Image: {w}×{h}")

    # Grid covering the mat
    gx = np.arange(-0.40, 0.36, 0.05)
    gy = np.arange(0.00,  0.42, 0.05)

    # ── Direct Isaac projection (CYAN) ────────────────────────────────────────
    draw_grid(img, gx, gy,
              lambda x, y: world_to_pixel(x, y, 0.0),
              color=(255, 220, 0),   # cyan
              dot_r=3, line_thick=1, label=False)

    # ── H_SIM homography projection (RED) ─────────────────────────────────────
    draw_grid(img, gx, gy,
              lambda x, y: world_to_pixel_hsim(x, y),
              color=(0, 80, 255),    # red
              dot_r=2, line_thick=1, label=False)

    # ── Legend ────────────────────────────────────────────────────────────────
    cv2.rectangle(img, (8, 8), (280, 60), (30,30,30), -1)
    cv2.circle(img,    (22, 25), 5, (255,220,0), -1)
    cv2.putText(img, "Direct Isaac pinhole projection", (32,29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,220,0), 1, cv2.LINE_AA)
    cv2.circle(img,    (22, 47), 5, (0,80,255), -1)
    cv2.putText(img, "H_SIM homography projection",    (32,51),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,80,255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, img)
    print(f"Saved → {out_path}")
    print()
    print("If CYAN and RED grids overlap: H_SIM correctly represents Isaac.")
    print("If they diverge: H_SIM is miscalibrated (offset IS in H_SIM).")
    print("The gap between them = the error we see in placement.")

if __name__ == "__main__":
    main()