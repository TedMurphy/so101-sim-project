#!/usr/bin/env python3
"""
pick_grid_points.py
--------------------
Click on any debug_real_ep*_cylinder.png image to find the Isaac cm
coordinates that correspond to that pixel.

Usage:
    python pick_grid_points.py path/to/debug_real_ep000_cylinder.png

Left-click  → prints Isaac cm coords + real-world metres for that pixel
Right-click → quit

The math (inverse of save_annotated_real back-projection):
    real-world (metres)  = H_REAL @ [px, py, 1]   (homogeneous divide)
    isaac_cm             = (-real_x * 100,  real_y * 100)
"""

import sys
import cv2
import numpy as np

# ── paste your H_REAL matrix here ────────────────────────────────────────────
H_REAL = np.array([
    [+1.15486834e-03, +7.69862523e-06, -3.50418283e-01],
    [+1.82245483e-06, +1.20933059e-03, -7.59824916e-02],
    [-1.04645431e-04, +2.21165457e-04, +1.00000000e+00],
], dtype=np.float64)
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_isaac_cm(px: int, py: int) -> tuple[float, float]:
    """Return (ix_cm, iy_cm) in Isaac coordinates for a real-image pixel."""
    v = H_REAL @ np.array([px, py, 1.0], dtype=np.float64)
    real_x = v[0] / v[2]
    real_y = v[1] / v[2]
    ix_cm = -real_x * 100.0
    iy_cm =  real_y * 100.0
    return ix_cm, iy_cm


click_count = [0]

def on_mouse(event, x, y, flags, param):
    img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        click_count[0] += 1
        ix_cm, iy_cm = pixel_to_isaac_cm(x, y)
        print(f"  Click {click_count[0]:>2d}  pixel=({x:4d},{y:4d})  →  "
              f"Isaac cm = ({ix_cm:+.2f}, {iy_cm:.2f})")
        # draw a temporary marker so you can see where you clicked
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.circle(img, (x, y), 6, (255, 255, 255), 1)
        cv2.imshow("Pick grid points  (left-click = coord, right-click = quit)", img)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print("Usage: python pick_grid_points.py <path_to_real_image.png>")
        sys.exit(1)

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not open image: {img_path}")
        sys.exit(1)

    win = "Pick grid points  (left-click = coord, right-click = quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, on_mouse, img)
    cv2.imshow(win, img)

    print("\nLeft-click on the image to get Isaac cm coordinates.")
    print("Right-click (or press Q / Esc) to quit.\n")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord('q'), 27):   # Q or Esc
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print("\nDone. Copy the Isaac cm values above into GRID_CM in replay_bug.py.")


if __name__ == "__main__":
    main()