#!/usr/bin/env python3
"""
pick_pixels.py
--------------
Left-click anywhere on an image to print raw pixel coordinates.
Use this to record calibration points from real or SIM images.

Usage:
    python pick_pixels.py path/to/image.png

Left-click  → prints (col, row)
Right-click → quit
"""

import sys
import cv2
import numpy as np

click_count = [0]

def on_mouse(event, x, y, flags, param):
    img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        click_count[0] += 1
        print(f"  Click {click_count[0]:>2d}  pixel = ({x}, {y})")
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.circle(img, (x, y), 6, (255, 255, 255), 1)
        cv2.putText(img, f"{click_count[0]}: ({x},{y})",
                    (x + 8, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow(WIN, img)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.destroyAllWindows()


def main():
    global WIN
    if len(sys.argv) < 2:
        print("Usage: python pick_pixels.py <image.png>")
        sys.exit(1)

    path = sys.argv[1]
    img  = cv2.imread(path)
    if img is None:
        print(f"Could not open: {path}")
        sys.exit(1)

    WIN = f"Pixel picker — {path}  (left-click = record, right-click = quit)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    cv2.setMouseCallback(WIN, on_mouse, img)
    cv2.imshow(WIN, img)

    print(f"\nImage: {path}")
    print("Left-click to record pixel coordinates. Right-click or Q/Esc to quit.\n")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord('q'), 27):
            break
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print(f"\nDone — {click_count[0]} point(s) recorded.")


if __name__ == "__main__":
    main()