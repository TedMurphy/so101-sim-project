#!/usr/bin/env python3
"""
ruler_sim.py  — annotate the sim overhead frame with a horizontal ruler
showing distances from the robot centre to the left and right mat edges.

Run from the scripts directory:
  python3 ruler_sim.py --episode 7
"""

import argparse, os, sys
import numpy as np
import cv2

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from homography_pix2world import world2pix_sim

# ── World coords ──────────────────────────────────────────────────────────────
# Robot base (from task cfg): pos = (-0.040, 0.04, ...)
# Centre of the arm footprint (used for ruler Y level)
ROBOT_X  = -0.040   # world X of robot centre  (from task cfg pos=(-0.040, 0.04, ...))
ROBOT_Y  =  0.02    # Y for the ruler line (near the mat near-edge where width = 75 cm)

# Mat edges interpolated at ROBOT_Y
# near-left(-0.410,0) far-left(-0.385,0.405) → lerp
def mat_x_left(y):
    return -0.410 + (-0.385 - -0.410) * (y / 0.405)

def mat_x_right(y):
    return  0.340 + ( 0.315 -  0.340) * (y / 0.405)

LEFT_X  = mat_x_left(ROBOT_Y)
RIGHT_X = mat_x_right(ROBOT_Y)

DIST_LEFT  = abs(ROBOT_X - LEFT_X)    # metres
DIST_RIGHT = abs(RIGHT_X - ROBOT_X)   # metres

# ── Draw ──────────────────────────────────────────────────────────────────────
RULER_Y   = ROBOT_Y          # world Y where the ruler sits
TICK_HALF = 0.012            # half-height of end ticks (world metres)
COL_RULER = (255, 220,  40)  # yellow
COL_ROBOT = (100, 220, 255)  # cyan for robot centre mark
FONT      = cv2.FONT_HERSHEY_SIMPLEX


def w2p(wx, wy):
    c, r = world2pix_sim(wx, wy)
    return int(round(c)), int(round(r))


def draw_ruler(img, x_a, x_b, y_world, color, label, label_side="above"):
    """Horizontal ruler in world-X from x_a to x_b at y_world."""
    pa = w2p(x_a, y_world)
    pb = w2p(x_b, y_world)
    # tick ends (draw at ±TICK_HALF in world-Y)
    ta_top = w2p(x_a, y_world - TICK_HALF)
    ta_bot = w2p(x_a, y_world + TICK_HALF)
    tb_top = w2p(x_b, y_world - TICK_HALF)
    tb_bot = w2p(x_b, y_world + TICK_HALF)

    cv2.line(img, pa, pb, color, 2, cv2.LINE_AA)
    cv2.line(img, ta_top, ta_bot, color, 2, cv2.LINE_AA)
    cv2.line(img, tb_top, tb_bot, color, 2, cv2.LINE_AA)

    mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
    off = -18 if label_side == "above" else 22
    cv2.putText(img, label, (mid[0] - 50, mid[1] + off),
                FONT, 0.60, color, 2, cv2.LINE_AA)


def annotate(img):
    h, w = img.shape[:2]

    # Robot centre vertical line
    rx, ry = w2p(ROBOT_X, RULER_Y)
    cv2.line(img, (rx, 0), (rx, h), COL_ROBOT, 1, cv2.LINE_AA)
    cv2.putText(img, "robot centre", (rx + 4, 20),
                FONT, 0.45, COL_ROBOT, 1, cv2.LINE_AA)

    # Left-edge vertical line
    lx, ly = w2p(LEFT_X, RULER_Y)
    cv2.line(img, (lx, 0), (lx, h), (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(img, "mat left", (lx + 4, 20),
                FONT, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    # Right-edge vertical line
    rx2, ry2 = w2p(RIGHT_X, RULER_Y)
    cv2.line(img, (rx2, 0), (rx2, h), (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(img, "mat right", (rx2 - 80, 20),
                FONT, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    # Left ruler: mat-left → robot-centre
    draw_ruler(img, LEFT_X, ROBOT_X, RULER_Y, COL_RULER,
               f"{DIST_LEFT*100:.1f} cm", label_side="above")

    # Right ruler: robot-centre → mat-right
    draw_ruler(img, ROBOT_X, RIGHT_X, RULER_Y + 0.025, COL_RULER,
               f"{DIST_RIGHT*100:.1f} cm", label_side="below")

    # Summary box
    lines = [
        f"robot X = {ROBOT_X*100:.1f} cm",
        f"left edge X = {LEFT_X*100:.1f} cm  →  dist = {DIST_LEFT*100:.1f} cm",
        f"right edge X = {RIGHT_X*100:.1f} cm  →  dist = {DIST_RIGHT*100:.1f} cm",
    ]
    bx, by = 8, h - 8 - len(lines) * 20
    cv2.rectangle(img, (bx-4, by-16), (bx+420, by + len(lines)*20 - 4),
                  (20, 20, 20), -1)
    for i, l in enumerate(lines):
        cv2.putText(img, l, (bx, by + i * 20),
                    FONT, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=7)
    ap.add_argument("--sim_frame", type=str, default="",
                    help="Path to sim overhead PNG (default: ~/so101-sim-project/sim_overhead_ep<N>_frame5.png)")
    ap.add_argument("--out", type=str, default="",
                    help="Output path (default: ruler_ep<N>.png in CWD)")
    args = ap.parse_args()

    sim_path = args.sim_frame or os.path.expanduser(
        f"~/so101-sim-project/sim_overhead_ep{args.episode}_frame5.png")
    out_path = args.out or f"ruler_ep{args.episode}.png"

    if not os.path.exists(sim_path):
        sys.exit(f"Sim frame not found: {sim_path}")

    img = cv2.imread(sim_path)
    if img is None:
        sys.exit(f"Could not read: {sim_path}")
    img = cv2.resize(img, (640, 480))

    print(f"Robot centre X  = {ROBOT_X*100:.1f} cm  (world)")
    print(f"Mat left  at Y={ROBOT_Y:.2f}m : X = {LEFT_X*100:.1f} cm  → dist = {DIST_LEFT*100:.1f} cm")
    print(f"Mat right at Y={ROBOT_Y:.2f}m : X = {RIGHT_X*100:.1f} cm  → dist = {DIST_RIGHT*100:.1f} cm")

    annotate(img)
    cv2.imwrite(out_path, img)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
