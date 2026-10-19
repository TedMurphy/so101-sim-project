#!/usr/bin/env python3
"""
Homography recalibration tool — click 4 mat corners in each image.

Usage:
    python recalibrate_homography.py  [--real PATH] [--sim PATH]

Defaults:
    --real  /home/ted/real_overhead.png   (or pass the actual path)
    --sim   /home/ted/sim_overhead.png

Pick corners in this order for BOTH images:
    TL  Top-Left     (far-left, far from robot)
    TR  Top-Right    (far-right, far from robot)
    BL  Bottom-Left  (near-left, robot side)
    BR  Bottom-Right (near-right, robot side)

The script prints the updated _H_REAL / _H_SIM matrices to paste into
homography_pix2world.py, then writes the whole updated file to ./homography_pix2world_new.py.
"""

import argparse
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" / "Agg" if TkAgg is absent
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import date

# ── World coordinates of the 4 mat corners (Isaac Lab metres) ─────────────────
# Edit these to match your actual mat-corner green-sphere positions.
# Convention: world_X = -Isaac_X  →  right in image = positive world_X
#             world_Y =  Isaac_Y
WORLD_CORNERS_XY = [
    (-0.330, 0.005),   # TL
    ( 0.400, 0.005),   # TR
    (-0.310, 0.380),   # BL
    ( 0.375, 0.400),   # BR
]
CORNER_LABELS = ["TL", "TR", "BL", "BR"]
CORNER_COLORS = ["red", "blue", "green", "orange"]

# ─────────────────────────────────────────────────────────────────────────────

def click_corners(img_bgr, title, n=4):
    """Show image; user clicks n corners; returns list of (col, row)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img_rgb)
    ax.set_title(f"{title}\nClick {n} corners in order: TL → TR → BL → BR\n"
                 f"(close window when done)", fontsize=11)

    clicks = []
    markers = []

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        col, row = event.xdata, event.ydata
        idx = len(clicks)
        if idx >= n:
            return
        clicks.append((col, row))
        lbl = CORNER_LABELS[idx]
        col_c = CORNER_COLORS[idx]
        m = ax.plot(col, row, "x", color=col_c, markersize=14, markeredgewidth=3)[0]
        t = ax.text(col + 8, row - 8, lbl, color=col_c, fontsize=13, fontweight="bold")
        markers.append((m, t))
        fig.canvas.draw()
        print(f"  {lbl}  pixel ({col:.1f}, {row:.1f})")
        if len(clicks) == n:
            ax.set_title(f"{title} — {n} corners picked. Close this window to continue.",
                         fontsize=11, color="green")
            fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    if len(clicks) != n:
        print(f"[ERROR] Expected {n} clicks, got {len(clicks)}. Exiting.")
        sys.exit(1)
    return clicks


def compute_homography(pixel_pts, world_pts):
    """
    Compute pixel→world homography H such that:
        [wx, wy, 1]^T  ∝  H * [col, row, 1]^T
    Returns H (3×3).
    """
    src = np.array(pixel_pts, dtype=np.float64)   # (N,2) col, row
    dst = np.array(world_pts, dtype=np.float64)   # (N,2) wx, wy
    H, mask = cv2.findHomography(src, dst, method=0)   # 0 = least squares
    return H


def verify(H, pixel_pts, world_pts, label):
    print(f"\n{label} residuals:")
    for (col, row), (wx_exp, wy_exp), lbl in zip(pixel_pts, world_pts, CORNER_LABELS):
        pt = np.array([[[col, row]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, H.astype(np.float32))
        wx_got, wy_got = out[0, 0, 0], out[0, 0, 1]
        print(f"  {lbl}  px=({col:.1f},{row:.1f})  "
              f"expect=({wx_exp:+.3f},{wy_exp:+.3f})  "
              f"got=({wx_got:+.3f},{wy_got:+.3f})  "
              f"err=({(wx_got-wx_exp)*100:+.1f}cm, {(wy_got-wy_exp)*100:+.1f}cm)")


def matrix_repr(H, name):
    """Return a Python array literal string for the matrix."""
    rows = []
    for row in H:
        r = ", ".join(f"{v:+.8e}" for v in row)
        rows.append(f"    [{r}],")
    inner = "\n".join(rows)
    return f"{name} = np.array([\n{inner}\n], dtype=np.float64)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default=None,
                        help="Path to real overhead image (640×480 jpg/png)")
    parser.add_argument("--sim",  default=None,
                        help="Path to sim overhead image (640×480 png)")
    args = parser.parse_args()

    real_path = args.real
    sim_path  = args.sim

    if real_path is None or sim_path is None:
        print("Paths not supplied — trying to find them in the current dir...")
        import glob, os
        candidates = sorted(glob.glob("/home/ted/so101-sim-project/sim_overhead_ep0_frame*.png"))
        if candidates:
            sim_path = sim_path or candidates[-1]
            print(f"  sim  → {sim_path}")
        candidates = sorted(glob.glob("/home/ted/real_overhead*.png") +
                            glob.glob("/home/ted/*.jpg"))
        if candidates:
            real_path = real_path or candidates[0]
            print(f"  real → {real_path}")

    if real_path is None or sim_path is None:
        print("Usage:  python recalibrate_homography.py --real REAL.png --sim SIM.png")
        sys.exit(1)

    real_img = cv2.imread(real_path)
    sim_img  = cv2.imread(sim_path)
    if real_img is None:
        print(f"[ERROR] Could not read real image: {real_path}"); sys.exit(1)
    if sim_img is None:
        print(f"[ERROR] Could not read sim image:  {sim_path}");  sys.exit(1)

    print(f"Real image: {real_img.shape[1]}×{real_img.shape[0]}  {real_path}")
    print(f"Sim  image: {sim_img.shape[1]}×{sim_img.shape[0]}   {sim_path}")
    print()
    print("World corners to use:")
    for lbl, (wx, wy) in zip(CORNER_LABELS, WORLD_CORNERS_XY):
        print(f"  {lbl}: world ({wx:+.3f}, {wy:+.3f})")
    print()
    print("Step 1 — REAL image (click TL → TR → BL → BR)")
    real_px = click_corners(real_img, "REAL overhead camera")

    print("\nStep 2 — SIM image (click TL → TR → BL → BR)")
    sim_px  = click_corners(sim_img,  "SIM overhead camera")

    print("\nComputing homographies ...")
    H_real = compute_homography(real_px, WORLD_CORNERS_XY)
    H_sim  = compute_homography(sim_px,  WORLD_CORNERS_XY)

    verify(H_real, real_px, WORLD_CORNERS_XY, "_H_REAL")
    verify(H_sim,  sim_px,  WORLD_CORNERS_XY, "_H_SIM")

    today = date.today().isoformat()
    print(f"\n{'─'*70}")
    print(f"# Calibrated {today}")
    print(matrix_repr(H_real, "_H_REAL"))
    print()
    print(matrix_repr(H_sim,  "_H_SIM"))
    print(f"{'─'*70}\n")

    # ── Write updated homography_pix2world.py ────────────────────────────────
    out_lines = f'''"""
Drop-in replacement for the linear pix2world / _det_pix2world in:
  - replay_interactive.py
  - batch_score_v2.py

Calibrated {today}.  Camera position set in cylinder_to_target_env_cfg.py.
640×480 images.

Calibration points (click order: TL, TR, BL, BR):

  Corner  real px         sim px          world (m)
  ──────  ──────────────  ──────────────  ──────────────
  TL      {real_px[0][0]:7.1f},{real_px[0][1]:7.1f}  {sim_px[0][0]:7.1f},{sim_px[0][1]:7.1f}  ({WORLD_CORNERS_XY[0][0]:+.3f},{WORLD_CORNERS_XY[0][1]:+.3f})
  TR      {real_px[1][0]:7.1f},{real_px[1][1]:7.1f}  {sim_px[1][0]:7.1f},{sim_px[1][1]:7.1f}  ({WORLD_CORNERS_XY[1][0]:+.3f},{WORLD_CORNERS_XY[1][1]:+.3f})
  BL      {real_px[2][0]:7.1f},{real_px[2][1]:7.1f}  {sim_px[2][0]:7.1f},{sim_px[2][1]:7.1f}  ({WORLD_CORNERS_XY[2][0]:+.3f},{WORLD_CORNERS_XY[2][1]:+.3f})
  BR      {real_px[3][0]:7.1f},{real_px[3][1]:7.1f}  {sim_px[3][0]:7.1f},{sim_px[3][1]:7.1f}  ({WORLD_CORNERS_XY[3][0]:+.3f},{WORLD_CORNERS_XY[3][1]:+.3f})

  world_X = -Isaac_X  (right in image = positive world_X)
  world_Y =  Isaac_Y
"""

import numpy as np
import cv2

# ── Homography matrices (pixel → world XY) ───────────────────────────────────
# Row 0 → world X,  Row 1 → world Y,  Row 2 → homogeneous scale

{matrix_repr(H_real, "_H_REAL")}

{matrix_repr(H_sim, "_H_SIM")}

# Inverse matrices (world → pixel, for visualisation)
_H_REAL_INV = np.linalg.inv(_H_REAL)
_H_SIM_INV  = np.linalg.inv(_H_SIM)


def pix2world_real(col: float, row: float):
    """Convert real-camera pixel (col, row) → world (X, Y) in metres."""
    pt = np.array([[[float(col), float(row)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_REAL.astype(np.float32))
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def pix2world_sim(col: float, row: float):
    """Convert sim-camera pixel (col, row) → world (X, Y) in metres."""
    pt = np.array([[[float(col), float(row)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_SIM.astype(np.float32))
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def world2pix_real(wx: float, wy: float):
    """Convert world (X, Y) → real-camera pixel (col, row)."""
    pt = np.array([[[float(wx), float(wy)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_REAL_INV.astype(np.float32))
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def world2pix_sim(wx: float, wy: float):
    """Convert world (X, Y) → sim-camera pixel (col, row)."""
    pt = np.array([[[float(wx), float(wy)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_SIM_INV.astype(np.float32))
    return float(out[0, 0, 0]), float(out[0, 0, 1])


# ── Aliases matching existing API ─────────────────────────────────────────────
_det_pix2world = pix2world_real   # drop-in for replay_interactive.py
pix2world      = pix2world_real   # drop-in for batch_score_v2.py


if __name__ == "__main__":
    print("=== Verification ({today}) ===")
    pts = [
        ({real_px[0]}, {sim_px[0]}, {WORLD_CORNERS_XY[0]}, "TL"),
        ({real_px[1]}, {sim_px[1]}, {WORLD_CORNERS_XY[1]}, "TR"),
        ({real_px[2]}, {sim_px[2]}, {WORLD_CORNERS_XY[2]}, "BL"),
        ({real_px[3]}, {sim_px[3]}, {WORLD_CORNERS_XY[3]}, "BR"),
    ]
    for rp, sp, (wx, wy), lbl in pts:
        gr = pix2world_real(*rp)
        gs = pix2world_sim(*sp)
        print(f"{{lbl}}  real→world=({gr[0]:+.3f},{gr[1]:+.3f}) exp=({wx:+.3f},{wy:+.3f})  "
              f"sim→world=({gs[0]:+.3f},{gs[1]:+.3f})")
'''

    out_path = "./homography_pix2world_new.py"
    with open(out_path, "w") as f:
        f.write(out_lines)
    print(f"Updated file written to: {out_path}")
    print("Review it, then rename it to homography_pix2world.py")


if __name__ == "__main__":
    main()