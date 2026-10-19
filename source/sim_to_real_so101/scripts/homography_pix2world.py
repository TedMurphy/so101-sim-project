"""
Drop-in replacement for the linear pix2world / _det_pix2world in:
  - replay_interactive.py
  - batch_score_v2.py

Recalibrated 2026-10-12 from 4 image corners matched to physical graph paper.
Camera pointing straight down.  640x480 images.

Calibration points (image corners → physical paper measurements):

  Corner  real px          world (cm)   world (m)
  TL      (  0,   0)       ( 0,  0)     (0.000,  0.000)
  TR      (639,   0)       (68,  0)     (0.680,  0.000)
  BL      (  0, 479)       ( 2, 50)     (0.020,  0.500)
  BR      (639, 479)       (66, 50)     (0.660,  0.500)

  world_X  increases to the right in the image (0 at left edge)
  world_Y  increases downward in the image (0 at top edge, 0.50 m at bottom)
  isaac_x = -world_X + X_OFFSET
  isaac_y =  world_Y + Y_OFFSET

Previous calibration (2026-10-08) had only 37.4 cm depth span; this one
correctly captures the physical 50 cm depth.

Residuals: exact fit (4 pts each — minimal set)
"""

import numpy as np
import cv2

# Homography matrices (pixel -> world XY)
# Row 0 -> world X,  Row 1 -> world Y,  Row 2 -> homogeneous scale

_H_REAL = np.array([
    [+1.0641627654969605e-03, +4.4363254411092787e-05, +0.0000000000000000e+00],
    [+0.0000000000000000e+00, +1.1090813850672171e-03, +0.0000000000000000e+00],
    [+0.0000000000000000e+00, +1.3048009790061474e-04, +1.0000000000000000e+00],
], dtype=np.float64)

_H_SIM = np.array([
    [+1.13726841e-03, +2.29757709e-06, -3.39481445e-01],
    [-6.72655796e-08, +8.54071587e-04, +3.87419490e-03],
    [-8.46790870e-05, +1.56871893e-04, +1.00000000e+00],
], dtype=np.float64)

# Inverse matrices (world -> pixel, for visualisation)
_H_REAL_INV = np.linalg.inv(_H_REAL)
_H_SIM_INV  = np.linalg.inv(_H_SIM)


def pix2world_real(col: float, row: float):
    """Convert real-camera pixel (col, row) -> world (X, Y) in metres."""
    pt = np.array([[[float(col), float(row)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_REAL.astype(np.float32))
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def pix2world_sim(col: float, row: float):
    """Convert sim-camera pixel (col, row) -> world (X, Y) in metres."""
    pt = np.array([[[float(col), float(row)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_SIM.astype(np.float32))
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def world2pix_real(wx: float, wy: float):
    """Convert world (X, Y) -> real-camera pixel (col, row)."""
    pt = np.array([[[float(wx), float(wy)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_REAL_INV.astype(np.float32))
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def world2pix_sim(wx: float, wy: float):
    """Convert world (X, Y) -> sim-camera pixel (col, row)."""
    pt = np.array([[[float(wx), float(wy)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_SIM_INV.astype(np.float32))
    return float(out[0, 0, 0]), float(out[0, 0, 1])


# Aliases matching existing API
_det_pix2world = pix2world_real   # drop-in for replay_interactive.py
pix2world      = pix2world_real   # drop-in for batch_score_v2.py


# ── Real-world ↔ SIM coordinate transform (cm) ───────────────────────────────
# Derived from the four mat corners measured in both spaces (2026-10-14):
#
#   Corner  REAL (cm)     SIM (cm)
#   TL      ( 0,  0)      (+30, +10)
#   TR      (68,  0)      (-36, +10)
#   BL      ( 2, 50)      (+28, +59)
#   BR      (66, 50)      (-34, +59)
#
# Units: centimetres in both input and output.
# Computed at import time via cv2.getPerspectiveTransform (exact fit, 4 pts).

_REAL_CM_PTS = np.array([[ 0.,  0.],
                          [68.,  0.],
                          [ 2., 50.],
                          [66., 50.]], dtype=np.float32)

_SIM_CM_PTS  = np.array([[ 30., 10.],
                          [-36., 10.],
                          [ 28., 59.],
                          [-34., 59.]], dtype=np.float32)

_H_REAL_TO_SIM = cv2.getPerspectiveTransform(_REAL_CM_PTS, _SIM_CM_PTS)
_H_SIM_TO_REAL = cv2.getPerspectiveTransform(_SIM_CM_PTS, _REAL_CM_PTS)


def real_to_sim(real_x_cm: float, real_y_cm: float):
    """Convert real-world (X, Y) in centimetres → SIM (X, Y) in centimetres.

    Uses a homography fitted to the four mat corners (exact, 4-pt minimal set).
    Input and output are both in centimetres.
    """
    pt  = np.array([[[float(real_x_cm), float(real_y_cm)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_REAL_TO_SIM)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def sim_to_real(sim_x_cm: float, sim_y_cm: float):
    """Convert SIM (X, Y) in centimetres → real-world (X, Y) in centimetres.

    Inverse of real_to_sim; same 4-pt homography, applied in reverse.
    """
    pt  = np.array([[[float(sim_x_cm), float(sim_y_cm)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_SIM_TO_REAL)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


if __name__ == "__main__":
    print("=== Verification (2026-10-12 recalibration) ===")
    cal_pts = [
        ((  0,   0), (0.000, 0.000), "TL"),
        ((639,   0), (0.680, 0.000), "TR"),
        ((  0, 479), (0.020, 0.500), "BL"),
        ((639, 479), (0.660, 0.500), "BR"),
    ]
    for (col, row), (wx, wy), lbl in cal_pts:
        gr = pix2world_real(col, row)
        print(f"{lbl}  pixel ({col:3d},{row:3d}) -> world ({gr[0]:+.4f},{gr[1]:.4f})  "
              f"expected ({wx:+.3f},{wy:.3f})")

    print("\n=== Verification real_to_sim (2026-10-14) ===")
    corner_pts = [
        (( 0.,  0.), ( 30., 10.), "TL"),
        ((68.,  0.), (-36., 10.), "TR"),
        (( 2., 50.), ( 28., 59.), "BL"),
        ((66., 50.), (-34., 59.), "BR"),
    ]
    for (rx, ry), (sx, sy), lbl in corner_pts:
        got = real_to_sim(rx, ry)
        print(f"{lbl}  real ({rx:5.1f},{ry:4.1f}) -> sim ({got[0]:+6.2f},{got[1]:5.2f})  "
              f"expected ({sx:+.0f},{sy:.0f})")