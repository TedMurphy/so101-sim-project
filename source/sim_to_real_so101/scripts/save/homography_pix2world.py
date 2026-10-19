"""
Drop-in replacement for the linear pix2world / _det_pix2world in:
  - replay_interactive.py
  - batch_score_v2.py

Calibrated 2026-10-08 from 4 mat corners.  Camera pointing straight down.
640x480 images.

Calibration points (click order: TL, TR, BL, BR):

  Corner  real px         sim px          world (m)
  TL      (  0.0,   0.0)  (  8.5,   1.3)  (-0.330,+0.005)
  TR      (640.0,   0.0)  (631.5,   1.1)  (+0.400,+0.005)
  BL      (  0.0, 480.0)  (  4.8, 473.2)  (-0.310,+0.380)
  BR      (640.0, 480.0)  (634.1, 473.5)  (+0.375,+0.400)

  world_X = -Isaac_X  (right in image = positive world X)
  world_Y =  Isaac_Y

Residuals: exact fit (4 pts each)
"""

import numpy as np
import cv2

# Homography matrices (pixel -> world XY)
# Row 0 -> world X,  Row 1 -> world Y,  Row 2 -> homogeneous scale

_H_REAL = np.array([
    [+1.10696274e-03, +5.09168491e-07, -3.30000013e-01],
    [-4.20778601e-07, +8.31701143e-04, +4.99999989e-03],
    [-8.41557221e-05, +1.32766194e-04, +1.00000000e+00],
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


if __name__ == "__main__":
    print("=== Verification (2026-10-08 calibration) ===")
    cal_pts = [
        ((  0.0,   0.0), (  8.5,   1.3), (-0.330, 0.005), "TL"),
        ((640.0,   0.0), (631.5,   1.1), ( 0.400, 0.005), "TR"),
        ((  0.0, 480.0), (  4.8, 473.2), (-0.310, 0.380), "BL"),
        ((640.0, 480.0), (634.1, 473.5), ( 0.375, 0.400), "BR"),
    ]
    for rp, sp, (wx, wy), lbl in cal_pts:
        gr = pix2world_real(*rp)
        gs = pix2world_sim(*sp)
        print(f"{lbl}  real px {rp} -> world ({gr[0]:+.3f},{gr[1]:+.3f})  exp ({wx:+.3f},{wy:+.3f})")
        print(f"      sim  px {sp} -> world ({gs[0]:+.3f},{gs[1]:+.3f})")