"""
Drop-in replacement for the linear pix2world / _det_pix2world in:
  - replay_interactive.py
  - batch_score_v2.py

Calibrated 2026-09-23 from 4 mat corners identified by Isaac Sim green
sphere markers placed at the physical mat corners.  Camera at Z=0.610m,
nadir Rz180°.  Mat: near edge 75cm (robot side), far edge 70cm.  640×480.

Calibration points — same real/sim pixel clicks as 2026-09-11, but world
labels updated to match the Isaac green-sphere corner positions:

  Corner  real px         sim px          Isaac (cm)     world (m)
  ──────  ──────────────  ──────────────  ─────────────  ──────────────
  TL      ( 13.4,  67.0)  (  4.9,  60.9)  (+33.0, +0.5)  (-0.330,+0.005)
  TR      (631.5,  65.8)  (636.3,  62.1)  (-40.0, +0.5)  (+0.400,+0.005)
  BL      (  8.5, 404.9)  ( 24.4, 384.2)  (+31.0,+38.0)  (-0.310,+0.380)
  BR      (632.7, 400.0)  (611.9, 406.1)  (-37.5,+40.0)  (+0.375,+0.400)

  world_X = -Isaac_X  (Rz180° convention; right in image = positive world X)
  world_Y =  Isaac_Y

Previous calibration (2026-09-11) vs new — systematic shift per corner:
  TL: ΔX = +8.0 cm  ΔY = +0.5 cm
  TR: ΔX = +6.0 cm  ΔY = +0.5 cm
  BL: ΔX = +7.5 cm  ΔY = −2.5 cm
  BR: ΔX = +6.0 cm  ΔY = +2.0 cm
The consistent ~+6–8 cm world-X shift means Isaac was receiving cylinder
targets displaced ~7 cm too far in the −X direction (leftward in Isaac).

Residuals — H_REAL: exact fit (4 pts)  H_SIM: exact fit (4 pts)
NOTE: X convention — right in image = positive world X
      (same as 2026-09-11; opposite of pre-2026-09-11 matrices)
"""

import numpy as np
import cv2

# ── Homography matrices (pixel → world XY) ───────────────────────────────────
# Row 0 → world X,  Row 1 → world Y,  Row 2 → homogeneous scale

_H_REAL = np.array([
    [+1.15486834e-03, +7.69862523e-06, -3.50418283e-01],
    [+1.82245483e-06, +1.20933059e-03, -7.59824916e-02],
    [-1.04645431e-04, +2.21165457e-04, +1.00000000e+00],
], dtype=np.float64)

_H_SIM = np.array([
    [+1.16187718e-03, -8.87416243e-07, -3.35165718e-01],
    [-2.09208300e-06, +1.14928182e-03, -6.49881867e-02],
    [+1.84807624e-05, -2.50451697e-05, +1.00000000e+00],
], dtype=np.float64)
# ↑ Recomputed 2026-09-23: derived from the sim px column in the calibration
# table above (sim_overhead_ep*_frame5.png corners → world m).  The previous
# matrix had ~6-8 cm systematic error because it was computed from
# debug_sim_ep*_marker.png (replay_bug.py output) whose corner pixels differ.

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
# In replay_interactive.py, replace _det_pix2world with:
#   from homography_pix2world import pix2world_real as _det_pix2world
#
# In batch_score_v2.py, replace pix2world with:
#   from homography_pix2world import pix2world_real as pix2world
#
# If batch_score_v2 scores sim frames, use pix2world_sim instead.

_det_pix2world = pix2world_real   # drop-in for replay_interactive.py
pix2world      = pix2world_real   # drop-in for batch_score_v2.py


if __name__ == "__main__":
    print("=== Verification (mat-corner green-sphere calibration 2026-09-23) ===")
    cal_pts = [
        (( 13.4,  67.0), (  4.9,  60.9), (-0.330, 0.005), (+0.330, 0.005), "TL"),
        ((631.5,  65.8), (636.3,  62.1), ( 0.400, 0.005), (-0.400, 0.005), "TR"),
        ((  8.5, 404.9), ( 24.4, 384.2), (-0.310, 0.380), (+0.310, 0.380), "BL"),
        ((632.7, 400.0), (611.9, 406.1), ( 0.375, 0.400), (-0.375, 0.400), "BR"),
    ]
    print(f"{'Pt':<4} {'Real px':>16}  {'World exp':>16}  {'Real got':>16}  "
          f"{'Isaac via real':>18}  {'Isaac expect':>16}")
    for rp, sp, (wx, wy), (ix, iy), lbl in cal_pts:
        gr = pix2world_real(*rp)
        gs = pix2world_sim(*sp)
        # Isaac from real: ix = -wx, iy = wy
        print(f"{lbl:<4} {str(rp):>16}  ({wx:+.3f},{wy:+.3f})  "
              f"({gr[0]:+.3f},{gr[1]:+.3f})  "
              f"Isaac=({-gr[0]:+.3f},{gr[1]:+.3f})  "
              f"expect=({ix:+.3f},{iy:+.3f})")
    print()
    print("Sim check:")
    for rp, sp, (wx, wy), (ix, iy), lbl in cal_pts:
        gs = pix2world_sim(*sp)
        print(f"  {lbl}  sim px {sp} → world ({gs[0]:+.3f},{gs[1]:+.3f})  "
              f"Isaac=({-gs[0]:+.3f},{gs[1]:+.3f})  expect=({ix:+.3f},{iy:+.3f})")