"""calibrate_sim_homography.py — Empirically calibrate _H_SIM for Isaac Lab.

WHY THIS EXISTS
--------------
_H_SIM in homography_pix2world.py was built from mat-corner physical markers
at Z=0 (flat on the table).  The cylinder is placed at Z=0.050, which
introduces parallax from the overhead camera and an internal Isaac-coordinate
origin offset.  As a result H_SIM predicts the wrong SIM pixel for a given
world position — we measured ~15 cm error in Y.

THIS SCRIPT
-----------
1. Launches Isaac Lab with the same environment as replay_bug.py.
2. Places the green cylinder at a grid of KNOWN world positions (no episode
   playback needed — we just pin the cylinder).
3. After each placement (8 physics steps with re-pinning so the TiledCamera
   buffer actually refreshes), detects the actual SIM pixel of the cylinder
   blob via HSV blob detection.
4. Collects the (actual_SIM_col, actual_SIM_row) → (world_X, world_Y) pairs.
5. Runs cv2.findHomography on those pairs to produce a new _H_SIM.
6. Writes an updated homography_pix2world.py alongside the original (as a
   _calibrated suffix) so you can inspect before replacing.

USAGE
-----
    ./isaaclab.sh -p calibrate_sim_homography.py

OUTPUT
------
    ~/so101-sim-project/calib_sim/
        grid_pt_{i:02d}_wx{wx:+.3f}_wy{wy:.3f}.png   — annotated SIM frame per point
    ./homography_pix2world_calibrated.py               — drop-in replacement

COORDINATE CONVENTIONS (same as replay_bug.py)
------------------------------------------------
    Isaac_X = -world_X   (Rz-180 camera)
    Isaac_Y =  world_Y
    Z (cylinder centre) = 0.050 m  (hardcoded in _CYLINDER_Z)

H_SIM MEANING
-------------
    H_SIM  maps SIM pixel (col, row) → world (X, Y)   [forward direction]
    H_SIM_INV = inv(H_SIM) maps world (X, Y) → SIM pixel  [for annotation]
"""

import argparse
import os
import sys
import datetime

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Calibrate _H_SIM empirically.")
parser.add_argument("--task", type=str, default="Lerobot-So101-Teleop-Cylinder-To-Target")
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args([
    "--livestream", "2",
    "--kit_args",
    "--ext-folder /home/ted/isaac-sim/extsDeprecated "
    "--enable isaacsim.core.utils "
    "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false",
])
args_cli.enable_cameras = True
args_cli.device = "cpu"

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports after Isaac Sim is up ─────────────────────────────────────────────
import cv2
import numpy as np
import torch
import gymnasium as gym
import warnings

import isaaclab_tasks                          # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks                 # noqa: F401

# ── Tunable constants ─────────────────────────────────────────────────────────

# Output directory for per-point PNG frames.
OUT_DIR = os.path.expanduser("~/so101-sim-project/calib_sim")
os.makedirs(OUT_DIR, exist_ok=True)

# Cylinder Z height in Isaac metres.
_CYLINDER_Z = 0.050

# Physics steps per grid point (must be > 0; 8 is enough for TiledCamera refresh).
_SETTLE_STEPS = 12

# HSV blob detection window for SIM (wide because RTX tone-mapping shifts hue).
_HSV_LO = np.array(( 30,  30,  40), dtype=np.uint8)
_HSV_HI = np.array(( 95, 255, 255), dtype=np.uint8)
_MIN_AREA = 200   # px² — ignore tiny noise blobs

# Calibration grid in WORLD coordinates (metres).
# world_X: negative = left in image (Isaac +X = robot right = image right)
# world_Y: positive = far from camera (Isaac +Y = away from camera)
# Keep away from the mat edges and the arm base to avoid occlusion.
_GRID_WX = [-0.25, -0.10,  0.00, +0.15, +0.30]   # world X (m)
# Start Y at 0.10 — at 0.06 the cylinder is near the robot base and gets
# occluded; the HSV detector latches on a fixed green element in the scene.
_GRID_WY = [ 0.10,  0.19,  0.28,  0.37 ]          # world Y (m)

# Minimum number of successful detections needed before fitting homography.
_MIN_POINTS = 6


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pin_cylinder(obj, isaac_x: float, isaac_y: float, isaac_z: float, device):
    """Write (isaac_x, isaac_y, isaac_z) as the cylinder's root state."""
    ss = obj.data.root_state_w.clone()
    ss[:, :3]  = torch.tensor([[isaac_x, isaac_y, isaac_z]],
                               dtype=torch.float32, device=device)
    ss[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]],
                               dtype=torch.float32, device=device)
    ss[:, 7:]  = 0.0
    obj.data.root_state_w[:] = ss
    obj.write_root_state_to_sim(ss)


def _settle(env, obj, isaac_x, isaac_y, isaac_z, n_steps, zero_action):
    """Pin the cylinder and run n_steps physics steps so the camera refreshes."""
    for _ in range(n_steps):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            _pin_cylinder(obj, isaac_x, isaac_y, isaac_z, zero_action.device)
        env.step(zero_action)


def _detect_blob(env):
    """Return (col, row) of the largest green blob in the SIM camera, or (None, None)."""
    rgb = env.unwrapped.scene["camera_overhead"].data.output["rgb"]
    fp  = rgb[0].cpu().numpy().astype(np.uint8)
    if fp.shape[-1] == 4:
        fp = fp[..., :3]
    bgr  = cv2.cvtColor(fp, cv2.COLOR_RGB2BGR)
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _HSV_LO, _HSV_HI)
    ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ctrs:
        return None, None, fp
    c = max(ctrs, key=cv2.contourArea)
    if cv2.contourArea(c) < _MIN_AREA:
        return None, None, fp
    M = cv2.moments(c)
    if M["m00"] <= 0:
        return None, None, fp
    col = float(M["m10"] / M["m00"])
    row = float(M["m01"] / M["m00"])
    return col, row, fp   # fp is RGB array (for saving as PNG)


def _save_frame(rgb_np, col, row, wx, wy, pt_idx, path):
    """Save an annotated PNG for one calibration point."""
    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    # Blue circle at detected blob
    if col is not None:
        cv2.circle(bgr, (int(round(col)), int(round(row))), 12, (255, 80, 0), 2)
        cv2.drawMarker(bgr, (int(round(col)), int(round(row))),
                       (255, 80, 0), cv2.MARKER_CROSS, 24, 2)

    # Label
    txt  = f"pt{pt_idx:02d}  world=({wx:+.3f},{wy:.3f})m"
    txt2 = f"blob=({col:.1f},{row:.1f})" if col is not None else "blob=NONE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, line in enumerate([txt, txt2]):
        cv2.putText(bgr, line, (10, 20 + i * 22), font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, line, (10, 20 + i * 22), font, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(path, bgr)
    print(f"  → saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  calibrate_sim_homography.py — empirical _H_SIM calibration")
    print("=" * 70)

    # ── Launch environment ────────────────────────────────────────────────────
    # parse_env_cfg signature changed across Isaac Lab versions; try both.
    try:
        env_cfg = parse_env_cfg(args_cli.task,
                                use_gpu=False, num_envs=1, use_fabric=True)
    except TypeError:
        try:
            env_cfg = parse_env_cfg(args_cli.task, num_envs=1)
        except TypeError:
            env_cfg = parse_env_cfg(args_cli.task)
            env_cfg.scene.num_envs = 1
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    device = env.unwrapped.device
    cyl    = env.unwrapped.scene["cylinder"]
    try:
        n_act = env.unwrapped.action_space.shape[-1]
    except Exception:
        n_act = 6
    zero_action = torch.zeros(1, n_act, dtype=torch.float32, device=device)

    # Prime the sim with a few settling steps before we start so the renderer
    # is fully warmed up (avoids a black first frame on some Isaac versions).
    print("\n[CAL] Priming renderer with 16 warm-up steps …")
    _settle(env, cyl, 0.0, 0.20, _CYLINDER_Z, 16, zero_action)
    print("[CAL] Warm-up done.\n")

    # ── Grid sweep ────────────────────────────────────────────────────────────
    # We collect:   src_pts = SIM pixel (col, row)
    #               dst_pts = world    (wx,  wy)
    # Then fit: H such that  H @ [col, row, 1]^T  →  [wx, wy, scale]^T
    # i.e. findHomography(src=sim_pixels, dst=world_pts) → _H_SIM
    src_pts  = []   # SIM pixel coords
    dst_pts  = []   # world coords
    failures = []   # grid points where blob was not found

    total_pts = len(_GRID_WX) * len(_GRID_WY)
    pt_idx    = 0

    # Duplicate-blob guard: track the last accepted blob pixel per wy row.
    # If two different wx positions at the same wy land within _DUP_THRESH px,
    # the second detection is a static false positive (arm/base, not the cylinder).
    _DUP_THRESH  = 8.0    # px — blobs this close across different wx are duplicates
    _last_by_wy  = {}     # wy → (col, row) of last accepted blob in this row

    for wx in _GRID_WX:
        for wy in _GRID_WY:
            pt_idx += 1
            # Isaac convention: Isaac_X = -world_X, Isaac_Y = world_Y
            ix = -wx
            iy =  wy
            iz =  _CYLINDER_Z

            print(f"[CAL] Point {pt_idx:02d}/{total_pts}  "
                  f"world=({wx:+.3f},{wy:.3f})m  "
                  f"Isaac=({ix:+.3f},{iy:.3f},{iz:.3f})m")

            # Place and settle
            _settle(env, cyl, ix, iy, iz, _SETTLE_STEPS, zero_action)

            # Detect blob
            col, row, rgb_np = _detect_blob(env)

            # Duplicate-blob guard
            is_dup = False
            if col is not None and wy in _last_by_wy:
                prev_col, prev_row = _last_by_wy[wy]
                dist = np.hypot(col - prev_col, row - prev_row)
                if dist < _DUP_THRESH:
                    print(f"  !! DUPLICATE blob at ({col:.1f},{row:.1f}) — "
                          f"same pixel as previous wx at this wy (dist={dist:.1f}px). "
                          f"Skipping — cylinder likely occluded.")
                    is_dup = True
                    col, row = None, None   # treat as not found

            # Save annotated frame  (replace special chars in stem only, keep .png)
            _stem = (f"grid_pt_{pt_idx:02d}_wx{wx:+.3f}_wy{wy:.3f}"
                     .replace("+", "p").replace("-", "m").replace(".", "d"))
            fname = _stem + ".png"
            _save_frame(rgb_np, col, row, wx, wy, pt_idx,
                        os.path.join(OUT_DIR, fname))

            if col is not None:
                print(f"  blob at SIM pixel ({col:.1f}, {row:.1f})")
                _last_by_wy[wy] = (col, row)
                src_pts.append([col, row])
                dst_pts.append([wx,  wy])
            else:
                if not is_dup:
                    print(f"  !! BLOB NOT FOUND — skipping this point")
                failures.append((wx, wy))

    # ── Fit homography ────────────────────────────────────────────────────────
    print(f"\n[CAL] Collected {len(src_pts)} / {total_pts} detections.")
    if failures:
        print(f"[CAL] Failed points (world): {failures}")

    if len(src_pts) < _MIN_POINTS:
        print(f"[CAL] ERROR: need at least {_MIN_POINTS} points; got {len(src_pts)}.")
        print("[CAL] Check HSV thresholds (_HSV_LO/_HSV_HI) or increase the grid.")
        env.close()
        simulation_app.close()
        return

    src_np = np.array(src_pts, dtype=np.float64)   # (N,2) SIM pixels
    dst_np = np.array(dst_pts, dtype=np.float64)   # (N,2) world metres

    # cv2.findHomography(srcPoints, dstPoints) gives H such that
    #   dst_h = H @ src_h   in homogeneous coords.
    # i.e. H maps SIM pixel → world XY  (the same convention as _H_SIM).
    H_new, mask = cv2.findHomography(src_np, dst_np, cv2.RANSAC, 0.010)

    if H_new is None:
        print("[CAL] cv2.findHomography returned None — check for degenerate points.")
        env.close()
        simulation_app.close()
        return

    inliers = int(mask.sum()) if mask is not None else len(src_pts)
    print(f"[CAL] Homography fit: {inliers}/{len(src_pts)} inliers (RANSAC).")

    # ── Residuals ─────────────────────────────────────────────────────────────
    print("\n[CAL] Residuals (SIM pixel → world, new H_SIM):")
    max_err = 0.0
    for i, ((col, row), (wx, wy)) in enumerate(zip(src_pts, dst_pts)):
        pt_h  = H_new @ np.array([col, row, 1.0])
        wx_p  = pt_h[0] / pt_h[2]
        wy_p  = pt_h[1] / pt_h[2]
        err_m = np.hypot(wx_p - wx, wy_p - wy)
        max_err = max(max_err, err_m)
        print(f"  pt{i+1:02d} SIM({col:.1f},{row:.1f}) "
              f"→ pred ({wx_p:+.4f},{wy_p:.4f})  "
              f"truth ({wx:+.4f},{wy:.4f})  "
              f"err={err_m*100:.2f}cm")
    print(f"[CAL] Max residual: {max_err*100:.2f} cm")

    H_new_inv = np.linalg.inv(H_new)

    # ── Write calibrated module ───────────────────────────────────────────────
    out_py = os.path.join(OUT_DIR, "homography_pix2world_calibrated.py")
    _write_calibrated_module(H_new, H_new_inv, src_pts, dst_pts, out_py)

    print(f"\n[CAL] Written → {out_py}")
    print("[CAL] Review residuals above, then copy over homography_pix2world.py")
    print("[CAL]   cp", out_py, "/path/to/homography_pix2world.py")

    env.close()
    simulation_app.close()


# ── Module writer ─────────────────────────────────────────────────────────────

def _write_calibrated_module(H, H_inv, src_pts, dst_pts, path):
    """Write a drop-in replacement homography_pix2world.py with the new _H_SIM."""
    # Format numpy matrix as Python literal
    def _fmt(m):
        rows = []
        for r in m:
            rows.append("    [{:+.8e}, {:+.8e}, {:+.8e}],".format(*r))
        return "\n".join(rows)

    today = datetime.date.today().isoformat()
    n     = len(src_pts)

    # Build calibration-point table comment
    lines = []
    for i, ((col, row), (wx, wy)) in enumerate(zip(src_pts, dst_pts)):
        lines.append(f"  pt{i+1:02d}  SIM px ({col:6.1f},{row:6.1f})  "
                     f"world ({wx:+.3f},{wy:.3f})")
    pt_table = "\n".join(lines)

    # Read the REAL homography from the existing module (keep it unchanged).
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from homography_pix2world import _H_REAL, _H_REAL_INV
        H_real_rows  = _fmt(_H_REAL)
        H_real_i_rows = _fmt(_H_REAL_INV)
    except Exception:
        # Fallback: use the values from the reference file
        _H_REAL = np.array([
            [+1.10696274e-03, +5.09168491e-07, -3.30000013e-01],
            [-4.20778601e-07, +8.31701143e-04, +4.99999989e-03],
            [-8.41557221e-05, +1.32766194e-04, +1.00000000e+00],
        ])
        _H_REAL_INV = np.linalg.inv(_H_REAL)
        H_real_rows  = _fmt(_H_REAL)
        H_real_i_rows = _fmt(_H_REAL_INV)

    content = f'''"""
Drop-in replacement for homography_pix2world.py.

  _H_REAL    — unchanged (calibrated 2026-10-08 from mat corners, Z=0)
  _H_SIM     — EMPIRICALLY RECALIBRATED on {today} from {n} cylinder
               placements at Z={_CYLINDER_Z:.3f} m in Isaac Lab.

Calibration points ({n} total):
{pt_table}

Coordinate conventions (same as original):
  world_X = -Isaac_X
  world_Y =  Isaac_Y
  H_SIM maps SIM pixel (col,row) → world (X,Y)  [forward direction]
  H_SIM_INV maps world (X,Y) → SIM pixel         [for annotation]
"""

import numpy as np
import cv2

# ── Real-camera homography (pixel → world XY) — UNCHANGED ──────────────────
_H_REAL = np.array([
{H_real_rows}
], dtype=np.float64)

_H_REAL_INV = np.linalg.inv(_H_REAL)

# ── SIM homography (SIM pixel → world XY) — EMPIRICALLY CALIBRATED {today} ─
_H_SIM = np.array([
{_fmt(H)}
], dtype=np.float64)

_H_SIM_INV = np.linalg.inv(_H_SIM)


def pix2world_real(col: float, row: float):
    """Convert real-camera pixel (col, row) → world (X, Y) in metres."""
    pt = np.array([[[float(col), float(row)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_REAL.astype(np.float32))
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def pix2world_sim(col: float, row: float):
    """Convert SIM-camera pixel (col, row) → world (X, Y) in metres."""
    pt = np.array([[[float(col), float(row)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_SIM.astype(np.float32))
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def world2pix_real(wx: float, wy: float):
    """Convert world (X, Y) → real-camera pixel (col, row)."""
    pt = np.array([[[float(wx), float(wy)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_REAL_INV.astype(np.float32))
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def world2pix_sim(wx: float, wy: float):
    """Convert world (X, Y) → SIM-camera pixel (col, row)."""
    pt = np.array([[[float(wx), float(wy)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_SIM_INV.astype(np.float32))
    return float(out[0, 0, 0]), float(out[0, 0, 1])


# Aliases matching existing API
_det_pix2world = pix2world_real   # drop-in for replay_interactive.py
pix2world      = pix2world_real   # drop-in for batch_score_v2.py


if __name__ == "__main__":
    print("=== Verification — calibrated {today} ===")
    print("\\nSIM centre pixel (320, 240):", pix2world_sim(320, 240))
    print("SIM TL pixel (0, 0):         ", pix2world_sim(0, 0))
    print("SIM TR pixel (640, 0):       ", pix2world_sim(640, 0))
    print("SIM BL pixel (0, 480):       ", pix2world_sim(0, 480))
    print("SIM BR pixel (640, 480):     ", pix2world_sim(640, 480))
'''

    with open(path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    main()