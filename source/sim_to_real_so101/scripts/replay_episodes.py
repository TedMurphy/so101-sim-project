"""replay_episodes.py — Coordinate alignment debugger.

Pipeline:
  REAL IMAGE
  1. Read frame-0 from the real overhead video for the chosen episode.
  2. Detect the GREEN CYLINDER and YELLOW/ORANGE CUP via HSV blob detection.
  3. Convert detections pixel → world coords via _H_REAL homography
     (calibrated 2026-10-12; origin = top-left pixel of real image).
  4. Save the annotated real frame (detected blob circles drawn).

  SIM
  5. Launch Isaac Lab, load the episode, step to a stable frame.
  6. Place the GREEN CYLINDER in the USD stage at the world coords from step 3.
     TGS warm-start: one extra physics step + re-placement to clear stale
     PhysX contact impulses that cause lateral drift.
  7. Capture the SIM camera frame and annotate it:
       • graphpaper mat PNG overlay (mat boundary reference)
       • Isaac coordinate grid + intersection dot labels (SHOW_ISAAC_GRID)
       • white origin diamond at Isaac (0,0,0)  (SHOW_ORIGIN_MARKER)
  8. Save the annotated SIM frame for side-by-side comparison with step 4.

  ALIGNMENT CHECK
  The two saved images are compared visually: the cylinder's apparent position
  in the real frame should match its placed position in the SIM frame.
  Known issue: project_isaac_3d_to_sim_pixel has ~188px Y error, so the
  3D-projection overlays (crosshair, Isaac dot, placement dot) are disabled.

Usage:
    ./isaaclab.sh -p replay_episodes.py
    ./isaaclab.sh -p replay_episodes.py --episode 1

    NOTE: The SIM must be viewed through a running WebRTC streaming client.
    Launch it in a separate terminal window BEFORE starting the episode:

        ~/IsaacLab$ "/opt/Isaac Sim WebRTC Streaming Client/isaacsim-webrtc-streaming-client" &

    The client streams the Isaac Sim render to your browser; without it the
    SIM viewport is not visible even though the script runs successfully.

Outputs (written to ~/so101-sim-project/):
    debug_real_ep{N}_cylinder.png   — real frame annotated with detected cylinder + cup blobs
    debug_sim_ep{N}_marker.png      — SIM frame annotated with mat overlay, coord grid, origin marker

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COORDINATE SPACES  (read this first — the rest of the file won't make sense without it)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Four spaces are in play.  Every conversion lives in exactly one function.

1. PIXEL  (px_col, px_row)
   - Origin: top-left corner of the real overhead camera image.
   - Units: pixels.  +col goes right, +row goes DOWN.
   - Used by: OpenCV image reads, blob detector, drawn annotations.

2. REAL-WORLD  (wx, wy)  — also called "world coords"
   - Origin: a fixed physical point on the real table (calibrated 2026-10-12).
   - Units: metres.  +x goes right, +y goes away from the camera (into the scene).
   - Produced by: _H_REAL homography  →  pix2world_real(col, row)
   - This is the authoritative measurement.  All other spaces derive from it.

3. ISAAC  (ix, iy)  — Isaac Lab / PhysX world frame
   - Origin: the Isaac Lab scene origin (centre of env_0).
   - Units: metres.  Axes are FLIPPED relative to real-world:
       ix = -wx  (real +x  →  Isaac -x, i.e. left in the bird's-eye view)
       iy =  wy  (real +y  →  Isaac +y, same direction)
   - Produced by: real_world_to_isaac(wx, wy)
   - Used by: write_root_link_pose_to_sim(), all PhysX / USD placement calls.

4. USD PRIM  (translate attribute on the USD stage)
   - Same axes and units as Isaac, but written as a Gf.Vec3d to the USD stage
     so the RTX renderer sees the updated position.
   - Produced by: place_cylinder_at_coords() which calls both the PhysX tensor
     API (write_root_link_pose_to_sim) and the USD translate op (t_op.Set).

ORIENTATION — quaternions
  Every PhysX pose is a 7-element vector: [x, y, z, qw, qx, qy, qz].
  The last four numbers are a quaternion — a compact way to encode a 3D rotation.
  Identity quaternion (qw=1, qx=qy=qz=0) means "no rotation; use the default
  upright orientation."  The cylinder is rotationally symmetric so orientation
  is always identity in this file.  If you were placing an oriented object
  (a box, a screwdriver) you would supply a non-identity quaternion.

End-to-end for a cylinder placement:
  real camera pixel  →[_H_REAL]→  real-world metres  →[real_world_to_isaac]→
  Isaac metres  →[write_root_link_pose_to_sim + t_op.Set]→  PhysX + USD prim
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Coordinate alignment debugger.")
parser.add_argument("--task",         type=str, default="Lerobot-So101-Teleop-Cylinder-To-Target")
parser.add_argument("--real_repo_id", type=str, default="Screener2/FullGrid_20261007_095459")
parser.add_argument("--episode",      type=int, default=1)
parser.add_argument("--capture_frame",type=int, default=9999)
AppLauncher.add_app_launcher_args(parser)

# Read user-facing flags from the real command line, keeping required sim flags.
# Mirrors the pattern used in replay_interactive.py.
_ep      = 1
_repo_id = None
_cf      = 9999
for _i, _a in enumerate(sys.argv[1:], 1):
    if _a == "--episode" and _i + 1 < len(sys.argv):
        try: _ep = int(sys.argv[_i + 1])
        except ValueError: pass
    elif _a == "--real_repo_id" and _i + 1 < len(sys.argv):
        _repo_id = sys.argv[_i + 1]
    elif _a == "--capture_frame" and _i + 1 < len(sys.argv):
        try: _cf = int(sys.argv[_i + 1])
        except ValueError: pass

_extra = ["--real_repo_id", _repo_id] if _repo_id is not None else []
args_cli = parser.parse_args([
    "--livestream", "2", "--episode", str(_ep),
    "--enable_cameras",
    "--kit_args",
    "--ext-folder /home/ted/isaac-sim/extsDeprecated "
    "--enable isaacsim.core.utils "
    "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false",
] + _extra)
args_cli.capture_frame = _cf
args_cli.device        = "cpu"

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports that need Isaac Sim to be up first ────────────────────────────────
import gymnasium as gym
import numpy as np
import pandas as pd
import torch

import isaaclab_tasks                          # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks                 # noqa: F401
from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface
from apply_graphpaper_texture import apply_graphpaper_texture

# ── Homography: pixel ↔ world coordinate transforms ──────────────────────────
# Recalibrated 2026-10-12 from 4 image corners matched to physical graph paper.
# 640x480 images.
#
# Calibration points (image corners → physical paper measurements):
#
#   Corner  real px          world (cm)   world (m)
#   TL      (  0,   0)       ( 0,  0)     (0.000,  0.000)
#   TR      (639,   0)       (68,  0)     (0.680,  0.000)
#   BL      (  0, 479)       ( 2, 50)     (0.020,  0.500)
#   BR      (639, 479)       (66, 50)     (0.660,  0.500)
#
#   world_X  increases to the right in the image (0 at left edge)
#   world_Y  increases downward in the image (0 at top edge, 0.50 m at bottom)
#   isaac_x = -world_X + X_OFFSET
#   isaac_y =  world_Y + Y_OFFSET
#
# Residuals: exact fit (4 pts each — minimal set)

_H_REAL = np.array([
    [+1.0641627654969605e-03, +4.4363254411092787e-05, +0.0000000000000000e+00],
    [+0.0000000000000000e+00, +1.1090813850672171e-03, +0.0000000000000000e+00],
    [+0.0000000000000000e+00, +1.3048009790061474e-04, +1.0000000000000000e+00],
], dtype=np.float64)

_H_REAL_INV = np.linalg.inv(_H_REAL)


def pix2world_real(col: float, row: float):
    """Convert real-camera pixel (col, row) -> world (X, Y) in metres."""
    import cv2 as _cv2_h
    pt  = np.array([[[float(col), float(row)]]], dtype=np.float32)
    out = _cv2_h.perspectiveTransform(pt, _H_REAL.astype(np.float32))
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def world2pix_real(wx: float, wy: float):
    """Convert world (X, Y) -> real-camera pixel (col, row)."""
    import cv2 as _cv2_h
    pt  = np.array([[[float(wx), float(wy)]]], dtype=np.float32)
    out = _cv2_h.perspectiveTransform(pt, _H_REAL_INV.astype(np.float32))
    return float(out[0, 0, 0]), float(out[0, 0, 1])


# Aliases matching existing API
_pix2world = pix2world_real   # used throughout this file

# ── Constants ─────────────────────────────────────────────────────────────────
OUT_DIR  = os.path.expanduser("~/so101-sim-project")
os.makedirs(OUT_DIR, exist_ok=True)

CAM_PATH = "/World/envs/env_0/CameraOverhead"

# Camera parameters — single source of truth is cylinder_to_target_env_cfg.py.
# Import from there; fall back to the same values if the import fails at startup.
try:
    from sim_to_real_so101.tasks.cylinder_to_target_env_cfg import (
        CAMERA_X              as CAM_X,
        CAMERA_Y              as CAM_Y,
        CAMERA_HEIGHT_ABOVE_TABLE as CAM_Z,
        overhead_camera       as _cam_cfg,
    )
except Exception:
    CAM_X, CAM_Y, CAM_Z = -0.025, 0.383, 0.538  # env_cfg fallback (CAMERA_X, CAMERA_Y, CAMERA_HEIGHT_ABOVE_TABLE)
    _cam_cfg = None

_CYL_HSV_LO = np.array(( 35,  40,  40), dtype=np.uint8)
_CYL_HSV_HI = np.array(( 85, 255, 255), dtype=np.uint8)
_CUP_HSV_LO = np.array(( 15,  50,  50), dtype=np.uint8)   # yellow cup (orange-yellow hue)
_CUP_HSV_HI = np.array(( 38, 255, 255), dtype=np.uint8)

CAPTURE_FRAME = _cf  # set via --capture_frame N (default 150 ≈ 5 s at 30 fps)

# ── Annotation toggles (set False to hide, True to show) ─────────────────────
SHOW_TARGET_CROSSHAIR = False   # green circle+lines at 3D-projected real-world position
SHOW_ORIGIN_MARKER    = True    # white diamond at Isaac (0,0,0) — SIM world origin reference
SHOW_ISAAC_DOT        = False   # red dot at Isaac FK cylinder position

# ── Manual crosshair override (SIM image calibration aid) ────────────────────
# Set CROSSHAIR_OVERRIDE = True and tune _X / _Y in Isaac metres until the
# green crosshair sits exactly over the cylinder blob in the SIM image.
# Once aligned the values give you the true Isaac position for that cylinder
# (use them to verify / refine REAL_TO_ISAAC_X_OFFSET_M / Y_OFFSET_M).
# Set CROSSHAIR_OVERRIDE = False to revert to the auto-computed position.
CROSSHAIR_OVERRIDE          = False
#CROSSHAIR_OVERRIDE_ISAAC_X_M = -0.2699   # metres; positive = robot-left / SIM-left
#CROSSHAIR_OVERRIDE_ISAAC_Y_M =  0.3304   # metres; positive = farther from robot
CROSSHAIR_OVERRIDE_ISAAC_X_M =  0.0   # metres; positive = robot-left / SIM-left
CROSSHAIR_OVERRIDE_ISAAC_Y_M =  0.40  # metres; 0.40 m is well within camera view (~row 300)

# ── Isaac coordinate grid overlay ────────────────────────────────────────────
# Draws a grid in Isaac Lab coordinates over the SIM image.
# Isaac X: robot-right is negative, robot-left is positive (Rz180).
# Isaac Y: distance forward from robot base.
# Grid lines every ISAAC_GRID_STEP_CM centimetres.
SHOW_ISAAC_GRID       = False
ISAAC_GRID_X_MIN_CM   = -70    # leftmost Isaac X column (cm)
ISAAC_GRID_X_MAX_CM   =  70    # rightmost Isaac X column (cm)
ISAAC_GRID_Y_MIN_CM   = -20    # nearest Isaac Y row (cm)
ISAAC_GRID_Y_MAX_CM   =  70    # farthest Isaac Y row (cm)
ISAAC_GRID_STEP_CM    =   5    # spacing between lines (cm)
SHOW_BLOB_DOT         = False   # blue circle at HSV blob centroid
SHOW_PLACEMENT_DOT    = False   # orange dot at homography-derived placement position
SHOW_STAGE_RED_SPHERES  = False # 3-D red USD spheres: DebugPlacementMarker + DebugOriginMarker

# ── USD graphpaper mat (Option 2) ────────────────────────────────────────────
# Loads mat.usda directly into the Isaac stage so the graphpaper texture appears
# in the SIM render by construction — no projection math, correct scale guaranteed.
# The mat's LOCAL coords (X:−0.40..+0.40, Y:+0.210..−0.490) are placed into Isaac
# WORLD frame (X:−0.435..+0.365, Y:0.000..+0.700) via:
#   world = Rz180 @ local + translate(−0.035, +0.210, 0.0)
# This matches the runtime AABB observed in ep13.
ADD_USD_MAT   = False
MAT_USD_FILE  = "/home/ted/so101-sim-project/mat.usda"   # ← adjust path if needed

# ── USD direct grid lines (drawn in Isaac world frame) ───────────────────────
# Draws thin BasisCurves lines at exact 10 cm intervals in Isaac world space.
# No texture, no UV mapping — lines are metrically correct by construction.
# Gold = constant-Y lines (run in X), Cyan = constant-X lines (run in Y).
# White = Y=0 axis line (robot near edge).
# Covers mat AABB: X=-0.435..+0.365, Y=0.000..+0.700, step=0.10 m.
DRAW_ISAAC_GRID_LINES = True

# ── Real-world coordinate grid overlay (drawn on the REAL image) ─────────────
# Projects a regular grid of real-world (X, Y) positions back to real camera
# pixels via _H_REAL_INV so you can see how the pixel→world mapping works.
# Constant-Y lines are gold, constant-X lines are cyan, zero axes are white.
# Every other labelled line keeps the annotation readable.
SHOW_REAL_WORLD_GRID    = True
REAL_GRID_X_MIN_M       = -0.05   # leftmost world X to draw (metres); origin at TL corner
REAL_GRID_X_MAX_M       =  0.75   # rightmost world X to draw (metres)
REAL_GRID_Y_MIN_M       = -0.05   # nearest world Y to draw (metres)
REAL_GRID_Y_MAX_M       =  0.55   # farthest world Y to draw (metres)
REAL_GRID_STEP_M        =  0.05   # line spacing (metres); 0.05 = every 5 cm

# ── SIM Isaac-frame grid overlay (drawn on the SIM image) ────────────────────
# Draws a grid directly in Isaac Lab world-frame coordinates (metres), fed
# straight into project_isaac_3d_to_sim_pixel — no real_world_to_isaac
# conversion.  Lines align with the mat's grey squares and the arm's reach
# because the grid IS Isaac's coordinate system.
# Extents from mat.usda: X -0.40..+0.40 (80 cm), Y -0.490..+0.210 (70 cm).
# Colour: constant-X = cyan (255,200,0), constant-Y = gold (0,200,255), axes = white.
SHOW_SIM_WORLD_GRID     = False
SIM_GRID_X_MIN_M        = -0.435  # mat left edge (Isaac AABB); lines at −43.5,−33.5,−23.5...+36.5cm — on physical squares ✓
SIM_GRID_X_MAX_M        =  0.365  # mat right edge (Isaac AABB); lines hit actual mat square boundaries ✓
SIM_GRID_Y_MIN_M        =  0.00   # mat near edge = Y=0.000; snapped to 10cm multiple so lines hit 0,10,20...cm ✓
SIM_GRID_Y_MAX_M        =  0.70   # mat far edge = Y=0.700; snapped to 10cm multiple ✓
SIM_GRID_STEP_M         =  0.10  # line spacing (metres) = 10 cm
SIM_GRID_SAMPLES        =  30     # points sampled per line to capture perspective curvature


# ── Real-world → Isaac calibration offsets ───────────────────────────────────
# _H_REAL origin (2026-10-12 recalibration): world (0,0) = top-left pixel of
# the real image.  These offsets map that world frame to Isaac world coords.
#
# Derivation (episode 13, frame 0):
#   Same cylinder detected at pixel ≈ (12, 50) in real image.
#   Old _H_REAL gave real_world = (-0.315, +0.046) m.
#   New _H_REAL gives real_world = (+0.015, +0.055) m  (origin now at TL corner).
#   Isaac reset pos: Isaac = (+0.253, +0.177)  (from env.reset() dataset value).
#   Both are the same physical cylinder → solve for offsets:
#     Isaac_X = -real_X + X_OFFSET  →  -0.015 + X_OFFSET = 0.253  →  X_OFFSET = +0.268 m
#     Isaac_Y =  real_Y + Y_OFFSET  →   0.055 + Y_OFFSET = 0.177  →  Y_OFFSET = +0.122 m
#
# These are estimated from one episode.  Refine by averaging over multiple episodes.
#REAL_TO_ISAAC_X_OFFSET_M = +0.30  # derived 2026-10-14: real TL(0,0) → Isaac(+0.30,+0.10)    # metres; added to (-real_X)
#REAL_TO_ISAAC_Y_OFFSET_M = +0.09  # derived 2026-10-15: refined from +0.10 — both cylinder and cup were 1 cm too high in Isaac Y
REAL_TO_ISAAC_X_OFFSET_M = +0.31  # adjusted by eye.  increased X moves SIM left
REAL_TO_ISAAC_Y_OFFSET_M = +0.090  # adjusted by eye.  increased Y moves SIM down

def real_world_to_isaac(wx, wy):
    """Convert _H_REAL world coords to Isaac world coords.

    Applies Rz180 sign-flip AND the measured calibration origin offset.
    Returns (isaac_x, isaac_y) in metres.
    """
    return (
        float(-wx + REAL_TO_ISAAC_X_OFFSET_M),
        float( wy + REAL_TO_ISAAC_Y_OFFSET_M),
    )



# ─────────────────────────────────────────────────────────────────────────────
# STEP 1-3: detect cylinder in the real overhead video
# ─────────────────────────────────────────────────────────────────────────────
def detect_cylinder_in_real(real_root, episode_idx):
    import cv2, subprocess

    # Try per-episode mp4 first (offset always 0.0).
    # Fall back to the chunk file and look up the episode's start timestamp
    # from the meta parquet — same pattern as replay_interactive._det_get_video.
    video_path = None
    vid_offset = 0.0
    for pat in [
        f"videos/observation.images.overhead/chunk-000/episode_{episode_idx:06d}.mp4",
        f"videos/observation.images.overhead/chunk-{episode_idx // 500:03d}/episode_{episode_idx:06d}.mp4",
    ]:
        p = os.path.join(real_root, pat)
        if os.path.exists(p):
            video_path = p
            break

    if video_path is None:
        _chunk_vid = os.path.join(
            real_root,
            "videos/observation.images.overhead/chunk-000/file-000.mp4",
        )
        if os.path.exists(_chunk_vid):
            video_path = _chunk_vid
            # Look up start timestamp for this episode in the meta parquet.
            try:
                _meta_pq = os.path.join(
                    real_root, "meta/episodes/chunk-000/file-000.parquet"
                )
                _ep_meta = pd.read_parquet(_meta_pq)
                _row = _ep_meta[_ep_meta["episode_index"] == episode_idx]
                if not _row.empty:
                    vid_offset = float(
                        _row["videos/observation.images.overhead/from_timestamp"].iloc[0]
                    )
                    print(f"[REAL] Episode {episode_idx} start timestamp: {vid_offset:.3f}s")
                else:
                    print(f"[REAL] Warning: episode {episode_idx} not in meta parquet — using offset 0.0")
            except Exception as _te:
                print(f"[REAL] Warning: could not read episode timestamps: {_te}")

    if video_path is None:
        raise RuntimeError(f"No video found for episode {episode_idx} under {real_root}")
    print(f"[REAL] Video: {video_path}")

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe.stderr[:200]}")
    w, h = map(int, probe.stdout.strip().split(","))

    cmd = ["ffmpeg", "-ss", str(vid_offset), "-i", video_path,
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True)
    expected = w * h * 3
    if r.returncode != 0 or len(r.stdout) != expected:
        raise RuntimeError(
            f"ffmpeg failed: ret={r.returncode} got={len(r.stdout)} expected={expected}"
        )
    frame = np.frombuffer(r.stdout, dtype=np.uint8).reshape(h, w, 3).copy()
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _CYL_HSV_LO, _CYL_HSV_HI)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No green contours found in real frame-0")
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        raise RuntimeError(f"Green contour too small: area={cv2.contourArea(c):.0f}")
    M = cv2.moments(c)
    if M["m00"] == 0:
        raise RuntimeError("Zero-area moment on green contour")
    px = int(M["m10"] / M["m00"])
    py = int(M["m01"] / M["m00"])
    cyl_bx, cyl_by, cyl_bw, cyl_bh = cv2.boundingRect(c)
    # Strip the 5 px dilation padding so the box hugs the true object edge
    _dil = 4
    cyl_bx += _dil; cyl_by += _dil; cyl_bw -= 2 * _dil; cyl_bh -= 2 * _dil
    print(f"[REAL] Cylinder pixel centroid: ({px}, {py})")
    # ── Parallax correction: anchor corner + 9.6 px/cm offset ────────────────
    # Bottom half (py >= 240): UR corner, -1.5 cm col, +1.5 cm row
    # Top half    (py <  240): BR corner, -1.5 cm col, -1.5 cm row
    _cyl_off_px = round(1.5 * 9.6)  # 14 px
    if py >= 240:
        _cyl_anchor_col, _cyl_anchor_row = cyl_bx + cyl_bw, cyl_by          # UR
        _cyl_corr_col = _cyl_anchor_col - _cyl_off_px
        _cyl_corr_row = _cyl_anchor_row + _cyl_off_px
    else:
        _cyl_anchor_col, _cyl_anchor_row = cyl_bx + cyl_bw, cyl_by + cyl_bh  # BR
        _cyl_corr_col = _cyl_anchor_col - _cyl_off_px
        _cyl_corr_row = _cyl_anchor_row - _cyl_off_px
    wx, wy = _pix2world(_cyl_corr_col, _cyl_corr_row)
    print(f"[REAL] Cylinder anchor pixel:   ({_cyl_anchor_col},{_cyl_anchor_row})  corrected pixel: ({_cyl_corr_col},{_cyl_corr_row})")
    print(f"[REAL] Cylinder world coords:   ({wx:+.4f}, {wy:.4f})  metres  [parallax-corrected]")

    # ── Detect yellow cup in the RIGHT half of the image ─────────────────────
    # The cup sits on the right side of the mat; restricting to the right half
    # avoids false-positive matches on the yellow-tinted robot arm on the left.
    cup_px, cup_py, cup_wx, cup_wy = None, None, None, None
    cup_bx, cup_by, cup_bw, cup_bh = None, None, None, None
    try:
        right_mask_region = hsv.copy()
        right_mask_region[:, :w // 2] = 0   # zero-out left half before detection
        cup_mask = cv2.inRange(right_mask_region, _CUP_HSV_LO, _CUP_HSV_HI)
        cup_mask = cv2.morphologyEx(cup_mask, cv2.MORPH_OPEN,   k)
        cup_mask = cv2.morphologyEx(cup_mask, cv2.MORPH_DILATE, k)
        cup_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cup_contours:
            cc = max(cup_contours, key=cv2.contourArea)
            if cv2.contourArea(cc) >= 50:
                Mc = cv2.moments(cc)
                if Mc["m00"] != 0:
                    cup_px = int(Mc["m10"] / Mc["m00"])
                    cup_py = int(Mc["m01"] / Mc["m00"])
                    cup_bx, cup_by, cup_bw, cup_bh = cv2.boundingRect(cc)
                    # Strip the 5 px dilation padding
                    cup_bx += _dil; cup_by += _dil; cup_bw -= 2 * _dil; cup_bh -= 2 * _dil
                    print(f"[REAL] Cup pixel centroid:  ({cup_px}, {cup_py})")
                    # ── Parallax correction: anchor corner + 9.6 px/cm offset ──────────
                    # Top half    (cup_py < 240):  BL corner, +4.5 cm col, -4.5 cm row
                    # Bottom half (cup_py >= 240): UL corner, +4.5 cm col, +4.5 cm row
                    _cup_off_px = round(4.5 * 9.6)  # 43 px
                    if cup_py < 240:
                        _cup_anchor_col, _cup_anchor_row = cup_bx, cup_by + cup_bh  # BL
                        _cup_corr_col = _cup_anchor_col + _cup_off_px
                        _cup_corr_row = _cup_anchor_row - _cup_off_px
                    else:
                        _cup_anchor_col, _cup_anchor_row = cup_bx, cup_by           # UL
                        _cup_corr_col = _cup_anchor_col + _cup_off_px
                        _cup_corr_row = _cup_anchor_row + _cup_off_px
                    cup_wx, cup_wy = _pix2world(_cup_corr_col, _cup_corr_row)
                    print(f"[REAL] Cup anchor pixel:    ({_cup_anchor_col},{_cup_anchor_row})  corrected pixel: ({_cup_corr_col},{_cup_corr_row})")
                    print(f"[REAL] Cup world coords:    ({cup_wx:+.4f}, {cup_wy:.4f})  metres  [parallax-corrected]")
                else:
                    print("[REAL] Cup: zero-area moment — skipping")
            else:
                print(f"[REAL] Cup: largest yellow contour too small (area={cv2.contourArea(cc):.0f}) — skipping")
        else:
            print("[REAL] Cup: no yellow contours found in right half — cup not detected")
    except Exception as _cup_e:
        print(f"[REAL] Cup detection failed: {_cup_e}")

    return frame, px, py, wx, wy, cyl_bx, cyl_by, cyl_bw, cyl_bh, cup_px, cup_py, cup_wx, cup_wy, cup_bx, cup_by, cup_bw, cup_bh


# STEP 4
def save_annotated_real(frame_bgr, px, py, wx, wy, episode_idx,
                        cyl_bx=None, cyl_by=None, cyl_bw=None, cyl_bh=None,
                        cup_px=None, cup_py=None, cup_wx=None, cup_wy=None,
                        cup_bx=None, cup_by=None, cup_bw=None, cup_bh=None):
    import cv2
    out = frame_bgr.copy()
    h_img, w_img = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Cylinder bounding box (green) ─────────────────────────────────────────
    if cyl_bx is not None:
        cv2.rectangle(out, (cyl_bx, cyl_by), (cyl_bx + cyl_bw, cyl_by + cyl_bh),
                      (0, 255, 0), 1)
        cv2.putText(out, "cyl bbox", (cyl_bx, cyl_by - 4),
                    font, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
        _cul = pix2world_real(cyl_bx,          cyl_by)
        _cur = pix2world_real(cyl_bx + cyl_bw, cyl_by)
        _cbl = pix2world_real(cyl_bx,          cyl_by + cyl_bh)
        _cbr = pix2world_real(cyl_bx + cyl_bw, cyl_by + cyl_bh)
        print(f"[REAL] Cylinder bbox  UL=({cyl_bx},{cyl_by}) {_cul[0]:+.4f},{_cul[1]:.4f}m  "
              f"UR=({cyl_bx+cyl_bw},{cyl_by}) {_cur[0]:+.4f},{_cur[1]:.4f}m")
        print(f"[REAL] Cylinder bbox  BL=({cyl_bx},{cyl_by+cyl_bh}) {_cbl[0]:+.4f},{_cbl[1]:.4f}m  "
              f"BR=({cyl_bx+cyl_bw},{cyl_by+cyl_bh}) {_cbr[0]:+.4f},{_cbr[1]:.4f}m")

    # ── Cylinder anchor star (white) ─────────────────────────────────────────
    # Bottom half (py >= 240) → UR corner;  Top half (py < 240) → BR corner
    if cyl_bx is not None:
        _cyl_ax, _cyl_ay = (cyl_bx + cyl_bw, cyl_by) if py >= 240 \
                           else (cyl_bx + cyl_bw, cyl_by + cyl_bh)
        _s = 5
        cv2.line(out, (_cyl_ax-_s, _cyl_ay),    (_cyl_ax+_s, _cyl_ay),    (255,255,255), 1, cv2.LINE_AA)
        cv2.line(out, (_cyl_ax,    _cyl_ay-_s),  (_cyl_ax,    _cyl_ay+_s), (255,255,255), 1, cv2.LINE_AA)
        cv2.line(out, (_cyl_ax-_s, _cyl_ay-_s),  (_cyl_ax+_s, _cyl_ay+_s),(255,255,255), 1, cv2.LINE_AA)
        cv2.line(out, (_cyl_ax+_s, _cyl_ay-_s),  (_cyl_ax-_s, _cyl_ay+_s),(255,255,255), 1, cv2.LINE_AA)
        print(f"[REAL] Cylinder anchor: {'UR' if py>=240 else 'BR'} pixel ({_cyl_ax},{_cyl_ay})")

    # ── Cylinder corrected-position star (white, larger) ─────────────────────
    # Apply cm offsets in pixel space: 1 cm = 9.6 px, 1.5 cm = 14 px
    # Bottom half: UR anchor → -14px col, +14px row
    # Top half:    BR anchor → -14px col, -14px row
    if cyl_bx is not None:
        _cyl_off = round(1.5 * 9.6)  # 14 px
        _cyl_cx = _cyl_ax - _cyl_off
        _cyl_cy = _cyl_ay + _cyl_off if py >= 240 else _cyl_ay - _cyl_off
        _s2 = 7
        cv2.line(out, (_cyl_cx-_s2, _cyl_cy),     (_cyl_cx+_s2, _cyl_cy),     (255,255,255), 2, cv2.LINE_AA)
        cv2.line(out, (_cyl_cx,     _cyl_cy-_s2), (_cyl_cx,     _cyl_cy+_s2), (255,255,255), 2, cv2.LINE_AA)
        cv2.line(out, (_cyl_cx-_s2, _cyl_cy-_s2), (_cyl_cx+_s2, _cyl_cy+_s2),(255,255,255), 2, cv2.LINE_AA)
        cv2.line(out, (_cyl_cx+_s2, _cyl_cy-_s2), (_cyl_cx-_s2, _cyl_cy+_s2),(255,255,255), 2, cv2.LINE_AA)
        print(f"[REAL] Cylinder corrected pixel: ({_cyl_cx},{_cyl_cy})")

    # ── Cylinder detection dot (green) ───────────────────────────────────────
    cv2.circle(out, (px, py),  3, (0, 255, 0), -1)
    # Label anchored to corrected star, not raw centroid, so it belongs to the right point
    if cyl_bx is not None:
        _cyl_lbl = f"cyl ({wx:+.3f}, {wy:.3f})"
        _cyl_lw, _ = cv2.getTextSize(_cyl_lbl, font, 0.55, 2)[0]
        _cyl_tx = (_cyl_cx - 10 - _cyl_lw) if (_cyl_cx + 10 + _cyl_lw > w_img) else (_cyl_cx + 10)
        cv2.putText(out, _cyl_lbl, (_cyl_tx, _cyl_cy), font, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(out, f"cyl ({wx:+.3f}, {wy:.3f})",
                    (px + 22, py), font, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    # ── Cup bounding box (orange) ─────────────────────────────────────────────
    if cup_bx is not None:
        cv2.rectangle(out, (cup_bx, cup_by), (cup_bx + cup_bw, cup_by + cup_bh),
                      (0, 140, 255), 1)
        cv2.putText(out, "cup bbox", (cup_bx, cup_by - 4),
                    font, 0.38, (0, 140, 255), 1, cv2.LINE_AA)
        _pul = pix2world_real(cup_bx,          cup_by)
        _pur = pix2world_real(cup_bx + cup_bw, cup_by)
        _pbl = pix2world_real(cup_bx,          cup_by + cup_bh)
        _pbr = pix2world_real(cup_bx + cup_bw, cup_by + cup_bh)
        print(f"[REAL] Cup bbox       UL=({cup_bx},{cup_by}) {_pul[0]:+.4f},{_pul[1]:.4f}m  "
              f"UR=({cup_bx+cup_bw},{cup_by}) {_pur[0]:+.4f},{_pur[1]:.4f}m")
        print(f"[REAL] Cup bbox       BL=({cup_bx},{cup_by+cup_bh}) {_pbl[0]:+.4f},{_pbl[1]:.4f}m  "
              f"BR=({cup_bx+cup_bw},{cup_by+cup_bh}) {_pbr[0]:+.4f},{_pbr[1]:.4f}m")

    # ── Cup anchor star (white) ──────────────────────────────────────────────
    # Top half (cup_py < 240) → BL corner;  Bottom half (cup_py >= 240) → UL corner
    if cup_bx is not None and cup_py is not None:
        _cup_ax, _cup_ay = (cup_bx, cup_by + cup_bh) if cup_py < 240 \
                           else (cup_bx, cup_by)
        _s = 5
        cv2.line(out, (_cup_ax-_s, _cup_ay),    (_cup_ax+_s, _cup_ay),    (255,255,255), 1, cv2.LINE_AA)
        cv2.line(out, (_cup_ax,    _cup_ay-_s),  (_cup_ax,    _cup_ay+_s), (255,255,255), 1, cv2.LINE_AA)
        cv2.line(out, (_cup_ax-_s, _cup_ay-_s),  (_cup_ax+_s, _cup_ay+_s),(255,255,255), 1, cv2.LINE_AA)
        cv2.line(out, (_cup_ax+_s, _cup_ay-_s),  (_cup_ax-_s, _cup_ay+_s),(255,255,255), 1, cv2.LINE_AA)
        print(f"[REAL] Cup anchor:      {'BL' if cup_py<240 else 'UL'} pixel ({_cup_ax},{_cup_ay})")

    # ── Cup corrected-position star (white, larger) ───────────────────────────
    # Apply cm offsets in pixel space: 1 cm = 9.6 px, 4.5 cm = 43 px
    # Top half:    BL anchor → +43px col, -43px row
    # Bottom half: UL anchor → +43px col, +43px row
    if cup_bx is not None and cup_py is not None:
        _cup_off = round(4.5 * 9.6)  # 43 px
        _cup_cx = _cup_ax + _cup_off
        _cup_cy = _cup_ay - _cup_off if cup_py < 240 else _cup_ay + _cup_off
        _s2 = 7
        cv2.line(out, (_cup_cx-_s2, _cup_cy),     (_cup_cx+_s2, _cup_cy),     (255,255,255), 2, cv2.LINE_AA)
        cv2.line(out, (_cup_cx,     _cup_cy-_s2), (_cup_cx,     _cup_cy+_s2), (255,255,255), 2, cv2.LINE_AA)
        cv2.line(out, (_cup_cx-_s2, _cup_cy-_s2), (_cup_cx+_s2, _cup_cy+_s2),(255,255,255), 2, cv2.LINE_AA)
        cv2.line(out, (_cup_cx+_s2, _cup_cy-_s2), (_cup_cx-_s2, _cup_cy+_s2),(255,255,255), 2, cv2.LINE_AA)
        print(f"[REAL] Cup corrected pixel:      ({_cup_cx},{_cup_cy})")

    # ── Yellow cup detection circle (orange) ──────────────────────────────────
    if cup_px is not None and cup_py is not None:
        cv2.circle(out, (cup_px, cup_py),  3, (0, 140, 255), -1)   # BGR orange dot
        _cup_lbl = f"cup ({cup_wx:+.3f}, {cup_wy:.3f})"
        _cup_lw, _ = cv2.getTextSize(_cup_lbl, font, 0.55, 2)[0]
        # Label anchored to corrected star, not raw centroid
        if cup_bx is not None:
            _cup_tx = (_cup_cx - 10 - _cup_lw) if (_cup_cx + 10 + _cup_lw > w_img) else (_cup_cx + 10)
            cv2.putText(out, _cup_lbl, (_cup_tx, _cup_cy), font, 0.55, (0, 140, 255), 2, cv2.LINE_AA)
        else:
            _cup_tx = (cup_px - 22 - _cup_lw) if (cup_px + 22 + _cup_lw > w_img) else (cup_px + 22)
            cv2.putText(out, _cup_lbl, (_cup_tx, cup_py), font, 0.55, (0, 140, 255), 2, cv2.LINE_AA)


    # ── Real-world coordinate grid ────────────────────────────────────────────
    # Back-project a regular (world_X, world_Y) grid through _H_REAL_INV to
    # real camera pixels.  Because _H_REAL_INV is a homography (planar
    # perspective transform), world straight lines map to straight lines in
    # the image, so we only need to project the two endpoints of each grid line.
    # Constant-X lines are drawn in cyan, constant-Y lines in gold,
    # axis lines (X=0, Y=0) in white.  Labels are placed at the image edge.
    if SHOW_REAL_WORLD_GRID:
        _GX_MIN = REAL_GRID_X_MIN_M
        _GX_MAX = REAL_GRID_X_MAX_M
        _GY_MIN = REAL_GRID_Y_MIN_M
        _GY_MAX = REAL_GRID_Y_MAX_M
        _GS     = REAL_GRID_STEP_M

        def _gvals(lo, hi, step):
            vals, v = [], lo
            while v <= hi + step * 0.01:
                vals.append(round(v, 6))
                v += step
            return vals

        _xs_m = _gvals(_GX_MIN, _GX_MAX, _GS)
        _ys_m = _gvals(_GY_MIN, _GY_MAX, _GS)

        def _world_to_real_px(wx_m, wy_m):
            """Back-project world (metres) → real image pixel (col, row) via _H_REAL_INV."""
            pt = _H_REAL_INV @ np.array([wx_m, wy_m, 1.0])
            if abs(pt[2]) < 1e-9:
                return None
            return int(round(pt[0] / pt[2])), int(round(pt[1] / pt[2]))

        def _clip_to_frame(p1, p2):
            """Clip line segment p1→p2 to the image rect; returns clipped pair or None."""
            ok, _p1c, _p2c = cv2.clipLine((0, 0, w_img, h_img), p1, p2)
            return (_p1c, _p2c) if ok else None

        # Collect projected intersection points so drop-lines can snap to them
        _grid_pts = {}   # (wx_idx, wy_idx) → (px, py) or None

        # --- Draw constant-X lines (cyan / white for X=0)
        for _wx_m in _xs_m:
            _is_x0    = abs(_wx_m) < 1e-6
            _line_col = (255, 255, 255) if _is_x0 else (255, 200, 0)
            _thick    = 2 if _is_x0 else 1
            _p0 = _world_to_real_px(_wx_m, _GY_MIN)
            _p1 = _world_to_real_px(_wx_m, _GY_MAX)
            if _p0 is None or _p1 is None:
                continue
            _seg = _clip_to_frame(_p0, _p1)
            if _seg is None:
                continue
            cv2.line(out, _seg[0], _seg[1], _line_col, _thick, cv2.LINE_AA)
            # Label near the top of the line
            _lbl_x = f"X={_wx_m*100:+.0f}cm"
            _lw2, _lh2 = cv2.getTextSize(_lbl_x, font, 0.32, 1)[0]
            _lx2 = max(2, min(_seg[0][0] - _lw2 // 2, w_img - _lw2 - 2))
            _ly2 = max(_lh2 + 3, _seg[0][1] + _lh2 + 2)
            cv2.putText(out, _lbl_x, (_lx2, _ly2),
                        font, 0.32, _line_col, 1, cv2.LINE_AA)

        # --- Draw constant-Y lines (gold / white for Y=0)
        for _wy_m in _ys_m:
            _is_y0    = abs(_wy_m) < 1e-6
            _line_col = (255, 255, 255) if _is_y0 else (0, 200, 255)
            _thick    = 2 if _is_y0 else 1
            _p0 = _world_to_real_px(_GX_MIN, _wy_m)
            _p1 = _world_to_real_px(_GX_MAX, _wy_m)
            if _p0 is None or _p1 is None:
                continue
            _seg = _clip_to_frame(_p0, _p1)
            if _seg is None:
                continue
            cv2.line(out, _seg[0], _seg[1], _line_col, _thick, cv2.LINE_AA)
            # Label on the right edge
            _lbl_y = f"Y={_wy_m*100:.0f}cm"
            _lw3, _lh3 = cv2.getTextSize(_lbl_y, font, 0.32, 1)[0]
            _ey3 = max(_lh3 + 2, min(_seg[1][1] - 2, h_img - 2))
            cv2.putText(out, _lbl_y, (w_img - _lw3 - 4, _ey3),
                        font, 0.32, _line_col, 1, cv2.LINE_AA)

        # --- Dots at each visible intersection
        for _wx_m in _xs_m:
            for _wy_m in _ys_m:
                _ipt = _world_to_real_px(_wx_m, _wy_m)
                _grid_pts[(_wx_m, _wy_m)] = _ipt
                if _ipt is None:
                    continue
                _ipx, _ipy = _ipt
                if not (0 <= _ipx < w_img and 0 <= _ipy < h_img):
                    continue
                _dc = (255, 255, 255) if (abs(_wx_m) < 1e-6 and abs(_wy_m) < 1e-6) else (180, 180, 180)
                cv2.circle(out, (_ipx, _ipy), 3 if abs(_wx_m) < 1e-6 and abs(_wy_m) < 1e-6 else 2, _dc, -1)


        print(f"[REAL] World-coord grid drawn: "
              f"X={_GX_MIN*100:.0f}..{_GX_MAX*100:.0f}cm  "
              f"Y={_GY_MIN*100:.0f}..{_GY_MAX*100:.0f}cm  "
              f"step={_GS*100:.0f}cm")


    path = os.path.join(OUT_DIR, f"debug_real_ep{episode_idx:03d}_cylinder.png")
    cv2.imwrite(path, out)
    print(f"[REAL] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SIM helpers
# ─────────────────────────────────────────────────────────────────────────────
def configure_overhead_camera(stage):
    import omni.kit.viewport.utility as vp_util
    from pxr import UsdGeom, Gf, Sdf
    for old in ["/World/envs/env_0/DebugSideCamera", "/World/envs/env_0/DebugPerspCamera"]:
        if stage.GetPrimAtPath(old).IsValid():
            stage.RemovePrim(Sdf.Path(old))
    cam_prim = stage.GetPrimAtPath(CAM_PATH)
    if not cam_prim.IsValid():
        print(f"[CAM] WARNING: {CAM_PATH} not found")
        return
    xf  = UsdGeom.Xformable(cam_prim)
    ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
    for t in ("xformOp:translate", "xformOp:translation"):
        if t in ops:
            ops[t].Set(Gf.Vec3d(CAM_X, CAM_Y, CAM_Z)); break
    # Read rotation from env_cfg (Rz180° + tilt) — same as replay_interactive.
    if _cam_cfg is not None:
        _rot = _cam_cfg.offset.rot   # scipy array [x, y, z, w]
        qx, qy, qz, qw = float(_rot[0]), float(_rot[1]), float(_rot[2]), float(_rot[3])
    else:
        # Rz(180°) + -4° X tilt hardcoded fallback
        qx, qy, qz, qw = 0.0, 0.034899, 0.999391, 0.0
    print(f"[CAM] applying rot Gf.Quatd(w={qw:.5f}, x={qx:.5f}, y={qy:.5f}, z={qz:.5f})")
    for r in ("xformOp:orient", "xformOp:orientation"):
        if r in ops:
            ops[r].Set(Gf.Quatd(qw, qx, qy, qz)); break
    if "xformOp:rotateXYZ" in ops:
        ops["xformOp:rotateXYZ"].Set(Gf.Vec3f(0.0, 0.0, 180.0))
    vp = vp_util.get_active_viewport()
    vp.camera_path = CAM_PATH
    # Match replay_interactive: force viewport texture resolution to 640×480
    # before the texture is applied so Kit composes the material at the same
    # render resolution both scripts use.
    try:
        vp.set_texture_resolution((640, 480))
        print("[CAM] Viewport texture resolution → 640×480")
    except Exception as _e:
        print(f"[CAM] set_texture_resolution failed: {_e}")
    print("[CAM] Viewport → CameraOverhead")




def place_cylinder_at_coords(env, wx, wy):
    """
    Move the SIM cylinder to the detected real-world coordinates.
    Coordinate convention: isaac_X = -world_X, isaac_Y = world_Y.
    The SIM camera uses Rz(180°) so the image is horizontally mirrored;
    Isaac +X appears on the LEFT in the SIM image. Negating world_X maps
    correctly: a cylinder at world_X=-0.313 (left in real) goes to
    isaac_X=+0.313 (left in SIM). Confirmed by visual observation.
    Writes to both PhysX and the USD authored xformOp:translate so RTX
    renders it at the correct position.

    Isaac Lab 4.x API:
      write_root_link_pose_to_sim(pose)  — pose shape (N,7): [x,y,z, qw,qx,qy,qz]
      write_root_com_velocity_to_sim(vel) — vel shape (N,6): [vx,vy,vz, wx,wy,wz]
      root_view                           — replaces root_physx_view
      root_link_pose_w                    — replaces root_state_w[:, :7]
    """
    import warnings as _w
    from pxr import UsdGeom, Gf
    import omni.usd

    _ix, _iy = real_world_to_isaac(wx, wy)
    xyz    = (_ix, _iy, 0.059)   # rest height — avoids contact-penetration XY drift
    device = env.unwrapped.device
    obj    = env.unwrapped.scene["cylinder"]

    # ── Read BEFORE (use new API, fall back to deprecated) ───────────────────
    with _w.catch_warnings():
        _w.simplefilter("ignore", DeprecationWarning)
        try:
            before = obj.data.root_link_pose_w[0, :3].tolist()
        except AttributeError:
            before = obj.data.root_state_w[0, :3].tolist()
    print(f"[PLACE] Cylinder before placement: Isaac=({before[0]:+.4f}, {before[1]:.4f}, {before[2]:.4f})")

    # ── Pose and velocity tensors ─────────────────────────────────────────────
    # PhysX pose convention: [x, y, z, qw, qx, qy, qz]
    # The last four values are a quaternion representing the object's orientation.
    # A quaternion encodes a 3D rotation as four numbers; identity (1,0,0,0) means
    # "no rotation" — the object sits in its default upright orientation.
    # The cylinder is rotationally symmetric so orientation never matters here,
    # and we always use identity.  If you were placing an oriented object (a box,
    # a tool) you would compute a non-identity quaternion for its rotation.
    pose_t = torch.tensor([[_ix, _iy, 0.059, 1.0, 0.0, 0.0, 0.0]],   # [x, y, z, qw, qx, qy, qz]
                          dtype=torch.float32, device=device)
    vel_t  = torch.zeros(1, 6, dtype=torch.float32, device=device)

    physx_ok = False

    # ── Attempt 1: Isaac Lab 4.x preferred API ────────────────────────────────
    try:
        obj.write_root_link_pose_to_sim(pose_t)
        obj.write_root_com_velocity_to_sim(vel_t)
        physx_ok = True
        print("[PLACE] write_root_link_pose_to_sim() succeeded")
    except Exception as _e1:
        print(f"[PLACE] write_root_link_pose_to_sim failed ({_e1})")

    # ── Attempt 2: root_view (replaces root_physx_view) ──────────────────────
    if not physx_ok:
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore", DeprecationWarning)
                _view = obj.root_view
            # set_transforms: (N,7) [x,y,z, qw,qx,qy,qz] — PhysX warp view
            try:
                _xform = pose_t.cpu().numpy()   # already [x,y,z,qw,qx,qy,qz]
                _view.set_transforms(_xform, indices=None)
                _view.set_velocities(vel_t.cpu().numpy(), indices=None)
                physx_ok = True
                print("[PLACE] root_view.set_transforms() succeeded")
            except Exception as _e2a:
                print(f"[PLACE] root_view.set_transforms failed ({_e2a})")
            if not physx_ok:
                # set_world_poses variant used by some Isaac Lab versions
                try:
                    pos_np = pose_t[:, :3].cpu().numpy()
                    rot_np = pose_t[:, 3:].cpu().numpy()   # [qw,qx,qy,qz]
                    _view.set_world_poses(positions=pos_np, orientations=rot_np,
                                         indices=None)
                    physx_ok = True
                    print("[PLACE] root_view.set_world_poses() succeeded")
                except Exception as _e2b:
                    print(f"[PLACE] root_view.set_world_poses failed ({_e2b})")
        except Exception as _e2:
            print(f"[PLACE] root_view access failed ({_e2})")

    # ── Attempt 3: legacy write_root_state_to_sim ────────────────────────────
    if not physx_ok:
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore", DeprecationWarning)
                state = obj.data.root_state_w.clone()
                state[:, :3]  = torch.tensor([list(xyz)], dtype=torch.float32, device=device)
                state[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
                state[:, 7:]  = 0.0
                obj.write_root_state_to_sim(state)
            print("[PLACE] write_root_state_to_sim() legacy fallback used")
        except Exception as _e3:
            print(f"[PLACE] write_root_state_to_sim failed ({_e3})")

    # ── USD authored translate so RTX render agrees with PhysX ───────────────
    stage = omni.usd.get_context().get_stage()
    # Resolve cylinder prim path: try root_view, then root_physx_view, then cfg
    try:
        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            try:
                _prim_paths = obj.root_view.prim_paths
            except AttributeError:
                _prim_paths = obj.root_physx_view.prim_paths
        leaf_path = _prim_paths[0]
        prim_path = "/".join(leaf_path.split("/")[:5])   # /World/envs/env_0/Cylinder
    except Exception:
        prim_path = obj.cfg.prim_path.replace("{ENV_IDX}", "0")
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        xf = UsdGeom.Xformable(prim)
        t_op = next(
            (op for op in xf.GetOrderedXformOps()
             if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
            None,
        )
        if t_op is None:
            t_op = xf.AddTranslateOp()
        t_op.Set(Gf.Vec3d(*xyz))
        print(f"[PLACE] USD translate → {prim_path}  Isaac=({_ix:+.4f}, {_iy:.4f}, 0.059)")
    else:
        print(f"[PLACE] WARNING: prim not found at {prim_path}")

    # ── Read AFTER to confirm PhysX accepted the new position ────────────────
    with _w.catch_warnings():
        _w.simplefilter("ignore", DeprecationWarning)
        try:
            after = obj.data.root_link_pose_w[0, :3].tolist()
        except AttributeError:
            after = obj.data.root_state_w[0, :3].tolist()
    print(f"[PLACE] Cylinder after placement:  Isaac=({after[0]:+.4f}, {after[1]:.4f}, {after[2]:.4f})")
    exp = (_ix, _iy, 0.059)
    ok  = all(abs(after[i] - exp[i]) < 0.005 for i in range(3))
    print(f"[PLACE] Position accepted by PhysX: {'YES ✓' if ok else 'NO — cylinder did NOT move, check API'}")

    # ── Red placement marker: visual-only USD Sphere at the placed position ────
    # This renders as a red dot in the camera image at the exact Isaac position
    # we commanded. It doesn't move with physics, so it stays put as a reference
    # even if the cylinder drifts. Compare it against the blue blob to measure drift.
    if SHOW_STAGE_RED_SPHERES:
        try:
            from pxr import Sdf, Vt
            _mp = Sdf.Path("/World/DebugPlacementMarker")
            if stage.GetPrimAtPath(_mp).IsValid():
                stage.RemovePrim(_mp)
            _sp = UsdGeom.Sphere.Define(stage, _mp)
            _sp.GetRadiusAttr().Set(0.010)                                   # 1 cm dot
            _sp.GetDisplayColorAttr().Set(Vt.Vec3fArray([(1.0, 0.15, 0.15)]))
            UsdGeom.Xformable(_sp).AddTranslateOp().Set(Gf.Vec3d(*xyz))
            print(f"[PLACE] Red sphere marker → /World/DebugPlacementMarker  Isaac=({xyz[0]:+.4f},{xyz[1]:.4f})")
        except Exception as _me:
            print(f"[PLACE] Red marker skipped: {_me}")


def place_cup_at_coords(env, wx, wy):
    """
    Move the SIM target cup to the detected real-world coordinates.
    Mirrors place_cylinder_at_coords() but targets scene["target"] at z=0.060
    (the cup sits slightly higher than the cylinder).
    """
    import warnings as _w
    from pxr import UsdGeom, Gf
    import omni.usd

    _ix, _iy = real_world_to_isaac(wx, wy)
    xyz    = (_ix, _iy, 0.060)
    device = env.unwrapped.device
    obj    = env.unwrapped.scene["target"]

    # ── Read BEFORE ──────────────────────────────────────────────────────────
    with _w.catch_warnings():
        _w.simplefilter("ignore", DeprecationWarning)
        try:
            before = obj.data.root_link_pose_w[0, :3].tolist()
        except AttributeError:
            before = obj.data.root_state_w[0, :3].tolist()
    print(f"[CUP] Cup before placement: Isaac=({before[0]:+.4f}, {before[1]:.4f}, {before[2]:.4f})")

    # ── Pose tensor: identity quaternion = upright orientation ───────────────
    pose_t = torch.tensor([[_ix, _iy, 0.060, 1.0, 0.0, 0.0, 0.0]],
                          dtype=torch.float32, device=device)
    vel_t  = torch.zeros(1, 6, dtype=torch.float32, device=device)

    physx_ok = False
    try:
        obj.write_root_link_pose_to_sim(pose_t)
        obj.write_root_com_velocity_to_sim(vel_t)
        physx_ok = True
        print(f"[CUP] PhysX placement (4.x API) → Isaac=({_ix:+.4f}, {_iy:.4f}, 0.060)")
    except Exception as _pe:
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore", DeprecationWarning)
                state = obj.data.root_state_w.clone()
                state[:, :3]  = torch.tensor([[_ix, _iy, 0.060]], dtype=torch.float32, device=device)
                state[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
                state[:, 7:]  = 0.0
                obj.write_root_state_to_sim(state)
            physx_ok = True
            print(f"[CUP] PhysX placement (legacy API) → Isaac=({_ix:+.4f}, {_iy:.4f}, 0.060)")
        except Exception as _pe2:
            print(f"[CUP] All PhysX APIs failed: {_pe2}")

    # ── USD authored translate so RTX renderer agrees ────────────────────────
    stage = omni.usd.get_context().get_stage()
    try:
        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            try:
                leaf_path = obj.root_view.prim_paths[0]
            except AttributeError:
                leaf_path = obj.root_physx_view.prim_paths[0]
        prim_path = "/".join(leaf_path.split("/")[:5])
    except Exception:
        prim_path = obj.cfg.prim_path.replace("{ENV_IDX}", "0")
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        xf   = UsdGeom.Xformable(prim)
        t_op = next(
            (op for op in xf.GetOrderedXformOps()
             if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
            None,
        )
        if t_op is None:
            t_op = xf.AddTranslateOp()
        t_op.Set(Gf.Vec3d(*xyz))
        print(f"[CUP] USD translate → {prim_path}  Isaac=({_ix:+.4f}, {_iy:.4f}, 0.060)")
    else:
        print(f"[CUP] WARNING: prim not found at {prim_path}")

    # ── Read AFTER to confirm ─────────────────────────────────────────────────
    with _w.catch_warnings():
        _w.simplefilter("ignore", DeprecationWarning)
        try:
            after = obj.data.root_link_pose_w[0, :3].tolist()
        except AttributeError:
            after = obj.data.root_state_w[0, :3].tolist()
    print(f"[CUP] Cup after placement:  Isaac=({after[0]:+.4f}, {after[1]:.4f}, {after[2]:.4f})")
    exp = (_ix, _iy, 0.060)
    ok  = all(abs(after[i] - exp[i]) < 0.005 for i in range(3))
    print(f"[CUP] Position accepted by PhysX: {'YES ✓' if ok else 'NO — cup did NOT move, check API'}")


def _add_mat_to_stage(stage, mat_usd_path):
    """Load mat.usda into the Isaac stage with the correct world-frame transform.

    The mat USDA defines the mesh in LOCAL coordinates:
        X: -0.40 .. +0.40   (80 cm wide)
        Y: +0.210 .. -0.490  (70 cm deep; near/robot side = +Y in local)
        Z: ±0.003            (6 mm thick)

    Isaac Lab places it in WORLD frame via Rz180 + translate(-0.035, +0.210, 0):
        world_X = -local_X - 0.035   →  world X: -0.435 .. +0.365
        world_Y = -local_Y + 0.210   →  world Y:  0.000 .. +0.700  (near = Y=0)

    The graphpaper texture (2400×2100 px @ 30 px/cm) has 1 cm cells → each
    major visible square is 10 cells = 10 cm in Isaac world space.

    Called once after the stage is obtained.  Idempotent: removes any existing
    /World/DebugGraphpaperMat prim before re-adding.
    """
    from pxr import UsdGeom, Gf, Sdf

    prim_path = Sdf.Path("/World/DebugGraphpaperMat")
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)

    # Define an Xform and reference the mat USDA inside it so we can apply
    # our world-frame transform independently of whatever the USDA has.
    xform = UsdGeom.Xform.Define(stage, prim_path)

    # 1. Translate into Isaac world frame
    xform.AddTranslateOp().Set(Gf.Vec3d(-0.035, 0.210, 0.0))

    # 2. Rz180: quaternion (w=0, x=0, y=0, z=1) — negates both X and Y
    xform.AddOrientOp().Set(Gf.Quatd(0.0, Gf.Vec3d(0.0, 0.0, 1.0)))

    # Reference the USDA at its root prim "/World/Mat"
    xform.GetPrim().GetReferences().AddReference(mat_usd_path, "/World/Mat")

    print(f"[MAT] Loaded {mat_usd_path} → {prim_path}  "
          f"(Rz180 + translate(-0.035, +0.210, 0.0))")
    print(f"[MAT] Expected world AABB:  X=-0.435..+0.365  Y=0.000..+0.700")


def _draw_isaac_grid_lines(stage):
    """Draw 10 cm grid lines as chains of UsdGeom.Sphere dots in Isaac world-frame.

    ATTEMPT 8 — DOTTED SPHERE LINES:
      After 7 failed attempts (BasisCurves, Mesh×4, Capsule×2, Sphere+ScaleOp),
      the ScaleOp approach was also invisible — Isaac Sim's analytic sphere renderer
      likely ignores xform scale transforms and renders the sphere at its base radius.
      At R=5mm the sphere is ~2px at DLSS 320×240 input = invisible.

      NEW APPROACH: Place MANY individual spheres along each grid line.
        • R = 0.015 m (15 mm) — large enough at DLSS 320×240 (≈6px diameter)
        • DOT_STEP = 0.05 m (5 cm) — close enough to read as dotted grid lines
        • Each sphere uses the CONFIRMED-WORKING DebugGridMarker pattern:
            UsdGeom.Sphere.Define → Set radius → Set displayColor → AddTranslateOp
          Placed directly under /World (no Scope parent).

    Mat AABB: X=-0.435..+0.365, Y=0.000..+0.700, step=0.10 m.
    Colours:
      Gold  (1.0, 0.65, 0.0) — constant-Y lines (run in X direction)
      Cyan  (0.0, 0.85, 1.0) — constant-X lines (run in Y direction)
      White (1.0, 1.0,  1.0) — Y=0 / X=0 axis lines
    Z = 0.015 m (15 mm above mat top at Z≈0.003) — clear of mat surface.
    Idempotent: removes /World/DebugGridLine_N prims before re-adding.
    """
    from pxr import UsdGeom, Sdf, Vt, Gf

    X_MIN, X_MAX = -0.435, 0.365
    Y_MIN, Y_MAX =  0.000, 0.700
    GRID_STEP  = 0.10    # 10 cm between grid lines
    DOT_STEP   = 0.05    # 5 cm between sphere dots within each line
    Z          = 0.050   # 50 mm — SAME height as confirmed-working magenta corner spheres
    R          = 0.0025  # 2.5 mm radius → 0.5 cm diameter (Attempt 10 — user-requested size)

    # Clean up all previous DebugGridLine_N prims
    for i in range(600):
        old = Sdf.Path(f"/World/DebugGridLine_{i}")
        if stage.GetPrimAtPath(old).IsValid():
            stage.RemovePrim(old)
        else:
            break

    def _dot(idx, cx, cy, cz, rgb):
        """Place one sphere dot — mirrors the confirmed-working DebugGridMarker pattern exactly."""
        p = Sdf.Path(f"/World/DebugGridLine_{idx}")
        if stage.GetPrimAtPath(p).IsValid():
            stage.RemovePrim(p)
        sp = UsdGeom.Sphere.Define(stage, p)
        sp.GetRadiusAttr().Set(R)
        sp.GetDisplayColorAttr().Set(Vt.Vec3fArray([rgb]))
        UsdGeom.Xformable(sp).AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))

    dot_idx = 0

    # Constant-Y lines: run in X direction, placed at Y = Y_MIN..Y_MAX step GRID_STEP
    # Bright RED — high contrast against the white/tan mat (gold/white camouflaged before)
    y = Y_MIN
    while y <= Y_MAX + 1e-6:
        yr  = round(y, 6)
        rgb = (1.0, 0.0, 1.0) if abs(yr) < 1e-6 else (1.0, 0.0, 0.0)
        lbl = "(magenta/Y=0)" if abs(yr) < 1e-6 else "(red)"
        count = 0
        x = X_MIN
        while x <= X_MAX + 1e-6:
            _dot(dot_idx, round(x, 6), yr, Z, rgb)
            dot_idx += 1
            count += 1
            x = round(x + DOT_STEP, 6)
        y_cm = int(round(yr * 100))
        print(f"[GRIDLINES] Y={y_cm:+d}cm {lbl}: {count} dots → DebugGridLine_{dot_idx-count}..{dot_idx-1}")
        y = round(y + GRID_STEP, 6)

    # Constant-X lines: run in Y direction, placed at X = X_MIN..X_MAX step GRID_STEP
    # Bright LIME-GREEN — high contrast against the white/tan mat (cyan camouflaged before)
    x = X_MIN
    while x <= X_MAX + 1e-6:
        xr  = round(x, 6)
        rgb = (1.0, 0.0, 1.0) if abs(xr) < 1e-6 else (0.0, 1.0, 0.0)
        lbl = "(magenta/X=0)" if abs(xr) < 1e-6 else "(lime)"
        count = 0
        y = Y_MIN
        while y <= Y_MAX + 1e-6:
            _dot(dot_idx, xr, round(y, 6), Z, rgb)
            dot_idx += 1
            count += 1
            y = round(y + DOT_STEP, 6)
        x_cm = int(round(xr * 100))
        print(f"[GRIDLINES] X={x_cm:+d}cm {lbl}: {count} dots → DebugGridLine_{dot_idx-count}..{dot_idx-1}")
        x = round(x + GRID_STEP, 6)

    print(f"[GRIDLINES] Total: {dot_idx} sphere dots at /World/DebugGridLine_0..{dot_idx-1}"
          f"  [Attempt 10: Z=50mm, R=2.5mm (0.5cm diam), red/lime colors]")





def project_isaac_3d_to_sim_pixel(xyz_isaac, env):
    """Project an Isaac-world point to SIM camera pixel via the 3D camera model.

    NOTE: this projection has a known ~188px Y error.  A planar homography
    approach was previously attempted (calibrated from magenta corner spheres)
    but was removed because the spheres were never present at runtime.

    Returns (col, row) ints, or None on failure.
    """
    import numpy as np

    try:
        from pxr import UsdGeom, Usd
        import omni.usd
        cam   = env.unwrapped.scene["camera_overhead"]
        K     = cam.data.intrinsic_matrices[0].cpu().numpy()
        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath("/World/envs/env_0/CameraOverhead")
        if not cam_prim.IsValid():
            return None
        tc    = Usd.TimeCode.Default()
        tf_gf = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(tc)
        M     = np.array([[tf_gf[r][c] for c in range(4)] for r in range(4)])
        pos_w = M[3, :3]
        X_ros = M[0, :3]
        Y_ros = -M[1, :3]
        Z_ros = -M[2, :3]
        R_w2c = np.stack([X_ros, Y_ros, Z_ros], axis=0)
        P_w   = np.array(xyz_isaac, dtype=np.float64)
        P_c   = R_w2c @ (P_w - pos_w)
        if P_c[2] <= 1e-4:
            return None
        px  = K @ (P_c / P_c[2])
        return int(round(float(px[0]))), int(round(float(px[1])))
    except Exception as _e:
        import traceback; traceback.print_exc()
        return None


def annotate_and_save_sim(rgb_frame, wx, wy, path_out, label_tag="", isaac_pos=None,
                          placement_pos=None,
                          env=None, cup_pos=None,
                          cup_world=None):
    """Annotate the SIM image with overlays (grid dots, origin marker, blob), save.

    isaac_pos:     [ix, iy, iz] — Isaac world coords of the cylinder at capture time
                   (read from root_state_w).  Shows where the cylinder actually IS.
    placement_pos: [ix, iy, iz] — Isaac coords of the homography-derived placement
                   (before origin move).  Orange dot.
    cup_pos:       [ix, iy, iz] — Isaac coords where the yellow cup was placed
                   (commanded position, used to project the dot onto the SIM image).
    cup_world:     (wx, wy) — real-world coords detected in the REAL image.
                   Shown as "cup (wx,wy)" in the tooltip, comparable to the cylinder's
                   "cyl (wx,wy)" line.  Isaac actual position comes from root_state_w.
    """
    import cv2
    img = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    h_img, w_img = img.shape[:2]

    # ── Detect the green cylinder blob ───────────────────────────────────────
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _CYL_HSV_LO, _CYL_HSV_HI)
    ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.32   # small
    thick = 1
    WHITE = (255, 255, 255)
    BLACK = (0,   0,   0)

    def _label(im, lines, anchor_x, anchor_y, side="right"):
        """Draw a white-background black-text box to the right or left of an anchor."""
        sizes = [cv2.getTextSize(l, font, scale, thick) for l in lines]
        box_w = max(s[0][0] for s in sizes) + 6
        lh    = max(s[0][1] for s in sizes)
        box_h = len(lines) * (lh + 3) + 3
        if side == "right":
            bx = anchor_x + 14
            if bx + box_w > im.shape[1]:
                bx = anchor_x - box_w - 14
        else:
            bx = anchor_x - box_w - 14
            if bx < 0:
                bx = anchor_x + 14
        by = max(anchor_y - box_h // 2, 2)
        cv2.rectangle(im, (bx - 1, by - 1), (bx + box_w, by + box_h), WHITE, -1)
        cv2.rectangle(im, (bx - 1, by - 1), (bx + box_w, by + box_h), (180,180,180), 1)
        for i, line in enumerate(lines):
            cv2.putText(im, line, (bx + 3, by + (i + 1) * (lh + 3)),
                        font, scale, BLACK, thick, cv2.LINE_AA)

    blob_px, blob_py = None, None
    if ctrs:
        c  = max(ctrs, key=cv2.contourArea)
        Mc = cv2.moments(c)
        if Mc["m00"] > 0:
            blob_px = int(Mc["m10"] / Mc["m00"])
            blob_py = int(Mc["m01"] / Mc["m00"])
            if SHOW_BLOB_DOT:
                cv2.circle(img, (blob_px, blob_py), 10, (255, 0, 0), 2)
                cv2.circle(img, (blob_px, blob_py),  3, (255, 0, 0), -1)
                _label(img, [f"blob  px=({blob_px},{blob_py})"],
                       blob_px, blob_py, side="right")
            print(f"[SIM{label_tag}] Green blob centroid: ({blob_px}, {blob_py})")

    # ── Project world → SIM pixel via 3D camera model (green crosshair) ─────
    # Isaac coords: ix = -world_X, iy = world_Y, iz = 0.05 (cylinder height).
    # Uses project_isaac_3d_to_sim_pixel (confirmed 3D projection). env required.
    _proj_result = None
    if env is not None:
        try:
            if CROSSHAIR_OVERRIDE:
                _ch_ix = CROSSHAIR_OVERRIDE_ISAAC_X_M
                _ch_iy = CROSSHAIR_OVERRIDE_ISAAC_Y_M
                print(f"[SIM{label_tag}] Crosshair OVERRIDE → Isaac ({_ch_ix:+.4f}, {_ch_iy:.4f})  "
                      f"[tune CROSSHAIR_OVERRIDE_ISAAC_X_M / _Y_M to align with blob]")
            else:
                _ch_ix, _ch_iy = real_world_to_isaac(wx, wy)
                print(f"[SIM{label_tag}] Crosshair AUTO → Isaac ({_ch_ix:+.4f}, {_ch_iy:.4f})  "
                      f"[offsets: X={REAL_TO_ISAAC_X_OFFSET_M:+.3f}  Y={REAL_TO_ISAAC_Y_OFFSET_M:+.3f}]")
            _proj_result = project_isaac_3d_to_sim_pixel([_ch_ix, _ch_iy, 0.05], env)
        except Exception as _pe3:
            print(f"[SIM{label_tag}] 3D projection failed: {_pe3}")
    if _proj_result is not None:
        spx, spy = _proj_result
        print(f"[SIM{label_tag}] Projected world ({wx:+.4f},{wy:.4f}) → SIM pixel ({spx},{spy})")
        if SHOW_TARGET_CROSSHAIR:
            if 0 <= spx < w_img and 0 <= spy < h_img:
                cv2.circle(img, (spx, spy), 22, (0, 255, 0), 3)
                cv2.circle(img, (spx, spy),  4, (0, 255, 0), -1)
                cv2.line(img, (spx - 32, spy), (spx + 32, spy), (0, 255, 0), 2)
                cv2.line(img, (spx, spy - 32), (spx, spy + 32), (0, 255, 0), 2)
                _label(img,
                       [f"target ({wx:+.3f},{wy:.3f})",
                        f"px=({spx},{spy})"],
                       spx, spy, side="right")
            else:
                print(f"[SIM{label_tag}] Crosshair pixel ({spx},{spy}) outside frame")
        if blob_px is not None:
            err = ((blob_px - spx) ** 2 + (blob_py - spy) ** 2) ** 0.5
            print(f"[SIM{label_tag}] Blob→crosshair pixel error: {err:.1f} px  "
                  f"({'OK ✓' if err < 20 else 'MISMATCH ✗'})")
    else:
        print(f"[SIM{label_tag}] Crosshair projection unavailable (env required)")

    # ── Red dot: Isaac cylinder position projected to SIM pixel ──────────────
    # Uses project_isaac_3d_to_sim_pixel (3D camera model projection).
    # This dot shows where Isaac says the cylinder is, regardless of HSV.
    # If it matches the crosshair: projection is accurate.
    # If it matches the blue blob: physics and rendering agree with Isaac.
    if isaac_pos is not None:
        try:
            _rproj = project_isaac_3d_to_sim_pixel(
                [float(isaac_pos[0]), float(isaac_pos[1]), float(isaac_pos[2])], env)
            if _rproj is not None:
                rpx, rpy = _rproj
                print(f"[SIM{label_tag}] Isaac  ({isaac_pos[0]:+.4f},{isaac_pos[1]:.4f}) → SIM px ({rpx},{rpy})")
                if SHOW_ISAAC_DOT and 0 <= rpx < w_img and 0 <= rpy < h_img:
                    cv2.circle(img, (rpx, rpy),  7, (0, 0, 200), -1)   # red filled dot
                    cv2.circle(img, (rpx, rpy),  8, (0,  0,  0),  1)   # thin black ring
                    _label(img,
                           [f"Isaac ({isaac_pos[0]:+.3f},{isaac_pos[1]:.3f})",
                            f"px=({rpx},{rpy})"],
                           rpx, rpy, side="right")
            else:
                print(f"[SIM{label_tag}] Isaac dot: project_isaac_3d_to_sim_pixel returned None")
        except Exception as _re:
            print(f"[SIM{label_tag}] Red dot projection failed: {_re}")

    # ── Orange dot: homography-derived placement position (before origin move) ─
    # Distinct from isaac_pos (cylinder's current position).  Shows where the
    # homography said to place the cylinder; if the cylinder was subsequently
    # moved to origin, both dots appear in the image for easy comparison.
    if placement_pos is not None:
        try:
            _pz = float(placement_pos[2]) if len(placement_pos) > 2 else 0.0
            _pproj = project_isaac_3d_to_sim_pixel(
                [float(placement_pos[0]), float(placement_pos[1]), _pz], env)
            if _pproj is not None:
                ppx, ppy = _pproj
                print(f"[SIM{label_tag}] Place  ({placement_pos[0]:+.4f},{placement_pos[1]:.4f}) → SIM px ({ppx},{ppy})")
                if SHOW_PLACEMENT_DOT and 0 <= ppx < w_img and 0 <= ppy < h_img:
                    cv2.circle(img, (ppx, ppy),  7, (0, 100, 255), -1)   # orange filled dot
                    cv2.circle(img, (ppx, ppy),  8, (0,   0,   0),  1)   # thin black ring
                    _label(img,
                           [f"place ({placement_pos[0]:+.3f},{placement_pos[1]:.3f})",
                            f"px=({ppx},{ppy})"],
                           ppx + 46, ppy, side="right")  # +46px ≈ 5 cm right
            else:
                print(f"[SIM{label_tag}] Place dot: project_isaac_3d_to_sim_pixel returned None")
        except Exception as _pe2:
            print(f"[SIM{label_tag}] Orange placement dot projection failed: {_pe2}")

    # ── Yellow circle: cup position ───────────────────────────────────────────
    # Mirrors the cylinder annotation:
    #   "cup"   line = real-world coords detected in the REAL image (cup_world)
    #   "isaac" line = what Isaac's physics reports via root_state_w (cup_isaac_pos)
    # The circle is projected from cup_isaac_pos so it tracks where PhysX put the cup.
    if cup_pos is not None:
        try:
            import warnings as _cw
            # ── Read actual Isaac position from PhysX (matches cylinder's isaac_pos) ──
            cup_isaac_pos = None
            if env is not None:
                try:
                    with _cw.catch_warnings():
                        _cw.simplefilter("ignore", DeprecationWarning)
                        try:
                            cup_isaac_pos = env.unwrapped.scene["target"].data.root_link_pose_w[0, :3].tolist()
                        except AttributeError:
                            cup_isaac_pos = env.unwrapped.scene["target"].data.root_state_w[0, :3].tolist()
                except Exception as _cr:
                    print(f"[SIM{label_tag}] Cup root_state_w read failed: {_cr}")

            # Project actual Isaac position → SIM pixel for circle placement
            _cup_proj_pos = cup_isaac_pos if cup_isaac_pos is not None else cup_pos
            _cz = float(_cup_proj_pos[2]) if len(_cup_proj_pos) > 2 else 0.060
            _cproj = project_isaac_3d_to_sim_pixel(
                [float(_cup_proj_pos[0]), float(_cup_proj_pos[1]), _cz], env)
            if _cproj is not None:
                cpx, cpy = _cproj
                print(f"[SIM{label_tag}] Cup    ({_cup_proj_pos[0]:+.4f},{_cup_proj_pos[1]:.4f}) → SIM px ({cpx},{cpy})")
                if 0 <= cpx < w_img and 0 <= cpy < h_img:
                    cv2.circle(img, (cpx, cpy), 10, (0, 215, 255), 2)   # BGR yellow ring
                    cv2.circle(img, (cpx, cpy),  3, (0, 215, 255), -1)  # yellow centre dot
                    # "cup" = real-world from REAL image; "isaac" = PhysX root_state_w
                    if cup_world is not None and cup_isaac_pos is not None:
                        _label(img,
                               [f"cup ({cup_world[0]:+.3f},{cup_world[1]:.3f})",
                                f"isaac ({cup_isaac_pos[0]:+.3f},{cup_isaac_pos[1]:.3f})"],
                               cpx, cpy, side="left")
                    elif cup_isaac_pos is not None:
                        _label(img,
                               [f"isaac ({cup_isaac_pos[0]:+.3f},{cup_isaac_pos[1]:.3f})"],
                               cpx, cpy, side="left")
            else:
                print(f"[SIM{label_tag}] Cup dot: project_isaac_3d_to_sim_pixel returned None")
        except Exception as _cpe:
            print(f"[SIM{label_tag}] Cup dot projection failed: {_cpe}")

    # ── White diamond: FIXED Isaac origin (0,0,0) — hardcoded, never drifts ───
    # Uses the same accurate 3D camera projection as the crosshair.
    # Ground-truth pixel for Isaac (0,0,0) so the user can visually confirm
    # where SIM world-origin is while tuning the crosshair.
    # Drawn as a white ◇ + "Isaac (0,0)" label so it never moves.
    if SHOW_ORIGIN_MARKER and env is not None:
        try:
            _origin_proj = project_isaac_3d_to_sim_pixel([0.0, 0.0, 0.0], env)
            if _origin_proj is not None:
                opx, opy = _origin_proj
                print(f"[SIM{label_tag}] Origin Isaac(0,0,0) → SIM px ({opx},{opy})")
                if 0 <= opx < w_img and 0 <= opy < h_img:
                    # White filled dot at exact origin pixel
                    cv2.circle(img, (opx, opy), 5, (255, 255, 255), -1)   # white fill
                    cv2.circle(img, (opx, opy), 6, (0,   0,   0),    1)   # thin black ring
                    # White diamond outline — large so it can't be missed
                    d = 16
                    diamond = np.array([
                        [opx,     opy - d],
                        [opx + d, opy    ],
                        [opx,     opy + d],
                        [opx - d, opy    ],
                    ], dtype=np.int32)
                    cv2.polylines(img, [diamond], isClosed=True, color=(255, 255, 255), thickness=2)
                    _label(img,
                           ["Isaac (0,0)",
                            f"px=({opx},{opy})"],
                           opx, opy, side="right")
            else:
                print(f"[SIM{label_tag}] Origin marker: 3D projection returned None")
        except Exception as _oe:
            print(f"[SIM{label_tag}] Origin marker projection failed: {_oe}")

    # ── Isaac coordinate grid ────────────────────────────────────────────────
    # Projects a regular grid of Isaac (X, Y) positions to SIM pixels and
    # draws labelled grid lines so you can read off Isaac coordinates visually.
    # X lines are cyan, Y lines are yellow, origin axes are white.
    if SHOW_ISAAC_GRID and env is not None:
        import cv2 as _cv2g

        _xs_cm = list(range(ISAAC_GRID_X_MIN_CM, ISAAC_GRID_X_MAX_CM + 1, ISAAC_GRID_STEP_CM))
        _ys_cm = list(range(ISAAC_GRID_Y_MIN_CM, ISAAC_GRID_Y_MAX_CM + 1, ISAAC_GRID_STEP_CM))

        # Build lookup: (ix_cm, iy_cm) → (px, py) or None
        _grid_px = {}
        for _gx in _xs_cm:
            for _gy in _ys_cm:
                _pt = project_isaac_3d_to_sim_pixel(
                    [_gx / 100.0, _gy / 100.0, 0.0], env
                )
                if _pt is not None:
                    _cx, _cy = _pt
                    if 0 <= _cx < w_img and 0 <= _cy < h_img:
                        _grid_px[(_gx, _gy)] = (_cx, _cy)
                    else:
                        _grid_px[(_gx, _gy)] = None
                else:
                    _grid_px[(_gx, _gy)] = None

        # Draw lines along constant Y (horizontal sweeps across X)
        for _gy in _ys_cm:
            _is_y_axis = (_gy == 0)
            _col = (255, 255, 255) if _is_y_axis else (0, 215, 255)  # white or gold
            _thick_line = 2 if _is_y_axis else 1
            _prev = None
            for _gx in _xs_cm:
                _cur = _grid_px.get((_gx, _gy))
                if _prev is not None and _cur is not None:
                    _cv2g.line(img, _prev, _cur, _col, _thick_line, _cv2g.LINE_AA)
                _prev = _cur

        # Draw lines along constant X (vertical sweeps across Y)
        for _gx in _xs_cm:
            _is_x_axis = (_gx == 0)
            _col = (255, 255, 255) if _is_x_axis else (255, 180, 0)  # white or cyan-blue
            _thick_line = 2 if _is_x_axis else 1
            _prev = None
            for _gy in _ys_cm:
                _cur = _grid_px.get((_gx, _gy))
                if _prev is not None and _cur is not None:
                    _cv2g.line(img, _prev, _cur, _col, _thick_line, _cv2g.LINE_AA)
                _prev = _cur

        # Draw dots + labels at each intersection
        for _gx in _xs_cm:
            for _gy in _ys_cm:
                _pt2 = _grid_px.get((_gx, _gy))
                if _pt2 is None:
                    continue
                _px2, _py2 = _pt2
                _is_origin = (_gx == 0 and _gy == 0)
                _is_axis   = (_gx == 0 or _gy == 0)
                _dot_col   = (255, 255, 255) if _is_origin else (200, 200, 200)
                _dot_r     = 4 if _is_origin else 2
                _cv2g.circle(img, (_px2, _py2), _dot_r, _dot_col, -1)

                # Label every intersection with Isaac coords
                _lbl = f"({_gx:+d},{_gy:d})"
                _lscale = 0.28
                (_lw, _lh), _ = _cv2g.getTextSize(_lbl, font, _lscale, 1)
                # Place label to avoid image edge
                _lx = _px2 + 4 if _px2 + 4 + _lw < w_img else _px2 - _lw - 4
                _ly = _py2 - 4 if _py2 - 4 > _lh else _py2 + _lh + 4
                _cv2g.putText(img, _lbl, (_lx, _ly), font, _lscale,
                              (0, 0, 0), 1, _cv2g.LINE_AA)

        print(f"[SIM{label_tag}] Isaac grid drawn: "
              f"X={ISAAC_GRID_X_MIN_CM}..{ISAAC_GRID_X_MAX_CM}cm  "
              f"Y={ISAAC_GRID_Y_MIN_CM}..{ISAAC_GRID_Y_MAX_CM}cm  "
              f"step={ISAAC_GRID_STEP_CM}cm")

    # ── SIM world-coordinate grid ─────────────────────────────────────────────
    # Projects the same (world_X, world_Y) grid used on the REAL image onto the
    # SIM image.  Each grid point is converted to Isaac coords via
    # real_world_to_isaac, then projected to a SIM pixel via
    # project_isaac_3d_to_sim_pixel.  Because the SIM camera is perspective,
    # each "line" is sampled at SIM_GRID_SAMPLES points and drawn as a polyline
    # so any curvature is captured.
    # Colour: constant-X = cyan (255,200,0), constant-Y = gold (0,200,255),
    #         zero axes = white.  Labels at the image edge.
    if SHOW_SIM_WORLD_GRID and env is not None:

        def _gvals_sim(lo, hi, step):
            vals, v = [], lo
            while v <= hi + step * 0.01:
                vals.append(round(v, 6))
                v += step
            return vals

        _sxs_m = _gvals_sim(SIM_GRID_X_MIN_M, SIM_GRID_X_MAX_M, SIM_GRID_STEP_M)
        _sys_m = _gvals_sim(SIM_GRID_Y_MIN_M, SIM_GRID_Y_MAX_M, SIM_GRID_STEP_M)

        def _w2sim(ix_m, iy_m):
            """Isaac Lab world metres → SIM pixel via 3D projection (no conversion)."""
            return project_isaac_3d_to_sim_pixel([ix_m, iy_m, 0.0], env)

        def _polyline_sim(img, pts, color, thickness):
            """Draw a polyline through the in-frame subset of pts."""
            _valid = [(p[0], p[1]) for p in pts
                      if p is not None and 0 <= p[0] < w_img and 0 <= p[1] < h_img]
            if len(_valid) < 2:
                return
            import numpy as _np3
            cv2.polylines(img,
                          [_np3.array(_valid, dtype=_np3.int32)],
                          isClosed=False, color=color,
                          thickness=thickness, lineType=cv2.LINE_AA)

        # Build full grid pixel lookup for dot rendering
        _sgrid_pts = {}
        for _swx in _sxs_m:
            for _swy in _sys_m:
                _sgrid_pts[(_swx, _swy)] = _w2sim(_swx, _swy)

        # --- Constant-X lines (cyan; white for X=0)
        for _swx in _sxs_m:
            _is_x0 = abs(_swx) < 1e-6
            _scol  = (255, 255, 255) if _is_x0 else (255, 200, 0)
            _sthk  = 2 if _is_x0 else 1
            _ys_sample = np.linspace(SIM_GRID_Y_MIN_M, SIM_GRID_Y_MAX_M, SIM_GRID_SAMPLES)
            _pts = [_w2sim(_swx, _sy) for _sy in _ys_sample]
            _polyline_sim(img, _pts, _scol, _sthk)
            # Label near the top of the line (first in-frame point)
            _lbl_x = f"X={_swx*100:+.0f}cm"
            _lw, _lh = cv2.getTextSize(_lbl_x, font, 0.32, 1)[0]
            _top_pt = next((p for p in _pts
                            if p is not None and 0 <= p[0] < w_img and 0 <= p[1] < h_img), None)
            if _top_pt is not None:
                _lx = max(2, min(_top_pt[0] - _lw // 2, w_img - _lw - 2))
                _ly = max(_lh + 3, _top_pt[1] + _lh + 2)
                cv2.putText(img, _lbl_x, (_lx, _ly), font, 0.32, _scol, 1, cv2.LINE_AA)

        # --- Constant-Y lines (gold; white for Y=0)
        for _swy in _sys_m:
            _is_y0 = abs(_swy) < 1e-6
            _scol  = (255, 255, 255) if _is_y0 else (0, 200, 255)
            _sthk  = 2 if _is_y0 else 1
            _xs_sample = np.linspace(SIM_GRID_X_MIN_M, SIM_GRID_X_MAX_M, SIM_GRID_SAMPLES)
            _pts = [_w2sim(_sx, _swy) for _sx in _xs_sample]
            _polyline_sim(img, _pts, _scol, _sthk)
            # Label on the right edge (last in-frame point)
            _lbl_y = f"Y={_swy*100:.0f}cm"
            _lwy, _lhy = cv2.getTextSize(_lbl_y, font, 0.32, 1)[0]
            _right_pt = next((p for p in reversed(_pts)
                              if p is not None and 0 <= p[0] < w_img and 0 <= p[1] < h_img), None)
            if _right_pt is not None:
                _ey = max(_lhy + 2, min(_right_pt[1] + _lhy // 2, h_img - 2))
                cv2.putText(img, _lbl_y, (w_img - _lwy - 4, _ey), font, 0.32, _scol, 1, cv2.LINE_AA)

        # --- Dots at visible intersections
        for _swx in _sxs_m:
            for _swy in _sys_m:
                _spt = _sgrid_pts.get((_swx, _swy))
                if _spt is None:
                    continue
                _spx2, _spy2 = _spt
                if not (0 <= _spx2 < w_img and 0 <= _spy2 < h_img):
                    continue
                _is_orig = abs(_swx) < 1e-6 and abs(_swy) < 1e-6
                _dc = (255, 255, 255) if _is_orig else (180, 180, 180)
                cv2.circle(img, (_spx2, _spy2), 3 if _is_orig else 2, _dc, -1)

        # --- Snap ring: highlight grid intersection nearest the detected cylinder
        # Convert detected real-world coords to Isaac frame for snapping.
        _scyl_ix, _scyl_iy = real_world_to_isaac(wx, wy)
        _ssnap_x = round(_scyl_ix / SIM_GRID_STEP_M) * SIM_GRID_STEP_M
        _ssnap_y = round(_scyl_iy / SIM_GRID_STEP_M) * SIM_GRID_STEP_M
        _ssnap_pt = _sgrid_pts.get((_ssnap_x, _ssnap_y)) or _w2sim(_ssnap_x, _ssnap_y)
        if _ssnap_pt is not None:
            _sspx, _sspy = _ssnap_pt
            if 0 <= _sspx < w_img and 0 <= _sspy < h_img:
                cv2.circle(img, (_sspx, _sspy), 7, (0, 255, 255), 2)
                _slbl = f"({_ssnap_x*100:+.0f},{_ssnap_y*100:.0f})cm"
                _slw, _slh = cv2.getTextSize(_slbl, font, 0.38, 1)[0]
                _slx = _sspx + 10 if _sspx + 10 + _slw < w_img else _sspx - _slw - 10
                cv2.putText(img, _slbl, (_slx, _sspy + _slh // 2),
                            font, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

        print(f"[SIM{label_tag}] World-coord grid drawn: "
              f"X={SIM_GRID_X_MIN_M*100:.0f}..{SIM_GRID_X_MAX_M*100:.0f}cm  "
              f"Y={SIM_GRID_Y_MIN_M*100:.0f}..{SIM_GRID_Y_MAX_M*100:.0f}cm  "
              f"step={SIM_GRID_STEP_M*100:.0f}cm")

    # ── Grid dot labels (shown when DRAW_ISAAC_GRID_LINES is on) ─────────────
    # Projects each 10 cm × 10 cm Isaac grid intersection to SIM pixels and
    # draws a compact "(X,Y)" cm label so the coordinate grid is readable.
    # Labels are drawn with a thin black stroke outline for legibility over any
    # background.  Same extents and step as _draw_isaac_grid_lines.
    if DRAW_ISAAC_GRID_LINES and env is not None:
        import numpy as _gnp
        _gl_xs = []
        _gx = SIM_GRID_X_MIN_M
        while _gx <= SIM_GRID_X_MAX_M + 1e-6:
            _gl_xs.append(round(_gx, 6))
            _gx = round(_gx + SIM_GRID_STEP_M, 6)
        _gl_ys = []
        _gy = SIM_GRID_Y_MIN_M
        while _gy <= SIM_GRID_Y_MAX_M + 1e-6:
            _gl_ys.append(round(_gy, 6))
            _gy = round(_gy + SIM_GRID_STEP_M, 6)

        _gl_font  = cv2.FONT_HERSHEY_SIMPLEX
        _gl_scale = 0.28
        _gl_thick = 1
        _gl_drawn = 0
        for _glx in _gl_xs:
            for _gly in _gl_ys:
                _glpt = project_isaac_3d_to_sim_pixel([_glx, _gly, 0.050], env)
                if _glpt is None:
                    continue
                _glpx, _glpy = _glpt
                if not (0 <= _glpx < w_img and 0 <= _glpy < h_img):
                    continue
                _xcm = int(round(_glx * 100))
                _ycm = int(round(_gly * 100))
                _txt = f"({_xcm:+d},{_ycm:d})"
                # Small filled dot at intersection
                cv2.circle(img, (_glpx, _glpy), 2, (255, 255, 0), -1)   # yellow dot
                # Text with thin black outline then white fill for readability
                _tw2, _th2 = cv2.getTextSize(_txt, _gl_font, _gl_scale, _gl_thick)[0]
                _tx = _glpx + 4
                if _tx + _tw2 > w_img:
                    _tx = _glpx - _tw2 - 4
                _ty = _glpy - 3
                if _ty - _th2 < 0:
                    _ty = _glpy + _th2 + 3
                cv2.putText(img, _txt, (_tx, _ty), _gl_font, _gl_scale,
                            (0, 0, 0), _gl_thick + 1, cv2.LINE_AA)   # black stroke
                cv2.putText(img, _txt, (_tx, _ty), _gl_font, _gl_scale,
                            (255, 255, 255), _gl_thick, cv2.LINE_AA)  # white fill
                _gl_drawn += 1
        print(f"[SIM{label_tag}] Grid dot labels: {_gl_drawn} intersections labelled")


    # ── Cylinder coordinate label ─────────────────────────────────────────────
    # Anchor to the Isaac 3D projection (wx,wy → Isaac → SIM pixel) so the label
    # tracks the actual placed cylinder coords rather than the HSV blob centroid,
    # which can misfire onto the robot arm.  Falls back to blob centroid if the
    # projection is unavailable.
    _cyl_anchor = _proj_result if _proj_result is not None else (
        (blob_px, blob_py) if blob_px is not None else None
    )
    if _cyl_anchor is not None:
        _cax, _cay = _cyl_anchor
        _cyl_lines = [f"cyl ({wx:+.3f}, {wy:.3f})"]
        if isaac_pos is not None:
            _cyl_lines.append(f"isaac ({isaac_pos[0]:+.3f},{isaac_pos[1]:.3f})")
        _label(img, _cyl_lines, _cax, _cay, side="left")

    cv2.imwrite(path_out, img)
    print(f"[SIM{label_tag}] Saved → {path_out}")


def _capture_and_annotate(env, wx, wy, episode_idx, label, suffix,
                          placement_pos=None,
                          pass_env=True, cup_pos=None,
                          cup_world=None):
    """Read camera buffer, run blob detection, draw crosshair + Isaac dots, save.

    placement_pos:  Isaac coords of the homography-derived placement (orange dot).
    cup_pos:        Isaac coords where the yellow cup was placed (yellow circle).
    cup_world:      (wx, wy) real-world coords from REAL image detection — shown as
                    "cup (wx,wy)" in the tooltip alongside "isaac (ix,iy)" from root_state_w.

    Returns the raw RGB numpy frame (or None on failure).
    """
    import warnings as _w3
    try:
        rgb      = env.unwrapped.scene["camera_overhead"].data.output["rgb"]
        frame_np = rgb[0].cpu().numpy().astype(np.uint8)
        if frame_np.shape[-1] == 4:
            frame_np = frame_np[..., :3]
        # Read Isaac's current reported cylinder position (may drift after env.step).
        try:
            with _w3.catch_warnings():
                _w3.simplefilter("ignore", DeprecationWarning)
                isaac_pos = env.unwrapped.scene["cylinder"].data.root_state_w[0, :3].tolist()
        except Exception:
            isaac_pos = None
        path_out = os.path.join(OUT_DIR, suffix)
        annotate_and_save_sim(frame_np, wx, wy, path_out,
                              label_tag=f" [{label}]", isaac_pos=isaac_pos,
                              placement_pos=placement_pos,
                              env=env if pass_env else None,
                              cup_pos=cup_pos,
                              cup_world=cup_world)
        return frame_np
    except Exception as e:
        print(f"[SIM] Camera capture failed ({label}): {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    episode_idx = args_cli.episode
    real_root   = os.path.expanduser(
        f"~/.cache/huggingface/lerobot/{args_cli.real_repo_id}"
    )

    print("=" * 60)
    print(f"  replay_episodes.py   episode={episode_idx}  capture_frame={CAPTURE_FRAME}")
    print("=" * 60)

    # ── STEP 1-4: detect cylinder in real image ───────────────────────────────
    print("\n── REAL IMAGE ───────────────────────────────────────────────")
    cup_px, cup_py, cup_wx, cup_wy = None, None, None, None   # defaults if detection fails
    try:
        frame_bgr, px, py, wx, wy, cyl_bx, cyl_by, cyl_bw, cyl_bh, cup_px, cup_py, cup_wx, cup_wy, cup_bx, cup_by, cup_bw, cup_bh = detect_cylinder_in_real(real_root, episode_idx)
        save_annotated_real(frame_bgr, px, py, wx, wy, episode_idx,
                            cyl_bx=cyl_bx, cyl_by=cyl_by, cyl_bw=cyl_bw, cyl_bh=cyl_bh,
                            cup_px=cup_px, cup_py=cup_py, cup_wx=cup_wx, cup_wy=cup_wy,
                            cup_bx=cup_bx, cup_by=cup_by, cup_bw=cup_bw, cup_bh=cup_bh)
    except Exception as e:
        print(f"[ERROR] Real detection failed: {e}")
        print("[FALLBACK] Using hardcoded position: world=(-0.26, 0.31)")
        wx, wy = -0.26, 0.31

    print(f"\n[COORDS] world=({wx:+.4f}, {wy:.4f})")

    # ── STEP 5: launch SIM, load episode data ────────────────────────────────
    print("\n── SIM ──────────────────────────────────────────────────────")
    # ── Load episode frames — handle file-000.parquet or per-episode parquets ──
    _chunk_idx = episode_idx // 500
    _parquet_candidates = [
        f"{real_root}/data/chunk-000/file-000.parquet",
        f"{real_root}/data/chunk-{_chunk_idx:03d}/file-000.parquet",
        f"{real_root}/data/chunk-{_chunk_idx:03d}/episode_{episode_idx:06d}.parquet",
        f"{real_root}/data/chunk-000/episode_{episode_idx:06d}.parquet",
    ]
    df = None
    for _pq in _parquet_candidates:
        if os.path.exists(_pq):
            df = pd.read_parquet(_pq)
            print(f"[DATA] Loaded parquet: {_pq}  ({len(df)} rows)")
            break
    if df is None:
        import glob as _glob
        _all_pq = sorted(_glob.glob(f"{real_root}/data/**/*.parquet", recursive=True))
        if _all_pq:
            print(f"[DATA] Scanning {len(_all_pq)} parquet files for episode {episode_idx} …")
            _chunks = []
            for _pq in _all_pq:
                _tmp = pd.read_parquet(_pq)
                if episode_idx in _tmp["episode_index"].values:
                    _chunks.append(_tmp[_tmp["episode_index"] == episode_idx])
            if _chunks:
                df = pd.concat(_chunks, ignore_index=True)
                print(f"[DATA] Found episode {episode_idx} across {len(_chunks)} files")
            else:
                raise RuntimeError(
                    f"Episode {episode_idx} not found in any parquet under {real_root}/data/"
                )
        else:
            raise RuntimeError(f"No parquet files found under {real_root}/data/")
    ep_df  = df[df["episode_index"] == episode_idx].sort_values("frame_index")
    frames = list(ep_df.iterrows())
    print(f"[SIM] {len(frames)} frames for episode {episode_idx}")

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=None, use_fabric=True
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    cameras = {
        obj.replace("camera_", ""): {
            "height": getattr(env.unwrapped.scene.cfg, obj).height,
            "width":  getattr(env.unwrapped.scene.cfg, obj).width,
        }
        for obj in env.unwrapped.scene.keys()
        if obj.startswith("camera_")
    }
    robot_iface = LeRobotSO101Interface(
        device=env.unwrapped.device,
        port="dummy", id="replay", cameras=cameras, fps=30, kind="leader",
    )

    env.reset()
    for _ in range(10):
        simulation_app.update()

    import omni.usd
    stage = omni.usd.get_context().get_stage()

    configure_overhead_camera(stage)

    # Show or hide the yellow cup depending on whether it was detected in the real image.
    # When detected, the cup will be placed at its real-world position below.
    # When not detected, hide it so it doesn't appear at a wrong default position.
    try:
        from pxr import UsdGeom
        target_prim = stage.GetPrimAtPath("/World/envs/env_0/Target")
        if target_prim.IsValid():
            if cup_wx is not None:
                UsdGeom.Imageable(target_prim).MakeVisible()
                print("[SIM] Target/cup made visible — will be placed at detected position")
            else:
                UsdGeom.Imageable(target_prim).MakeInvisible()
                print("[SIM] Target/cup hidden — not detected in real image")
        else:
            print("[SIM] /World/envs/env_0/Target not found")
    except Exception as e:
        print(f"[SIM] Could not set cup visibility: {e}")

    simulation_app.update()

    # ── Apply graphpaper mat texture (orange boundary + grid baked into PNG) ──
    # This is the same texture replay_interactive applies; the orange outline
    # is part of the graphpaper_mat.png image, not drawn by code.
    try:
        _tex_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "graphpaper_mat.png")
        # Get a fresh stage reference immediately before the call — same pattern
        # as replay_interactive, which does `import omni.usd as _omni_usd_tex;
        # _tex_stage = _omni_usd_tex.get_context().get_stage()` right here.
        import omni.usd as _omni_usd_tex
        _tex_stage = _omni_usd_tex.get_context().get_stage()
        apply_graphpaper_texture(_tex_stage, texture_path=_tex_path)
        # ONE update() — exactly matching replay_interactive.
        # (Three was tried previously; it did not help and diverges from interactive.)
        simulation_app.update()
        print("[SIM] Graphpaper mat texture applied")
    except Exception as _tex_e:
        import traceback as _tb
        print(f"[SIM] apply_graphpaper_texture failed: {_tex_e}")
        _tb.print_exc()

    # ── Recolour robot arm to match real SO-101 appearance ───────────────────
    # Real graphpaper photo pre-rotated 180° (robot at bottom, correct left/right chirality).
    # Patches OmniPBR shaders directly under Robot/Looks/ — no Mesh traversal needed
    # because geometry is in USD references/payloads and never appears as Mesh prims.
    try:
        import omni.usd as _omni_usd_rc
        _rc_stage = _omni_usd_rc.get_context().get_stage()
        from apply_robot_recolor import apply_robot_recolor as _apply_recolor
        _n_recoloured = _apply_recolor(_rc_stage)
        if _n_recoloured:
            simulation_app.update()
    except Exception as _rc_e:
        import traceback as _tb_rc
        print(f"[SIM] Robot recolour failed (continuing): {_rc_e}")
        _tb_rc.print_exc()

    # ── Check cylinder kinematic status ──────────────────────────────────────
    # Kinematic = USD/Fabric path only (tensor API won't move it).
    # Dynamic   = tensor API should work; failure is a flush/sync issue.
    _cyl_check_path = "/World/envs/env_0/Cylinder"
    _cyl_check_prim = stage.GetPrimAtPath(_cyl_check_path)
    if _cyl_check_prim.IsValid():
        _kin_attr = _cyl_check_prim.GetAttribute("physics:kinematicEnabled")
        _kin_val  = _kin_attr.Get() if _kin_attr.IsValid() else None
        _kin_msg  = "KINEMATIC — use USD translate path only" if _kin_val else "DYNAMIC — tensor API should work"
        print(f"[PLACE] Cylinder physics:kinematicEnabled = {_kin_val}  ({_kin_msg})")
    else:
        print(f"[PLACE] WARNING: could not find {_cyl_check_path} for kinematic check")

    # ── Place cylinder at real-detected coords ────────────────────────────────
    # Direct pipeline: real pixel → _H_REAL → (wx, wy) → real_world_to_isaac
    # → Isaac (ix, iy) → write_root_link_pose_to_sim.
    # No intermediate SIM homography involved — _H_REAL is the only transform.
    place_cylinder_at_coords(env, wx, wy)
    _pi_x, _pi_y = real_world_to_isaac(wx, wy)
    _placement_isaac = (_pi_x, _pi_y, 0.059)

    # Cup placement Isaac coords — computed even before place_cup_at_coords() runs
    # so _cup_isaac is always defined for the annotation calls below.
    if cup_wx is not None and cup_wy is not None:
        _ci_x, _ci_y = real_world_to_isaac(cup_wx, cup_wy)
        _cup_isaac = (_ci_x, _ci_y, 0.060)
    else:
        _cup_isaac = None

    # ── Clear TGS contact warm-start before replay ────────────────────────────
    # The PhysX TGS (Temporal Gauss-Seidel) solver warm-starts contact impulses
    # from the reset events' settling physics.  When objects are teleported to
    # their targets, those stale impulses are applied on the FIRST env.step(),
    # causing lateral XY drift (~2 cm) even at rest height.
    # Fix: place BOTH objects first, then one shared physics-only step consumes
    # the stale warm-start for ALL bodies simultaneously; then re-place both
    # into a clean contact state (normal-force only, no lateral impulses).

    # Step 1 — place both objects at their targets (exposes stale warm-starts)
    if cup_wx is not None and cup_wy is not None:
        place_cup_at_coords(env, cup_wx, cup_wy)

    # Step 2 — single sim.step clears TGS warm-start for ALL objects at once
    env.unwrapped.sim.step(render=False)

    # Step 3 — re-place both into the now-clean contact state
    place_cylinder_at_coords(env, wx, wy)
    if cup_wx is not None and cup_wy is not None:
        place_cup_at_coords(env, cup_wx, cup_wy)
        print(f"[CUP] Cup placed at real-world ({cup_wx:+.4f}, {cup_wy:.4f})")
    else:
        print("[CUP] Cup not detected — skipping cup placement")
    print("[FIX] TGS warm-start cleared — re-placed cylinder and cup")


    # ── Load graphpaper mat USD into stage (Option 2) ─────────────────────────
    # Places mat.usda at the correct Isaac world position so the graphpaper
    # texture appears in the SIM render by construction.  The mat's 1 cm texture
    # cells → 10 cm major squares are in exact Isaac world metres; no projection
    # math or overlay needed.  The prim is purely visual (kinematic physics body).
    if ADD_USD_MAT:
        _add_mat_to_stage(stage, MAT_USD_FILE)

    # ── Draw 10 cm grid lines directly in Isaac world frame ───────────────────
    # BasisCurves lines at exact metric positions — metrically correct by
    # construction, no texture repeat or UV mapping involved.
    if DRAW_ISAAC_GRID_LINES:
        _draw_isaac_grid_lines(stage)

    # NOTE: The TGS warm-start fix above calls sim.step(render=False) once and
    # re-places the cylinder.  The second placement lands into a clean PhysX
    # contact state (normal-force only, no lateral impulses from prior settle).
    # write_root_state_to_sim() updates PhysX immediately; the USD t_op.Set()
    # updates the authored USD value.  One simulation_app.update() lets Kit
    # compose the USD stage so the new translate is visible to the RTX renderer
    # before sim.render() fires.  Two sim.render() passes + scene.update()
    # then flush the double-buffered annotator and pull a fresh camera frame.

    # ── Pre-replay capture: render + scene.update() to refresh camera buffer ──
    # CRITICAL: sim.render() alone does NOT update data.output["rgb"].
    # The TiledCamera only pulls new data when scene.update() (or env.step())
    # calls camera.update(dt), which reads the replicator annotator output.
    # Without scene.update(), data.output still holds the INITIAL scene frame
    # (from before env.reset()), regardless of how many times sim.render() runs.
    print("\n[SIM] Flushing render pipeline …")
    simulation_app.update()   # let Kit compose the USD stage with new cylinder translate
    env.unwrapped.sim.render()
    simulation_app.update()
    env.unwrapped.sim.render()
    simulation_app.update()
    # NOTE: scene.update() is intentionally omitted here.
    # Calling scene.update() after place_cylinder_at_coords() runs a full physics
    # step with residual non-deterministic state from the reset_cylinder_position
    # event (seed=None), causing XY drift of up to 4 cm across identical runs.
    # The camera buffer will be refreshed by the first env.step() in the replay
    # loop below, which calls scene.update() internally.
    _physics_dt = env.unwrapped.sim.get_physics_dt()
    _cup_world = (cup_wx, cup_wy) if cup_wx is not None else None

    # ── STEP 6: replay episode frames (primary visual capture at frame 1) ─────
    # env.step() always calls scene.update() internally, so data.output is
    # guaranteed fresh after each step. We capture after frame 1 as a
    # second reference — by then physics has had one full step to settle.
    _n_replay = min(CAPTURE_FRAME, len(frames))
    if _n_replay < CAPTURE_FRAME:
        print(f"[SIM] Episode only has {len(frames)} frames; capping replay to {_n_replay}")
    print(f"\n[SIM] Replaying {_n_replay} frames …")

    # ── PRE-STEP readback: confirm cylinder position before any env.step() ────
    # Compare this to the post-placement readback above.  If they already differ,
    # physics drift is ruled out — something in the setup pipeline moved the cylinder.
    import warnings as _prestep_w
    try:
        with _prestep_w.catch_warnings():
            _prestep_w.simplefilter("ignore", DeprecationWarning)
            _pre_pos = env.unwrapped.scene["cylinder"].data.root_state_w[0, :3].tolist()
        print(f"[PRE-STEP] Cylinder position BEFORE first env.step(): "
              f"Isaac=({_pre_pos[0]:+.4f}, {_pre_pos[1]:.4f}, {_pre_pos[2]:.4f})")
    except Exception as _ppe:
        print(f"[PRE-STEP] Could not read cylinder position: {_ppe}")

    try:
        with _prestep_w.catch_warnings():
            _prestep_w.simplefilter("ignore", DeprecationWarning)
            _cup_pre_pos = env.unwrapped.scene["target"].data.root_state_w[0, :3].tolist()
        print(f"[PRE-STEP] Cup position BEFORE first env.step(): "
              f"Isaac=({_cup_pre_pos[0]:+.4f}, {_cup_pre_pos[1]:.4f}, {_cup_pre_pos[2]:.4f})")
    except Exception as _cppe:
        print(f"[PRE-STEP] Could not read cup position: {_cppe}")

    import warnings as _replay_w
    from omni.timeline import get_timeline_interface
    timeline = get_timeline_interface()
    timeline.play()

    # ── FIX: clear Isaac Lab's done/reset buffer before replay loop ───────────
    # After env.reset(), done=True; first env.step() re-runs reset_cylinder_position
    # event and overrides cylinder placement.  Clearing reset_buf prevents this.
    try:
        _env_unwrapped = env.unwrapped
        if hasattr(_env_unwrapped, 'reset_buf'):
            _env_unwrapped.reset_buf[:] = 0
            print("[FIX] Cleared reset_buf — env.step() will NOT auto-reset on first call")
        elif hasattr(_env_unwrapped, '_reset_buf'):
            _env_unwrapped._reset_buf[:] = 0
            print("[FIX] Cleared _reset_buf")
        elif hasattr(_env_unwrapped, 'episode_length_buf'):
            _env_unwrapped.episode_length_buf[:] = 0
            print("[FIX] Cleared episode_length_buf (fallback)")
        if hasattr(_env_unwrapped, 'termination_manager'):
            _tm = _env_unwrapped.termination_manager
            if hasattr(_tm, 'reset'):
                _tm.reset()
                print("[FIX] Reset termination_manager")
    except Exception as _rb_e:
        print(f"[FIX] reset_buf clear failed: {_rb_e}")

    for frame_idx in range(_n_replay):
        _, row = frames[frame_idx]
        real_action = torch.tensor(
            row["action"], dtype=torch.float32, device=env.unwrapped.device
        )
        sim_action = robot_iface.get_mapped_actions_vectorized(real_action)
        env.step(sim_action.unsqueeze(0))
        with _replay_w.catch_warnings():
            _replay_w.simplefilter("ignore", DeprecationWarning)
            _pos = env.unwrapped.scene["cylinder"].data.root_state_w[0, :3].tolist()
            try:
                _cup_pos_f = env.unwrapped.scene["target"].data.root_state_w[0, :3].tolist()
                _cup_pos_str = f"cup=({_cup_pos_f[0]:+.4f}, {_cup_pos_f[1]:.4f}, {_cup_pos_f[2]:.4f})"
            except Exception:
                _cup_pos_str = "cup=N/A"
        print(f"[SIM] stepped frame {frame_idx + 1}/{CAPTURE_FRAME}  "
              f"cyl=({_pos[0]:+.4f}, {_pos[1]:.4f}, {_pos[2]:.4f})  {_cup_pos_str}")
        # Capture at step 5 (early settle check) and at the LAST replay step.
        if frame_idx == 4:   # step 5 — cylinder should be at rest, arm not yet touching
            _capture_and_annotate(env, wx, wy, episode_idx,
                                  label="post-step-5",
                                  suffix=f"debug_sim_ep{episode_idx:03d}_step005.png",
                                  placement_pos=_placement_isaac,
                                  cup_pos=_cup_isaac,
                                  cup_world=_cup_world)

    print("\n── DONE ─────────────────────────────────────────────────────")
    print(f"  REAL      : {OUT_DIR}/debug_real_ep{episode_idx:03d}_cylinder.png")
    print(f"  SIM step5   : {OUT_DIR}/debug_sim_ep{episode_idx:03d}_step005.png")
    print("White diamond   = Isaac (0,0,0) origin marker (SHOW_ORIGIN_MARKER).")
    print("Yellow dots     = 10 cm Isaac grid intersections (DRAW_ISAAC_GRID_LINES).")
    print("Blue dot        = where the HSV blob detector found the cylinder in SIM.")
    print("Cylinder placed at Isaac (-wx, wy, 0.05) via homography-derived coords.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()