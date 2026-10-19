"""replay_bug.py — Coordinate alignment debugger.

Pipeline:
  1. Read frame-0 from the real overhead video for the chosen episode.
  2. Detect the GREEN CYLINDER via HSV blob detection.
  3. Convert pixel → world coords using _H_REAL (pix2world_real homography).
  4. Save the real frame with a circle drawn at the detected pixel.
  5. Launch Isaac Lab SIM, replay the episode for 5 frames
     (frame 5 is used because the arm is less extended than frame 1/2).
  6. Place a RED SPHERE at the detected world coords in the SIM USD stage.
  7. Project the cylinder world coords → SIM pixel using _H_SIM_INV.
  8. Draw a green crosshair + coordinate label on the SIM frame-5 image and save.

Usage:
    ./isaaclab.sh -p replay_bug.py
    ./isaaclab.sh -p replay_bug.py --episode 1

Outputs (written to ~/so101-sim-project/):
    debug_real_ep{N}_cylinder.png   — real frame with green circle at detected cylinder
    debug_sim_ep{N}_marker.png      — SIM frame-5 with red sphere + green crosshair + coords
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
AppLauncher.add_app_launcher_args(parser)

_ep = 1
for _i, _a in enumerate(sys.argv[1:], 1):
    if _a == "--episode" and _i < len(sys.argv):
        try:
            _ep = int(sys.argv[_i + 1])
        except (IndexError, ValueError):
            pass

_cf = 150   # default: ~5 s at 30 fps; override with --capture_frame N
for _i, _a in enumerate(sys.argv[1:], 1):
    if _a == "--capture_frame" and _i < len(sys.argv):
        try:
            _cf = int(sys.argv[_i + 1])
        except (IndexError, ValueError):
            pass

args_cli = parser.parse_args([
    "--livestream", "2", "--episode", str(_ep),
    "--kit_args",
    "--ext-folder /home/ted/isaac-sim/extsDeprecated "
    "--enable isaacsim.core.utils "
    "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false",
])
args_cli.enable_cameras = True
args_cli.device = "cpu"

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

from homography_pix2world import (
    pix2world_real  as _pix2world,
    _H_REAL,  _H_REAL_INV,
    _H_SIM,   _H_SIM_INV,
)

# ── Constants ─────────────────────────────────────────────────────────────────
OUT_DIR  = os.path.expanduser("~/so101-sim-project")
os.makedirs(OUT_DIR, exist_ok=True)

CAM_PATH = "/World/envs/env_0/CameraOverhead"

# Camera parameters — single source of truth is cylinder_to_target_env_cfg.py.
# Import from there; fall back to the same values if the import fails at startup.
try:
    from sim_to_real_so101.tasks.cylinder_to_target_env_cfg import overhead_camera as _cam_cfg
    _cp = _cam_cfg.offset.pos
    CAM_X, CAM_Y, CAM_Z = float(_cp[0]), float(_cp[1]), float(_cp[2])  # -0.035, 0.210, 0.600
except Exception:
    CAM_X, CAM_Y, CAM_Z = 0.025, 0.40, 0.55   # env_cfg fallback (CAMERA_X, CAMERA_Y, CAMERA_HEIGHT_ABOVE_TABLE)

_CYL_HSV_LO = np.array(( 35,  40,  40), dtype=np.uint8)
_CYL_HSV_HI = np.array(( 85, 255, 255), dtype=np.uint8)

CAPTURE_FRAME = _cf  # set via --capture_frame N (default 150 ≈ 5 s at 30 fps)

# ── Annotation toggles (set False to hide, True to show) ─────────────────────
SHOW_TARGET_CROSSHAIR = False   # green circle+lines at H_SIM-projected real-world position
SHOW_ORIGIN_MARKER    = False   # bright-red diamond at Isaac (0,0) reference point
SHOW_ISAAC_DOT        = False   # red dot at Isaac FK cylinder position
SHOW_BLOB_DOT         = False   # blue circle at HSV blob centroid
SHOW_PLACEMENT_DOT    = False   # orange dot at homography-derived placement position
SHOW_STAGE_RED_SPHERES  = False # 3-D red USD spheres: DebugPlacementMarker + DebugOriginMarker
SHOW_STAGE_GRID_MARKERS = True  # 3-D green USD spheres at grid corner positions

# ── Grid reference positions (Isaac cm) ──────────────────────────────────────
# Used for 3-D stage markers, 2-D image overlays, and real-image back-projection.
# Isaac X = -real_X  (Rz180 camera convention).
GRID_CM = [
    ( 33.0,  0.5),   # mat corner: right-near
    (-40.0,  0.5),   # mat corner: left-near
    ( 31.0, 38.0),   # mat corner: right-far
    (-37.5, 40.0),   # mat corner: left-far
    ( -5.0,  0.5),   # centre column: near edge
    ( -5.0, 39.0),   # centre column: far edge
]

# _H_SIM and _H_SIM_INV imported from homography_pix2world above.


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1-3: detect cylinder in the real overhead video
# ─────────────────────────────────────────────────────────────────────────────
def detect_cylinder_in_real(real_root, episode_idx):
    import cv2, subprocess

    video_path = None
    for pat in [
        f"videos/observation.images.overhead/chunk-000/episode_{episode_idx:06d}.mp4",
        f"videos/observation.images.overhead/chunk-{episode_idx // 500:03d}/episode_{episode_idx:06d}.mp4",
        "videos/observation.images.overhead/chunk-000/file-000.mp4",
    ]:
        p = os.path.join(real_root, pat)
        if os.path.exists(p):
            video_path = p
            break
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

    cmd = ["ffmpeg", "-ss", "0", "-i", video_path,
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True)
    expected = w * h * 3
    if r.returncode != 0 or len(r.stdout) != expected:
        raise RuntimeError(
            f"ffmpeg failed: ret={r.returncode} got={len(r.stdout)} expected={expected}"
        )
    frame = np.frombuffer(r.stdout, dtype=np.uint8).reshape(h, w, 3).copy()
    frame[415:, 455:] = 0

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
    print(f"[REAL] Cylinder pixel centroid: ({px}, {py})")
    wx, wy = _pix2world(px, py)
    print(f"[REAL] Cylinder world coords:   ({wx:+.4f}, {wy:.4f})  metres")
    return frame, px, py, wx, wy


# STEP 4
def save_annotated_real(frame_bgr, px, py, wx, wy, episode_idx):
    import cv2
    out = frame_bgr.copy()
    h_img, w_img = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Cylinder detection circle ─────────────────────────────────────────────
    cv2.circle(out, (px, py), 18, (0, 255, 0), 3)
    cv2.circle(out, (px, py),  3, (0, 255, 0), -1)
    cv2.putText(out, f"cyl ({wx:+.3f}, {wy:.3f})",
                (px + 22, py), font, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    # ── Back-project GRID_CM → real image pixels ──────────────────────────────
    # Isaac → real-world: real_x = -ix,  real_y = iy  (Rz180 convention)
    # Then _H_REAL_INV @ [real_x, real_y, 1]  →  real pixel
    for i, (ix_cm, iy_cm) in enumerate(GRID_CM):
        real_x = -ix_cm / 100.0
        real_y =  iy_cm / 100.0
        try:
            pt = _H_REAL_INV @ np.array([real_x, real_y, 1.0])
            gx = int(round(pt[0] / pt[2]))
            gy = int(round(pt[1] / pt[2]))
            print(f"[REAL] Grid[{i}] Isaac=({ix_cm:+.1f},{iy_cm:.1f})cm "
                  f"real=({real_x:+.3f},{real_y:.3f}) → px ({gx},{gy})")
            if 0 <= gx < w_img and 0 <= gy < h_img:
                cv2.circle(out, (gx, gy), 3, (0, 220, 0), -1)
                cv2.circle(out, (gx, gy), 4, (0,   0, 0),  1)
                label = f"({ix_cm:+.1f},{iy_cm:.1f})cm"
                lx = gx + 6 if gx < w_img // 2 else gx - cv2.getTextSize(label, font, 0.35, 1)[0][0] - 6
                cv2.putText(out, label, (lx, gy + 4), font, 0.35, (0, 220, 0), 1, cv2.LINE_AA)
        except Exception as _e:
            print(f"[REAL] Grid[{i}] back-projection failed: {_e}")

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
    for r in ("xformOp:orient", "xformOp:orientation"):
        if r in ops:
            ops[r].Set(Gf.Quatd(0.0, 0.0, 0.0, 1.0)); break
    if "xformOp:rotateXYZ" in ops:
        ops["xformOp:rotateXYZ"].Set(Gf.Vec3f(0.0, 0.0, 180.0))
    vp_util.get_active_viewport().camera_path = CAM_PATH
    print("[CAM] Viewport → CameraOverhead")




def place_cylinder_at_coords(env, wx, wy):
    """
    Move the SIM cylinder to the detected real-world coordinates.
    Coordinate convention: isaac_X = -world_X, isaac_Y = world_Y.
    The SIM camera uses Rz(180°) so the image is horizontally mirrored;
    Isaac +X appears on the LEFT in the SIM image. Negating world_X maps
    correctly: a cylinder at world_X=-0.313 (left in real) goes to
    isaac_X=+0.313 (left in SIM). Confirmed by visual observation.
    Writes to both PhysX (write_root_state_to_sim) and the USD authored
    xformOp:translate so RTX renders it at the correct position.
    """
    import warnings as _w
    from pxr import UsdGeom, Gf
    import omni.usd

    # Isaac Sim world X is inverted relative to real-camera world X because
    # the overhead camera is rotated Rz(180°).  The _H_SIM homography was
    # calibrated in REAL-camera convention (left pixel → negative real X).
    # To make the cylinder appear on the LEFT of the SIM image we must negate:
    #   Isaac X = -wx   (so Rz180 camera maps it to the LEFT)
    #   Isaac Y =  wy   (Y axis is the same in both frames)
    xyz    = (float(-wx), float(wy), 0.050)
    device = env.unwrapped.device
    obj    = env.unwrapped.scene["cylinder"]

    # ── Read BEFORE so we can see what reset left it at ──────────────────────
    with _w.catch_warnings():
        _w.simplefilter("ignore", DeprecationWarning)
        before = obj.data.root_state_w[0, :3].tolist()
    print(f"[PLACE] Cylinder before placement: Isaac=({before[0]:+.4f}, {before[1]:.4f}, {before[2]:.4f})")

    # ── Write via PhysX view directly (most reliable path) ───────────────────
    import omni.physics.tensors as tensors   # noqa: F401 — ensure tensor API ready
    try:
        # Preferred: use the raw PhysX rigid body view to set transforms
        pos_t   = torch.tensor([[float(-wx), float(wy), 0.050]],
                               dtype=torch.float32, device=device)
        rot_t   = torch.tensor([[1.0, 0.0, 0.0, 0.0]],          # w x y z
                               dtype=torch.float32, device=device)
        vel_t   = torch.zeros(1, 6, dtype=torch.float32, device=device)
        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            obj.root_physx_view.set_root_transforms(
                torch.cat([pos_t, rot_t], dim=1).cpu().numpy(), indices=None
            )
            obj.root_physx_view.set_root_velocities(vel_t.cpu().numpy(), indices=None)
        physx_ok = True
    except Exception as _pe:
        physx_ok = False
        print(f"[PLACE] PhysX view set failed ({_pe}) — falling back to write_root_state_to_sim")

    if not physx_ok:
        state = obj.data.root_state_w.clone()
        state[:, :3]  = torch.tensor([xyz], dtype=torch.float32, device=device)
        state[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        state[:, 7:]  = 0.0
        obj.data.root_state_w[:] = state
        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            obj.write_root_state_to_sim(state)

    # ── USD authored translate so RTX render agrees with PhysX ───────────────
    stage = omni.usd.get_context().get_stage()
    try:
        leaf_path  = obj.root_physx_view.prim_paths[0]
        prim_path  = "/".join(leaf_path.split("/")[:5])   # /World/envs/env_0/Cylinder
    except AttributeError:
        prim_path  = obj.cfg.prim_path.replace("{ENV_IDX}", "0")
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
        print(f"[PLACE] Cylinder → {prim_path}  Isaac=({-wx:+.4f}, {wy:.4f}, 0.050)  real=({wx:+.4f}, {wy:.4f})")
    else:
        print(f"[PLACE] WARNING: prim not found at {prim_path}")

    # ── Read AFTER to confirm PhysX accepted the new position ────────────────
    with _w.catch_warnings():
        _w.simplefilter("ignore", DeprecationWarning)
        after = obj.data.root_state_w[0, :3].tolist()
    print(f"[PLACE] Cylinder after placement:  Isaac=({after[0]:+.4f}, {after[1]:.4f}, {after[2]:.4f})")
    exp = (-wx, wy, 0.050)   # Isaac X = -real_X (Rz180 camera convention)
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


def _place_at_isaac_origin(env, stage):
    """Place cylinder + red sphere at Isaac (0,0,0.05) directly — no homography."""
    import warnings as _w
    from pxr import UsdGeom, Gf, Sdf, Vt
    import omni.usd

    xyz    = (0.0, 0.0, 0.050)
    device = env.unwrapped.device
    obj    = env.unwrapped.scene["cylinder"]

    # ── PhysX: try raw view first, fall back to write_root_state_to_sim ─────
    try:
        import omni.physics.tensors as _tensors   # noqa
        pos_t = torch.tensor([list(xyz)],          dtype=torch.float32, device=device)
        rot_t = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        vel_t = torch.zeros(1, 6,                  dtype=torch.float32, device=device)
        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            obj.root_physx_view.set_root_transforms(
                torch.cat([pos_t, rot_t], dim=1).cpu().numpy(), indices=None
            )
            obj.root_physx_view.set_root_velocities(vel_t.cpu().numpy(), indices=None)
    except Exception as _pe:
        state = obj.data.root_state_w.clone()
        state[:, :3]  = torch.tensor([list(xyz)], dtype=torch.float32, device=device)
        state[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        state[:, 7:]  = 0.0
        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            obj.data.root_state_w[:] = state
            obj.write_root_state_to_sim(state)
        print(f"[ORIGIN] PhysX view fallback ({_pe})")

    # ── USD authored translate so RTX renderer agrees ─────────────────────────
    if stage is None:
        stage = omni.usd.get_context().get_stage()
    try:
        leaf_path = obj.root_physx_view.prim_paths[0]
        cyl_path  = "/".join(leaf_path.split("/")[:5])
    except AttributeError:
        cyl_path  = obj.cfg.prim_path.replace("{ENV_IDX}", "0")
    cyl = stage.GetPrimAtPath(cyl_path)
    if cyl.IsValid():
        xf   = UsdGeom.Xformable(cyl)
        t_op = next(
            (op for op in xf.GetOrderedXformOps()
             if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None
        )
        if t_op is None:
            t_op = xf.AddTranslateOp()
        t_op.Set(Gf.Vec3d(*xyz))
        print(f"[ORIGIN] Cylinder USD translate → {cyl_path}  Isaac=(0.000, 0.000, 0.050)")
    else:
        print(f"[ORIGIN] WARNING: cylinder prim not found at {cyl_path}")

    # ── Visual-only red sphere at Isaac origin ────────────────────────────────
    if SHOW_STAGE_RED_SPHERES:
        mp = Sdf.Path("/World/DebugOriginMarker")
        if stage.GetPrimAtPath(mp).IsValid():
            stage.RemovePrim(mp)
        sp = UsdGeom.Sphere.Define(stage, mp)
        sp.GetRadiusAttr().Set(0.010)
        sp.GetDisplayColorAttr().Set(Vt.Vec3fArray([(1.0, 0.15, 0.15)]))
        UsdGeom.Xformable(sp).AddTranslateOp().Set(Gf.Vec3d(*xyz))
        print("[ORIGIN] Red sphere marker → /World/DebugOriginMarker  Isaac=(0.000, 0.000, 0.050)")


def _place_grid_markers(stage, positions_m):
    """Place GREEN visual-only USD Spheres at each Isaac (x, y, z) position.

    positions_m: list of (ix, iy, iz) in Isaac metres.
    These do not affect physics — they are display markers only.
    """
    from pxr import UsdGeom, Gf, Sdf, Vt
    for i, (ix, iy, iz) in enumerate(positions_m):
        mp = Sdf.Path(f"/World/DebugGridMarker_{i}")
        if stage.GetPrimAtPath(mp).IsValid():
            stage.RemovePrim(mp)
        sp = UsdGeom.Sphere.Define(stage, mp)
        sp.GetRadiusAttr().Set(0.010)                                    # 10 mm sphere
        sp.GetDisplayColorAttr().Set(Vt.Vec3fArray([(0.1, 0.9, 0.1)]))  # green
        UsdGeom.Xformable(sp).AddTranslateOp().Set(Gf.Vec3d(ix, iy, iz))
        ix_cm = int(round(ix * 100)); iy_cm = int(round(iy * 100))
        print(f"[GRID] Green sphere #{i} → /World/DebugGridMarker_{i}  "
              f"Isaac=({ix:+.3f},{iy:+.3f})  = ({ix_cm:+d},{iy_cm:+d}) cm")



def annotate_and_save_sim(rgb_frame, wx, wy, path_out, label_tag="", isaac_pos=None,
                          placement_pos=None, origin_pos=None, grid_positions=None):
    """Draw green crosshair + coord label on the SIM image, detect green blob, save.

    isaac_pos:     [ix, iy, iz] — Isaac world coords of the cylinder at capture time
                   (read from root_state_w).  Shows where the cylinder actually IS.
    placement_pos: [ix, iy, iz] — Isaac coords of the homography-derived placement
                   (before origin move).  Orange dot.
    origin_pos:    [ix, iy, iz] — Hardcoded Isaac (0,0,0.05) reference dot.  Always
                   drawn at the same pixel regardless of where the cylinder drifts.
                   Bright red with a diamond outline so it reads clearly.
    grid_positions: list of (ix, iy, iz) in Isaac metres — additional reference dots
                   drawn as small red circles with cm labels, e.g. (0.20,0,0.05).
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

    # ── Project world → SIM pixel (green crosshair = expected position) ──────
    try:
        pt  = _H_SIM_INV @ np.array([wx, wy, 1.0])
        spx = int(round(pt[0] / pt[2]))
        spy = int(round(pt[1] / pt[2]))
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
    except Exception as e:
        print(f"[SIM{label_tag}] Crosshair projection failed: {e}")

    # ── Red dot: Isaac placement position projected to SIM pixel ─────────────
    # Isaac X = -real_X, so real_wx = -isaac_x. Project via _H_SIM_INV.
    # This dot shows where Isaac says the cylinder is, regardless of HSV.
    # If it matches the crosshair: _H_SIM calibration is accurate.
    # If it matches the blue blob: physics and rendering agree with Isaac.
    if isaac_pos is not None:
        try:
            real_wx_i = -float(isaac_pos[0])   # Isaac X → real X
            pt3 = _H_SIM_INV @ np.array([real_wx_i, float(isaac_pos[1]), 1.0])
            rpx = int(round(pt3[0] / pt3[2]))
            rpy = int(round(pt3[1] / pt3[2]))
            print(f"[SIM{label_tag}] Isaac  ({isaac_pos[0]:+.4f},{isaac_pos[1]:.4f}) → SIM px ({rpx},{rpy})")
            if SHOW_ISAAC_DOT and 0 <= rpx < w_img and 0 <= rpy < h_img:
                cv2.circle(img, (rpx, rpy),  7, (0, 0, 200), -1)   # red filled dot
                cv2.circle(img, (rpx, rpy),  8, (0,  0,  0),  1)   # thin black ring
                _label(img,
                       [f"Isaac ({isaac_pos[0]:+.3f},{isaac_pos[1]:.3f})",
                        f"px=({rpx},{rpy})"],
                       rpx, rpy, side="right")
        except Exception as _re:
            print(f"[SIM{label_tag}] Red dot projection failed: {_re}")

    # ── Orange dot: homography-derived placement position (before origin move) ─
    # Distinct from isaac_pos (cylinder's current position).  Shows where the
    # homography said to place the cylinder; if the cylinder was subsequently
    # moved to origin, both dots appear in the image for easy comparison.
    if placement_pos is not None:
        try:
            real_wx_p = -float(placement_pos[0])   # Isaac X → real X
            pt4 = _H_SIM_INV @ np.array([real_wx_p, float(placement_pos[1]), 1.0])
            ppx = int(round(pt4[0] / pt4[2]))
            ppy = int(round(pt4[1] / pt4[2]))
            print(f"[SIM{label_tag}] Place  ({placement_pos[0]:+.4f},{placement_pos[1]:.4f}) → SIM px ({ppx},{ppy})")
            if SHOW_PLACEMENT_DOT and 0 <= ppx < w_img and 0 <= ppy < h_img:
                cv2.circle(img, (ppx, ppy),  7, (0, 100, 255), -1)   # orange filled dot
                cv2.circle(img, (ppx, ppy),  8, (0,   0,   0),  1)   # thin black ring
                _label(img,
                       [f"place ({placement_pos[0]:+.3f},{placement_pos[1]:.3f})",
                        f"px=({ppx},{ppy})"],
                       ppx, ppy, side="right")
        except Exception as _pe2:
            print(f"[SIM{label_tag}] Orange placement dot projection failed: {_pe2}")

    # ── Bright-red dot: FIXED Isaac origin (0,0) — hardcoded, never drifts ───
    # Projected via _H_SIM_INV @ [0, 0, 1] directly — no homography conversion
    # of detected wx/wy.  This is the ground-truth pixel for Isaac (0,0,0.05).
    # Drawn as a filled circle + diamond outline so it's visually distinct from
    # the "Isaac" dot (which follows the cylinder's live state and may drift).
    if origin_pos is not None:
        try:
            # Isaac origin: ix=0, iy=0 → real_wx=0, real_wy=0
            ox_isaac = float(origin_pos[0])
            oy_isaac = float(origin_pos[1])
            real_wx_o = -ox_isaac   # Isaac X = -real_X  (Rz180 convention)
            pt5 = _H_SIM_INV @ np.array([real_wx_o, float(oy_isaac), 1.0])
            opx = int(round(pt5[0] / pt5[2]))
            opy = int(round(pt5[1] / pt5[2]))
            print(f"[SIM{label_tag}] Origin ({ox_isaac:+.4f},{oy_isaac:.4f}) → SIM px ({opx},{opy})")
            if SHOW_ORIGIN_MARKER and 0 <= opx < w_img and 0 <= opy < h_img:
                # Bright filled circle
                cv2.circle(img, (opx, opy), 8, (0, 0, 255), -1)   # bright red
                cv2.circle(img, (opx, opy), 9, (0,   0,   0),  1) # black ring
                # Diamond outline so it's unmistakable
                d = 14
                diamond = np.array([
                    [opx,     opy - d],
                    [opx + d, opy    ],
                    [opx,     opy + d],
                    [opx - d, opy    ],
                ], dtype=np.int32)
                cv2.polylines(img, [diamond], isClosed=True, color=(0, 0, 255), thickness=2)
                _label(img,
                       [f"origin (0.000,0.000)",
                        f"px=({opx},{opy})"],
                       opx, opy, side="right")
        except Exception as _oe:
            print(f"[SIM{label_tag}] Origin dot projection failed: {_oe}")

    # ── Grid labels: 2D green dot + text at each grid marker's projected pixel ──
    # The dot is drawn in 2D so it is always visible even when the 3D sphere is
    # occluded by the robot arm.
    if grid_positions:
        for _gi, (_gix, _giy, _giz) in enumerate(grid_positions):
            try:
                _real_wx_g = -float(_gix)   # Isaac X = -real_X (Rz180 convention)
                _ptg = _H_SIM_INV @ np.array([_real_wx_g, float(_giy), 1.0])
                _gpx = int(round(_ptg[0] / _ptg[2]))
                _gpy = int(round(_ptg[1] / _ptg[2]))
                _ix_cm = _gix * 100
                _iy_cm = _giy * 100
                print(f"[SIM{label_tag}] Grid[{_gi}] ({_gix:+.3f},{_giy:+.3f}) "
                      f"= ({_ix_cm:+.2f},{_iy_cm:.2f})cm → px ({_gpx},{_gpy})")
                if 0 <= _gpx < w_img and 0 <= _gpy < h_img:
                    # 2D green dot — visible even when 3D sphere is behind the arm
                    cv2.circle(img, (_gpx, _gpy), 3, (0, 220, 0), -1)
                    cv2.circle(img, (_gpx, _gpy), 4, (0,   0, 0),  1)   # thin black ring
                    # Pick label side so it stays inside the image
                    _gside = "left" if _gpx > w_img // 2 else "right"
                    _label(img,
                           [f"({_ix_cm:+.2f},{_iy_cm:.2f})cm"],
                           _gpx, _gpy, side=_gside)
            except Exception as _ge:
                print(f"[SIM{label_tag}] Grid[{_gi}] label failed: {_ge}")

    # ── Top-left label ────────────────────────────────────────────────────────
    text  = f"world=({wx:+.2f}, {wy:.2f})  {label_tag}"
    scale2, thick2 = 0.5, 2
    (tw, th), bl = cv2.getTextSize(text, font, scale2, thick2)
    cv2.rectangle(img, (8, 8), (8 + tw + 8, 8 + th + bl + 8), (0, 0, 0), -1)
    cv2.putText(img, text, (12, 8 + th + 2), font, scale2, (0, 255, 0), thick2, cv2.LINE_AA)

    cv2.imwrite(path_out, img)
    print(f"[SIM{label_tag}] Saved → {path_out}")


def _capture_and_annotate(env, wx, wy, episode_idx, label, suffix,
                          placement_pos=None, origin_pos=None, grid_positions=None):
    """Read camera buffer, run blob detection, draw crosshair + Isaac dots, save.

    placement_pos:  Isaac coords of the homography-derived placement (orange dot).
    origin_pos:     Hardcoded Isaac (0,0,0.05) reference (bright-red diamond).
    grid_positions: list of (ix,iy,iz) Isaac metres — label text at projected pixel.

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
                              placement_pos=placement_pos, origin_pos=origin_pos,
                              grid_positions=grid_positions)
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
    print(f"  replay_bug.py   episode={episode_idx}  capture_frame={CAPTURE_FRAME}")
    print("=" * 60)

    # ── STEP 1-4: detect cylinder in real image ───────────────────────────────
    print("\n── REAL IMAGE ───────────────────────────────────────────────")
    try:
        frame_bgr, px, py, wx, wy = detect_cylinder_in_real(real_root, episode_idx)
        save_annotated_real(frame_bgr, px, py, wx, wy, episode_idx)
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

    # Hide the yellow cup so only the green cylinder is visible
    try:
        from pxr import UsdGeom
        target_prim = stage.GetPrimAtPath("/World/envs/env_0/Target")
        if target_prim.IsValid():
            UsdGeom.Imageable(target_prim).MakeInvisible()
            print("[SIM] Target/cup hidden")
        else:
            print("[SIM] /World/envs/env_0/Target not found — cup not hidden")
    except Exception as e:
        print(f"[SIM] Could not hide cup: {e}")

    simulation_app.update()

    # ── Place cylinder at real-detected coords, then move to Isaac origin ──────
    # Step A: homography-derived placement puts cylinder at Isaac(-wx, wy) and
    #         leaves a DebugPlacementMarker red sphere there for visual reference.
    place_cylinder_at_coords(env, wx, wy)
    _placement_isaac = (float(-wx), float(wy), 0.050)   # save before overwriting

    # Step B: move cylinder directly to Isaac (0, 0, 0.05) — no homography.
    #         Also plants DebugOriginMarker red sphere at the origin.
    #         The annotation captures will show both positions as dots.
    _place_at_isaac_origin(env, stage)

    # Step C: plant green visual-only USD spheres at each grid reference point.
    #         Coordinates are in cm; divide by 100 for Isaac metres.
    #         These do NOT move the cylinder — they are fixed 3D reference markers.
    _grid_positions_m = [(x / 100.0, y / 100.0, 0.050) for x, y in GRID_CM]
    if SHOW_STAGE_GRID_MARKERS:
        _place_grid_markers(stage, _grid_positions_m)

    # NOTE: Do NOT call sim.step(render=False) here — extra physics steps
    # cause the cylinder to drift from its placed position before capture.
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
    # Pull fresh annotator data into the camera's data.output buffer.
    _physics_dt = env.unwrapped.sim.get_physics_dt()
    env.unwrapped.scene.update(_physics_dt)

    print("\n[SIM] Capturing overhead camera (pre-replay, after scene.update) …")
    _pre_frame = _capture_and_annotate(env, wx, wy, episode_idx, label="pre-replay",
                                       suffix=f"debug_sim_ep{episode_idx:03d}_marker.png",
                                       placement_pos=_placement_isaac,
                                       origin_pos=[0.0, 0.0, 0.050],
                                       grid_positions=_grid_positions_m)

    # ── STEP 6: replay episode frames (primary visual capture at frame 1) ─────
    # env.step() always calls scene.update() internally, so data.output is
    # guaranteed fresh after each step. We capture after frame 1 as a
    # second reference — by then physics has had one full step to settle.
    _n_replay = min(CAPTURE_FRAME, len(frames))
    if _n_replay < CAPTURE_FRAME:
        print(f"[SIM] Episode only has {len(frames)} frames; capping replay to {_n_replay}")
    print(f"\n[SIM] Replaying {_n_replay} frames …")
    import warnings as _replay_w
    from omni.timeline import get_timeline_interface
    timeline = get_timeline_interface()
    timeline.play()

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
        print(f"[SIM] stepped frame {frame_idx + 1}/{CAPTURE_FRAME}  "
              f"cyl=({_pos[0]:+.4f}, {_pos[1]:.4f}, {_pos[2]:.4f})")
        # Capture at the LAST replay step so the arm has had time to retract.
        if frame_idx == _n_replay - 1:
            _capture_and_annotate(env, wx, wy, episode_idx,
                                  label=f"post-step-{_n_replay}",
                                  suffix=f"debug_sim_ep{episode_idx:03d}_step{_n_replay}.png",
                                  placement_pos=_placement_isaac,
                                  origin_pos=[0.0, 0.0, 0.050],
                                  grid_positions=_grid_positions_m)

    print("\n── DONE ─────────────────────────────────────────────────────")
    print(f"  REAL      : {OUT_DIR}/debug_real_ep{episode_idx:03d}_cylinder.png")
    print(f"  SIM pre   : {OUT_DIR}/debug_sim_ep{episode_idx:03d}_marker.png")
    print(f"  SIM step{_n_replay} : {OUT_DIR}/debug_sim_ep{episode_idx:03d}_step{_n_replay}.png")
    print("Green crosshair = _H_SIM projection of detected real-world coords.")
    print("Blue dot        = where the HSV blob detector found the cylinder in SIM.")
    print("Red dot         = where Isaac reports the cylinder (post-origin move: 0,0).")
    print("Orange dot      = homography-derived Isaac placement pos before origin move.")
    print("3D scene:  DebugPlacementMarker (red sphere) at Isaac(-wx, wy).")
    print("           DebugOriginMarker    (red sphere) at Isaac(0, 0).")
    print("Cylinder ends at Isaac (0, 0, 0.05).")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()