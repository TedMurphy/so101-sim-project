# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive camera-alignment viewer (WebRTC streaming).

1. Run this script with ./isaaclab.sh -p
2. Open your browser to http://localhost:8211/streaming/webrtc-demo/
3. Orbit the viewport to match the real overhead camera angle.
4. Press PLAY (▶) in the toolbar — the episode replays and auto-resets.
5. When the angle looks right:
     Create > Camera > Camera From View
   then read Translate XYZ / Orient XYZ from the Property panel.
6. Ctrl-C in the terminal to quit.
"""

import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive camera-alignment viewer.")
parser.add_argument("--task",         type=str, default="Lerobot-So101-Teleop-Cylinder-To-Target")
parser.add_argument("--real_repo_id", type=str, default="Screener2/V4_pnp_44_20260720_141907")
parser.add_argument("--episode",      type=int, default=0)
AppLauncher.add_app_launcher_args(parser)

import sys  # noqa: E402
# Read --episode from the real command line; keep required sim flags
_ep = 0
for _i, _a in enumerate(sys.argv[1:], 1):
    if _a == '--episode' and _i < len(sys.argv):
        _ep = int(sys.argv[_i + 1]) if _i + 1 < len(sys.argv) else 0
args_cli = parser.parse_args([
    '--livestream', '2', '--episode', str(_ep),
    '--kit_args',
    '--ext-folder /home/ted/isaac-sim/extsDeprecated --enable isaacsim.core.utils',
])
args_cli.enable_cameras = True
args_cli.device = 'cpu'

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import pandas as pd
import isaaclab_tasks                           # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks                  # noqa: F401
from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface


EPISODE_POSITIONS = {
    ## Calibration: OX=0.210101 SCALE_X=0.001154 OY=-0.004405 SCALE_Y=0.001189
    ## Corrections: DX_CYL=-0.060 DY_CYL=+0.020 DX_CUP=-0.040 DY_CUP=+0.000
    # Calibration: OX=0.210101 SCALE_X=0.001154 OY=-0.004405 SCALE_Y=0.001189
    # Corrections: none (DX corrections were wrong -- use RAW positions)
     0: {"cylinder": (+0.194, -0.272, 0.050), "target":   (+0.409, +0.282, 0.045)},
     1: {"cylinder": (+0.245, -0.202, 0.050), "target":   (+0.175, +0.271, 0.045)},
     2: {"cylinder": (+0.069, -0.232, 0.050), "target":   (+0.415, +0.275, 0.045)},
     3: {"cylinder": (+0.270, -0.296, 0.050), "target":   (+0.412, +0.265, 0.045)},
     4: {"cylinder": (+0.112, -0.288, 0.050), "target":   (+0.131, +0.277, 0.045)},
     5: {"cylinder": (+0.074, -0.248, 0.050), "target":   (+0.073, +0.290, 0.045)},
     6: {"cylinder": (-0.007, -0.270, 0.050), "target":   (+0.021, +0.281, 0.045)},
     #7: {"cylinder": (+0.284, -0.279, 0.050), "target":   (+0.021, +0.281, 0.045)},
     7: {"cylinder": (+0.233, -0.263, 0.050), "target":   (+0.021, +0.281, 0.045)},
     8: {"cylinder": (+0.246, -0.299, 0.050), "target":   (+0.021, +0.277, 0.045)},
     9: {"cylinder": (+0.181, -0.170, 0.050), "target":   (+0.028, +0.268, 0.045)},
    10: {"cylinder": (+0.129, -0.145, 0.050), "target":   (+0.029, +0.267, 0.045)},
    11: {"cylinder": (+0.129, -0.127, 0.050), "target":   (+0.029, +0.269, 0.045)},
    12: {"cylinder": (+0.039, -0.173, 0.050), "target":   (+0.027, +0.263, 0.045)},
    13: {"cylinder": (+0.028, -0.201, 0.050), "target":   (+0.030, +0.263, 0.045)},
    14: {"cylinder": (+0.015, -0.240, 0.050), "target":   (+0.042, +0.199, 0.045)},
    15: {"cylinder": (+0.095, -0.188, 0.050), "target":   (+0.124, +0.217, 0.045)},
    16: {"cylinder": (+0.209, -0.165, 0.050), "target":   (+0.416, +0.242, 0.045)},
    17: {"cylinder": (+0.227, -0.148, 0.050), "target":   (+0.415, +0.220, 0.045)},
    18: {"cylinder": (+0.260, -0.234, 0.050), "target":   (+0.267, +0.265, 0.045)},
    19: {"cylinder": (+0.223, -0.196, 0.050), "target":   (+0.414, +0.271, 0.045)},
    20: {"cylinder": (+0.274, -0.070, 0.050), "target":   (+0.012, +0.243, 0.045)},
    21: {"cylinder": (+0.222, -0.089, 0.050), "target":   (+0.299, +0.270, 0.045)},
    22: {"cylinder": (+0.188, -0.332, 0.050), "target":   (+0.190, +0.315, 0.045)},
    23: {"cylinder": (+0.163, -0.332, 0.050), "target":   (+0.140, +0.210, 0.045)},
    24: {"cylinder": (+0.069, -0.187, 0.050), "target":   (+0.064, +0.227, 0.045)},
    25: {"cylinder": (+0.113, -0.153, 0.050), "target":   (+0.049, +0.224, 0.045)},
    26: {"cylinder": (+0.302, -0.065, 0.050), "target":   (+0.047, +0.224, 0.045)},
    27: {"cylinder": (+0.290, -0.132, 0.050), "target":   (+0.047, +0.230, 0.045)},
    28: {"cylinder": (+0.252, -0.206, 0.050), "target":   (+0.064, +0.241, 0.045)},
    29: {"cylinder": (+0.160, -0.273, 0.050), "target":   (+0.291, +0.243, 0.045)},
    30: {"cylinder": (+0.252, -0.214, 0.050), "target":   (+0.290, +0.243, 0.045)},
    31: {"cylinder": (+0.209, -0.192, 0.050), "target":   (+0.415, +0.267, 0.045)},
    32: {"cylinder": (+0.392, -0.189, 0.050), "target":   (+0.164, +0.295, 0.045)},
    33: {"cylinder": (+0.164, -0.139, 0.050), "target":   (+0.073, +0.248, 0.045)},
    34: {"cylinder": (-0.018, -0.279, 0.050), "target":   (+0.021, +0.254, 0.045)},
    35: {"cylinder": (+0.197, -0.209, 0.050), "target":   (+0.109, +0.245, 0.045)},
    36: {"cylinder": (+0.230, -0.285, 0.050), "target":   (+0.177, +0.308, 0.045)},
    37: {"cylinder": (+0.301, -0.085, 0.050), "target":   (+0.190, +0.315, 0.045)},
    38: {"cylinder": (+0.271, -0.115, 0.050), "target":   (+0.169, +0.289, 0.045)},
    39: {"cylinder": (+0.230, -0.216, 0.050), "target":   (+0.102, +0.319, 0.045)},
    40: {"cylinder": (+0.151, -0.225, 0.050), "target":   (+0.078, +0.245, 0.045)},
    41: {"cylinder": (+0.169, -0.166, 0.050), "target":   (+0.102, +0.311, 0.045)},
    42: {"cylinder": (+0.297, -0.106, 0.050), "target":   (+0.115, +0.202, 0.045)},
    43: {"cylinder": (+0.178, -0.188, 0.050), "target":   (+0.415, +0.232, 0.045)},
    44: {"cylinder": (+0.266, -0.169, 0.050), "target":   (+0.411, +0.299, 0.045)},
}

def set_viewport_camera():
    """Configure CameraOverhead for training data capture and switch the
    streaming viewport back to the default Perspective camera."""
    import math
    import omni.kit.viewport.utility as vp_util

    CAM_X      = 0.21
    CAM_Y      = 0.025
    CAM_Z      = 0.52
    TILT_DEG   = 5.0
    FOCAL_MM   = 21.0
    H_APERTURE = 36.0   # horizontal sensor mm
    V_APERTURE = 22.5   # vertical sensor mm  (4:3 ratio)

    tilt_rad = math.radians(TILT_DEG)
    _cT, _sT = math.cos(tilt_rad / 2), math.sin(tilt_rad / 2)

    # q_tilt: rotation about world +Y axis (tilts toward near side / robot)
    _w1, _x1, _y1, _z1 = _cT, 0.0, +_sT, 0.0
    # q_base: 90 deg about world +Z (image-up = world -X, image-right = world +Y)
    _w2, _x2, _y2, _z2 = 0.7071, 0.0, 0.0, 0.7071
    # q_combined = q_tilt * q_base
    qw = _w1*_w2 - _x1*_x2 - _y1*_y2 - _z1*_z2
    qx = _w1*_x2 + _x1*_w2 + _y1*_z2 - _z1*_y2
    qy = _w1*_y2 - _x1*_z2 + _y1*_w2 + _z1*_x2
    qz = _w1*_z2 + _x1*_y2 - _y1*_x2 + _z1*_w2

    vp = vp_util.get_active_viewport()
    cam_path = "/World/envs/env_0/CameraOverhead"
    from pxr import UsdGeom, Gf, Sdf
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    # ── Remove any leftover debug camera prims from previous runs ─────────────
    for _old in ["/World/envs/env_0/DebugSideCamera",
                 "/World/envs/env_0/DebugPerspCamera"]:
        if stage.GetPrimAtPath(_old).IsValid():
            stage.RemovePrim(Sdf.Path(_old))
            print(f"[camera] removed old prim {_old}")

    # ── Configure CameraOverhead ──────────────────────────────────────────────
    cam_prim = stage.GetPrimAtPath(cam_path)
    if not cam_prim.IsValid():
        print(f"[camera] WARNING: {cam_path} not found, skipping")
        return
    xf = UsdGeom.Xformable(cam_prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(CAM_X, CAM_Y, CAM_Z))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(qw, qx, qy, qz))
    cam_api = UsdGeom.Camera(cam_prim)
    cam_api.GetFocalLengthAttr().Set(FOCAL_MM)
    cam_api.GetHorizontalApertureAttr().Set(H_APERTURE)
    cam_api.GetVerticalApertureAttr().Set(V_APERTURE)
    print(f"[camera] overhead set: pos=(0.21,0.025,0.52) tilt=5.0deg "
          f"focal={FOCAL_MM}mm h={H_APERTURE}mm v={V_APERTURE}mm")

    # Switch streaming viewport back to the default Perspective camera.
    # CameraOverhead still captures training data independently via rgb_overhead.
    vp.camera_path = "/OmniverseKit_Persp"
    print("[camera] viewport → Perspective")

    # Restore /OmniverseKit_Persp to the preferred close-up debugging angle.
    # Eye/target computed from Property panel values:
    #   Translate=(-0.42963, 0.01366, 0.74622)  RotXYZ=(39.74, 0, -90.29 deg)
    # /OmniverseKit_Persp is a Kit "free camera" — its USD prim is overwritten
    # each frame by the internal controller, so we must go through set_camera_view.
    PERSP_EYE    = (-0.42963, 0.01366, 0.74622)
    PERSP_TARGET = ( 0.19,    0.01,    0.00   )  # look-at: mat/robot area

    _set_ok = False
    try:
        try:
            from omni.isaac.core.utils.viewports import set_camera_view
        except ImportError:
            from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(
            eye=PERSP_EYE,
            target=PERSP_TARGET,
            camera_prim_path="/OmniverseKit_Persp",
            viewport_api=vp,
        )
        _set_ok = True
        print(f"[persp] set_camera_view OK  eye={PERSP_EYE}  target={PERSP_TARGET}")
    except Exception as _e:
        print(f"[persp] set_camera_view failed: {_e}")

    if not _set_ok:
        # Diagnostic fallback: report available xform ops so we know what to use
        persp_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
        if persp_prim.IsValid():
            _p_ops = {op.GetOpName(): op for op in UsdGeom.Xformable(persp_prim).GetOrderedXformOps()}
            print(f"[persp] ops on /OmniverseKit_Persp: {list(_p_ops.keys())}")
            for _n, _op in _p_ops.items():
                print(f"[persp]   {_n} = {_op.Get()}")
        else:
            print("[persp] /OmniverseKit_Persp prim not found in stage")


# ── Dynamic detection from real overhead video ─────────────────────────────
# Camera calibration (same parameters as fix_clamp.py from real session)
_DET_IMG_CX  = 320        # image centre col  (pixels)
_DET_IMG_CY  = 240        # image centre row  (pixels)
_DET_SCALE_X = 0.000816   # m/pixel  (image row  → world X)
_DET_SCALE_Y = 0.001081   # m/pixel  (image col  → world Y)
_DET_OX      = 0.090      # world X at image centre
_DET_OY      = 0.008      # world Y at image centre
# Exclude zone: robot body / arm appears here in the overhead view
_DET_EXCL   = dict(x_min=0.22, x_max=0.38, y_min=0.17, y_max=0.38)
# Yellow/orange HSV range (cylinder and cup share this colour)
_DET_HSV_LO = (15,  50,  50)
_DET_HSV_HI = (38, 255, 255)
_DET_HSV_LO_CYL = (35, 40, 40)   # green cylinder
_DET_HSV_HI_CYL = (85, 255, 255)


def _det_pix2world(px_col, py_row):
    """Pixel (col, row) → (world_X, world_Y)."""
    wx = _DET_OX + (py_row - _DET_IMG_CY) * _DET_SCALE_X
    wy = _DET_OY + (px_col - _DET_IMG_CX) * _DET_SCALE_Y
    return wx, wy


def _det_in_excl(wx, wy):
    e = _DET_EXCL
    return e["x_min"] <= wx <= e["x_max"] and e["y_min"] <= wy <= e["y_max"]


def _det_find_blob(frame_bgr, col_side="any", hsv_lo=None, hsv_hi=None):
    """
    Largest yellow/orange blob on a given image half.
    col_side: 'left' (px < IMG_CX → world Y < 0, cylinder side)
              'right' (px > IMG_CX → world Y > 0, target side)
              'any'
    Returns (world_X, world_Y) or None.
    """
    import cv2, numpy as np
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lo   = np.array(hsv_lo if hsv_lo is not None else _DET_HSV_LO, dtype=np.uint8)
    hi   = np.array(hsv_hi if hsv_hi is not None else _DET_HSV_HI, dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    if col_side == "left":
        mask[:, _DET_IMG_CX:] = 0
    elif col_side == "right":
        mask[:, :_DET_IMG_CX] = 0
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return _det_pix2world(cx, cy)


def _det_get_video(real_root, ep_idx):
    """Return (video_path, start_offset_s) for the episode."""
    # Per-episode files — offset is 0
    for pat in [
        f"videos/observation.images.overhead/chunk-000/episode_{ep_idx:06d}.mp4",
        f"videos/observation.images.overhead/chunk-{ep_idx // 500:03d}/episode_{ep_idx:06d}.mp4",
    ]:
        p = os.path.join(real_root, pat)
        if os.path.exists(p):
            return p, 0.0
    # Chunk file — look up episode start timestamp from metadata
    chunk_file = os.path.join(
        real_root,
        "videos/observation.images.overhead/chunk-000/file-000.mp4"
    )
    if os.path.exists(chunk_file):
        try:
            import pandas as _pd
            ep_meta = _pd.read_parquet(
                os.path.join(real_root, "meta/episodes/chunk-000/file-000.parquet")
            )
            row = ep_meta[ep_meta["episode_index"] == ep_idx]
            if not row.empty:
                ts = float(row["videos/observation.images.overhead/from_timestamp"].iloc[0])
                return chunk_file, ts
        except Exception as e:
            print(f"[DETECT] Warning: could not read episode timestamps: {e}")
        return chunk_file, 0.0
    return None, 0.0


def _det_read_frames(vpath, offsets_s):
    """BGR frames via system ffmpeg (handles AV1 which cv2 cannot decode)."""
    import subprocess, numpy as np
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", vpath],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        print(f"[DETECT] ffprobe failed: {probe.stderr[:120]}")
        return []
    try:
        w, h = map(int, probe.stdout.strip().split(","))
    except Exception as e:
        print(f"[DETECT] ffprobe parse error: {e}  out={probe.stdout!r}")
        return []
    frames = []
    for t in offsets_s:
        cmd = ["ffmpeg", "-ss", str(t), "-i", vpath,
               "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        r = subprocess.run(cmd, capture_output=True)
        expected = w * h * 3
        if r.returncode == 0 and len(r.stdout) == expected:
            frame = np.frombuffer(r.stdout, dtype=np.uint8).reshape(h, w, 3).copy()
            frame[415:, 455:] = 0  # mask monitor reflection bottom-right corner
            frames.append(frame)
        else:
            print(f"[DETECT] ffmpeg@{t}s: ret={r.returncode} got={len(r.stdout)} expected={expected}")
    return frames


def detect_episode_objects(episode_idx, real_root):
    """
    Detect cylinder and target-cup world positions from the real overhead video.

    Samples the first ~3 s of the episode video (before the robot has moved far):
      • Cylinder → largest yellow blob on the LEFT  image half  (world Y < 0)
      • Target   → largest yellow blob on the RIGHT image half  (world Y > 0)
    _DET_EXCL masks out the robot body so arm pixels are ignored.

    Returns {"cylinder": (x,y,z), "target": (x,y,z)}
    or None if detection fails → caller falls back to EPISODE_POSITIONS table.
    """
    try:
        import cv2, numpy as np  # noqa: F401
    except ImportError:
        print("[DETECT] cv2 not available — using table positions")
        return None

    vpath, vid_offset = _det_get_video(real_root, episode_idx)
    if vpath is None:
        print(f"[DETECT] No video found for episode {episode_idx}")
        return None

    sample_times = [vid_offset + t for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]]
    print(f"[DETECT] Ep{episode_idx}: video offset={vid_offset:.1f}s")
    frames = _det_read_frames(vpath, sample_times)
    if not frames:
        print(f"[DETECT] No frames read from {vpath}")
        return None

    import numpy as np
    import math as _math
    _BASE_X, _BASE_Y = -0.03, 0.0   # robot base world position
    _MIN_DIST = 0.12                  # blobs < 12 cm from base are arm servos, not objects

    def _near_base(wx, wy):
        return _math.sqrt((wx - _BASE_X)**2 + (wy - _BASE_Y)**2) < _MIN_DIST

    cyl_pts, tgt_pts = [], []
    for f in frames:
        c = _det_find_blob(f, "left", hsv_lo=_DET_HSV_LO_CYL, hsv_hi=_DET_HSV_HI_CYL)
        t = _det_find_blob(f, "right")
        if c is not None and not _det_in_excl(*c) and not _near_base(*c):
            cyl_pts.append(c)
        if t is not None and not _det_in_excl(*t) and not _near_base(*t):
            tgt_pts.append(t)

    if not cyl_pts:
        print(f"[DETECT] Ep{episode_idx}: cylinder not detected — falling back to table")
        return None
    if not tgt_pts:
        print(f"[DETECT] Ep{episode_idx}: target not detected — falling back to table")
        return None

    cyl_x = float(np.median([p[0] for p in cyl_pts]))
    cyl_y = float(np.median([p[1] for p in cyl_pts]))
    tgt_x = float(np.median([p[0] for p in tgt_pts]))
    tgt_y = float(np.median([p[1] for p in tgt_pts]))

    print(f"[DETECT] Ep{episode_idx}: "
          f"cyl=({cyl_x:.3f}, {cyl_y:.3f}) [{len(cyl_pts)}/{len(frames)} frames]  "
          f"tgt=({tgt_x:.3f}, {tgt_y:.3f}) [{len(tgt_pts)}/{len(frames)} frames]")
    return {
        "cylinder": (cyl_x, cyl_y, 0.050),
        "target":   (tgt_x, tgt_y, 0.045),
    }
# ── End detection pipeline ──────────────────────────────────────────────────


def set_object_positions(env, episode_idx, real_root=None):
    """Override cylinder & target world positions after env.reset().
    Runs dynamic video detection first; falls back to EPISODE_POSITIONS table."""
    pos = None
    if real_root is not None:
        pos = detect_episode_objects(episode_idx, real_root)
    if pos is None:
        pos = EPISODE_POSITIONS.get(episode_idx)
        if pos is not None:
            print(f"[INFO]: Fallback: using table positions for episode {episode_idx}.")
    if pos is None:
        print(f"[INFO]: No positions for episode {episode_idx}, using randomised placement.")
        return
    import torch
    device = env.unwrapped.device
    for obj_name, xyz in (("cylinder", pos["cylinder"]), ("target", pos["target"])):
        obj   = env.unwrapped.scene[obj_name]
        state = obj.data.root_state_w.clone()
        state[:, :3]  = torch.tensor([xyz], dtype=torch.float32, device=device)
        state[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        state[:, 7:]  = 0.0
        obj.write_root_state_to_sim(state)
    print(f"[INFO]: Object positions set for episode {episode_idx}.")

def main():
    # -- Load real dataset actions --
    real_root = os.path.expanduser(
        f"~/.cache/huggingface/lerobot/{args_cli.real_repo_id}"
    )
    df     = pd.read_parquet(f"{real_root}/data/chunk-000/file-000.parquet")
    ep_df  = df[df["episode_index"] == args_cli.episode].sort_values("frame_index")
    frames = list(ep_df.iterrows())
    print(f"[INFO]: Loaded {len(frames)} frames for episode {args_cli.episode}")

    # -- Build Isaac Lab environment --
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=None, use_fabric=True
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[INFO]: Environment ready.")
    print()
    print("[INFO]: -- Connect your browser --")
    print("[INFO]:   http://localhost:8211/streaming/webrtc-demo/")
    print()

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
        port="dummy", id="replay",
        cameras=cameras, fps=30, kind="leader",
    )

    from omni.timeline import get_timeline_interface
    timeline = get_timeline_interface()

    frame_idx = 0
    env.reset()
    set_object_positions(env, args_cli.episode, real_root=real_root)

    # Pump frames so viewport is fully initialised before moving camera
    for _ in range(10):
        simulation_app.update()
    set_viewport_camera()
    simulation_app.update()

    print("[INFO]: -- Ready --")
    print("[INFO]: 1. Orbit viewport to match real camera angle.")
    print("[INFO]: 2. Press Play to replay the episode (auto-resets at end).")
    print("[INFO]: 3. When view looks right:")
    print("[INFO]:      Create > Camera > Camera From View")
    print("[INFO]:    Read Translate / Orient from the Property panel.")
    print("[INFO]: 4. Ctrl-C here to quit.")
    print()

    while simulation_app.is_running():
        if timeline.is_playing():
            if frame_idx < len(frames):
                _, row = frames[frame_idx]
                real_action = torch.tensor(
                    row["action"], dtype=torch.float32,
                    device=env.unwrapped.device,
                )
                sim_action = robot_iface.get_mapped_actions_vectorized(real_action)
                env.step(sim_action.unsqueeze(0))
                frame_idx += 1
                # Log EEF position every 20 frames to calibrate cylinder placement
                if frame_idx % 20 == 1:
                    try:
                        robot = env.unwrapped.scene["robot"]
                        names = list(robot.data.body_names)
                        pos_w = robot.data.body_state_w[0, :, :3]
                        eef_name = names[-1]
                        eef_pos  = pos_w[len(names)-1].cpu().tolist()
                        print(f"[EEF] frame={frame_idx:4d}  body={eef_name}  "
                              f"x={eef_pos[0]:.4f}  y={eef_pos[1]:.4f}  z={eef_pos[2]:.4f}",
                              flush=True)
                    except Exception as _e:
                        print(f"[EEF ERROR] frame={frame_idx} err={_e}", flush=True)
                if frame_idx == 5:
                    try:
                        import numpy as _np; from PIL import Image as _I
                        _rgb = env.unwrapped.scene["camera_overhead"].data.output["rgb"]
                        _f = _rgb[0].cpu().numpy().astype(_np.uint8)
                        _f[385:, 480:] = 0  # mask off-mat region (bucket appears here when position is off-table)
                        _I.fromarray(_f[..., :3] if _f.shape[-1] == 4 else _f).save(
                            f"/home/ted/so101-sim-project/sim_overhead_ep{args_cli.episode}_frame5.png")
                        print("[INFO]: Saved sim_overhead_frame.png")
                    except Exception as _e:
                        print(f"[WARNING]: frame save failed: {_e}")
            else:
                print("[INFO]: Episode complete -- exiting.")
                import sys; sys.exit(0)


        simulation_app.update()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
