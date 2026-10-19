# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""replay_calibrate.py — same as replay_interactive but draws calibration markers on the
SIM overhead frame.

At frame 5 it:
  1. Detects the cylinder in the real overhead video → real pixel (cx, cy)
  2. Converts that pixel (and ±10 px offsets above/below) to world coords via _H_REAL
  3. Projects those world coords to SIM pixel via world2pix_sim (_H_SIM)
  4. Draws markers on the SIM frame and saves:
       ~/so101-sim-project/replay_calibrate_sim_ep<N>_frame5.png   — SIM with markers
       ~/so101-sim-project/replay_calibrate_real_ep<N>_frame0.png  — REAL with cylinder dot

The markers let you see whether _H_SIM places things in the right spot in the SIM image.
"""

import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Calibration-marker overlay on SIM frame.")
parser.add_argument("--task",         type=str, default="Lerobot-So101-Teleop-Cylinder-To-Target")
parser.add_argument("--real_repo_id", type=str, default="Screener2/V4_pnp_44_20260720_141907")
parser.add_argument("--episode",      type=int, default=0)
AppLauncher.add_app_launcher_args(parser)

import sys  # noqa: E402
_ep = 0
_repo_id = None
for _i, _a in enumerate(sys.argv[1:], 1):
    if _a == '--episode' and _i < len(sys.argv):
        _ep = int(sys.argv[_i + 1]) if _i + 1 < len(sys.argv) else 0
    elif _a == '--real_repo_id' and _i < len(sys.argv):
        _repo_id = sys.argv[_i + 1] if _i + 1 < len(sys.argv) else None
_extra = ['--real_repo_id', _repo_id] if _repo_id is not None else []
args_cli = parser.parse_args([
    '--livestream', '2', '--episode', str(_ep),
    '--enable_cameras',
] + _extra)
args_cli.device = 'cpu'

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys as _sys
_clean = [_sys.argv[0]]
_it = iter(_sys.argv[1:])
for _tok in _it:
    if _tok == '--livestream':
        next(_it, None)
    elif _tok.startswith('--livestream='):
        pass
    elif _tok == '--enable_cameras':
        pass
    else:
        _clean.append(_tok)
_sys.argv = _clean

print('[CHECKPOINT] 1: AppLauncher done', flush=True)

import torch
import gymnasium as gym
import pandas as pd
import isaaclab_tasks                           # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks                  # noqa: F401
from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface
from apply_graphpaper_texture import apply_graphpaper_texture
print('[CHECKPOINT] 2: imports done', flush=True)


# ── set_viewport_camera — unchanged from replay_interactive ──────────────────

def set_viewport_camera():
    import math
    import omni.kit.viewport.utility as vp_util
    try:
        from sim_to_real_so101.tasks.cylinder_to_target_env_cfg import (
            CAMERA_X              as CAM_X,
            CAMERA_Y              as CAM_Y,
            CAMERA_HEIGHT_ABOVE_TABLE as CAM_Z,
            CAMERA_APERTURE_MM    as H_APERTURE,
            overhead_camera       as _cam_cfg,
            TILT_X_DEG            as _TX,
            TILT_Y_DEG            as _TY,
        )
        FOCAL_MM = float(_cam_cfg.spawn.focal_length)
        _rot = _cam_cfg.offset.rot
        print(f"[camera] CAMERA_X={CAM_X}  CAMERA_Y={CAM_Y}  CAMERA_HEIGHT_ABOVE_TABLE={CAM_Z}")
        print(f"[camera] TILT_X_DEG={_TX}  TILT_Y_DEG={_TY}")
        print(f"[camera] env cfg rot array (x,y,z,w from scipy): {[round(float(v),6) for v in _rot]}")
        qx, qy, qz, qw = float(_rot[0]), float(_rot[1]), float(_rot[2]), float(_rot[3])
        print(f"[camera] applying rot Gf.Quatd(w={qw:.5f}, x={qx:.5f}, y={qy:.5f}, z={qz:.5f})")
    except Exception as _re:
        print(f"[camera] WARNING: could not read camera cfg from env cfg ({_re}); using hardcoded fallback")
        CAM_X, CAM_Y, CAM_Z = -0.021, 0.211, 0.61
        FOCAL_MM   = 18.14756
        H_APERTURE = 28.0
        qw, qx, qy, qz = 0.0, 0.0, 0.0, 1.0

    vp = vp_util.get_active_viewport()
    cam_path = "/World/envs/env_0/CameraOverhead"
    from pxr import UsdGeom, Gf, Sdf
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    for _old in ["/World/envs/env_0/DebugSideCamera", "/World/envs/env_0/DebugPerspCamera"]:
        if stage.GetPrimAtPath(_old).IsValid():
            stage.RemovePrim(Sdf.Path(_old))
            print(f"[camera] removed old prim {_old}")

    _diag_pre = stage.GetPrimAtPath(cam_path)
    if _diag_pre.IsValid():
        _diag_ops = {op.GetOpName(): op for op in UsdGeom.Xformable(_diag_pre).GetOrderedXformOps()}
        print(f"[camera_diag] xform ops on prim: {list(_diag_ops.keys())}")
        for _rn in ("xformOp:orient", "xformOp:orientation", "xformOp:rotateXYZ"):
            if _rn in _diag_ops:
                print(f"[camera_diag] {_rn} BEFORE override: {_diag_ops[_rn].Get()}")

    cam_prim = stage.GetPrimAtPath(cam_path)
    if not cam_prim.IsValid():
        print(f"[camera] WARNING: {cam_path} not found, skipping")
        return
    _cam_xform = UsdGeom.Xformable(cam_prim)
    _ops = {op.GetOpName(): op for op in _cam_xform.GetOrderedXformOps()}
    print(f"[camera] prim xform ops found: {list(_ops.keys())}")
    for _t_name in ("xformOp:translate", "xformOp:translation"):
        if _t_name in _ops:
            _ops[_t_name].Set(Gf.Vec3d(CAM_X, CAM_Y, CAM_Z))
            print(f"[camera] set {_t_name} → ({CAM_X}, {CAM_Y}, {CAM_Z})")
            break
    for _r_name in ("xformOp:orient", "xformOp:orientation"):
        if _r_name in _ops:
            _ops[_r_name].Set(Gf.Quatd(qw, qx, qy, qz))
            print(f"[camera] set {_r_name} → (w={qw}, x={qx}, y={qy}, z={qz})")
            break
    if "xformOp:rotateXYZ" in _ops:
        _ops["xformOp:rotateXYZ"].Set(Gf.Vec3f(0.0, 0.0, 180.0))
        print("[camera] set xformOp:rotateXYZ → (0, 0, 180)")

    print(f"[camera] overhead → pos=({CAM_X},{CAM_Y},{CAM_Z}) focal={FOCAL_MM}mm aperture={H_APERTURE}mm")

    vp.camera_path = "/World/envs/env_0/CameraOverhead"
    print("[camera] viewport → CameraOverhead")

    try:
        vp.set_texture_resolution((640, 480))
        print("[camera] viewport texture resolution → 640×480")
    except Exception as _e:
        print(f"[camera] set_texture_resolution failed: {_e}")

    _PERSP_EYE    = Gf.Vec3d(-0.0345, 0.0204, 0.75094)
    _PERSP_TARGET = Gf.Vec3d(-0.0345, 0.20,   0.035)
    try:
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        _vcs = ViewportCameraState("/OmniverseKit_Persp")
        _vcs.set_position_world(_PERSP_EYE,    True)
        _vcs.set_target_world  (_PERSP_TARGET, True)
        print(f"[persp] ViewportCameraState → eye={tuple(_PERSP_EYE)}  target={tuple(_PERSP_TARGET)}")
    except Exception as _e:
        print(f"[persp] ViewportCameraState failed ({_e}); trying USD xform fallback")
        try:
            _PERSP_ROT_XYZ = (11.78683, 0.0, -1.29801)
            persp_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
            if persp_prim.IsValid():
                _p_ops = {op.GetOpName(): op
                          for op in UsdGeom.Xformable(persp_prim).GetOrderedXformOps()}
                for _t in ("xformOp:translate", "xformOp:translation"):
                    if _t in _p_ops:
                        _p_ops[_t].Set(_PERSP_EYE)
                        print(f"[persp] fallback set {_t}")
                        break
                if "xformOp:rotateXYZ" in _p_ops:
                    _p_ops["xformOp:rotateXYZ"].Set(Gf.Vec3f(*_PERSP_ROT_XYZ))
                    print(f"[persp] fallback set rotateXYZ → {_PERSP_ROT_XYZ}")
            else:
                print("[persp] /OmniverseKit_Persp not found in stage")
        except Exception as _e2:
            print(f"[persp] USD fallback also failed: {_e2}")


# ── Isaac 3D → SIM pixel projection ──────────────────────────────────────────

def project_isaac_3d_to_sim_pixel(xyz_isaac, env):
    """
    Project a point in Isaac world coordinates to SIM camera pixel (col, row).

    Uses the camera USD prim's ComputeLocalToWorldTransform (ground truth, no
    homography) plus the camera's intrinsic matrix from Isaac Lab.

    USD cameras look along -Z (OpenGL convention). We flip to ROS convention
    (Z-forward = into scene) by negating the USD Y and Z axes.

    Returns (col, row) as ints, or None on error.
    """
    import numpy as np

    try:
        from pxr import UsdGeom, Usd
        import omni.usd

        cam  = env.unwrapped.scene["camera_overhead"]
        K    = cam.data.intrinsic_matrices[0].cpu().numpy()   # (3,3)

        # ── Camera world transform from USD prim (most reliable source) ──────
        stage    = omni.usd.get_context().get_stage()
        cam_path = "/World/envs/env_0/CameraOverhead"
        cam_prim = stage.GetPrimAtPath(cam_path)
        if not cam_prim.IsValid():
            print(f"[3D_PROJ] prim {cam_path} not found")
            return None

        tc = Usd.TimeCode.Default()
        tf_gf = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(tc)
        # tf_gf[row][col] — row-major; columns 0-2 are world-space axes of cam frame
        M = np.array([[tf_gf[r][c] for c in range(4)] for r in range(4)])
        # USD uses ROW-vector convention: v_world = v_local × M
        # → translation is in the LAST ROW (M[3, :3]), NOT last column
        # → local axis i in world coords is ROW i of upper-left 3×3 (M[i, :3])
        pos_w = M[3, :3]          # camera position in Isaac world (metres)
        # Rows of M[:3,:3] = where cam local +X, +Y, +Z point in world frame
        # USD camera: +X=right, +Y=up-in-image, +Z=BACKWARD (OpenGL: looking = -Z)
        # ROS camera: +X=right, +Y=down-in-image, +Z=FORWARD (into scene)
        # Conversion: ROS_X =  USD_X  (row 0, same)
        #             ROS_Y = -USD_Y  (-row 1, flip up→down)
        #             ROS_Z = -USD_Z  (-row 2, flip backward→forward)
        X_ros = M[0, :3]           # right in image
        Y_ros = -M[1, :3]          # down in image
        Z_ros = -M[2, :3]          # into scene (forward)

        # World-to-ROS-camera rotation: rows are the ROS axes expressed in world
        R_w2c = np.stack([X_ros, Y_ros, Z_ros], axis=0)   # (3,3)

        print(f"[3D_PROJ] cam pos_world = {np.round(pos_w, 4)}")
        print(f"[3D_PROJ] Z_ros in world = {np.round(Z_ros, 4)}  "
              f"(should point toward scene)")
        print(f"[3D_PROJ] K diag = {K[0,0]:.1f} {K[1,1]:.1f}  "
              f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

        # ── Project ──────────────────────────────────────────────────────────
        P_w = np.array(xyz_isaac, dtype=np.float64)
        P_c = R_w2c @ (P_w - pos_w)    # point in ROS camera frame

        print(f"[3D_PROJ] P_world={np.round(P_w,4)}  "
              f"P_cam(ROS)={np.round(P_c,4)}")

        if P_c[2] <= 1e-4:
            print(f"[3D_PROJ] WARNING: point behind or at camera (P_c[2]={P_c[2]:.4f})")
            return None

        px  = K @ (P_c / P_c[2])
        col, row = float(px[0]), float(px[1])
        print(f"[3D_PROJ] → sim pixel col={col:.1f}  row={row:.1f}")
        return int(round(col)), int(round(row))

    except Exception as _e:
        import traceback
        print(f"[3D_PROJ] failed: {_e}")
        traceback.print_exc()
        return None


# ── Detection pipeline ────────────────────────────────────────────────────────

_DET_IMG_CX  = 320
_DET_IMG_CY  = 240

from homography_pix2world import pix2world_real as _det_pix2world, world2pix_sim as _world2pix_sim
print('[CHECKPOINT] 3: homography imported', flush=True)

_DET_EXCL       = dict(x_min=-0.350, x_max=-0.135, y_min=0.325, y_max=0.490)
_DET_HSV_LO     = (15,  50,  50)
_DET_HSV_HI     = (38, 255, 255)
_DET_HSV_LO_CYL = (35,  40,  40)
_DET_HSV_HI_CYL = (85, 255, 255)

def _det_in_excl(wx, wy):
    e = _DET_EXCL
    return e["x_min"] <= wx <= e["x_max"] and e["y_min"] <= wy <= e["y_max"]


def _det_find_blob(frame_bgr, col_side="any", hsv_lo=None, hsv_hi=None):
    """Returns ((world_X, world_Y), (cx_px, cy_px)) or (None, None)."""
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
        return None, None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return None, None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return _det_pix2world(cx, cy), (cx, cy)


def _det_get_video(real_root, ep_idx):
    for pat in [
        f"videos/observation.images.overhead/chunk-000/episode_{ep_idx:06d}.mp4",
        f"videos/observation.images.overhead/chunk-{ep_idx // 500:03d}/episode_{ep_idx:06d}.mp4",
    ]:
        p = os.path.join(real_root, pat)
        if os.path.exists(p):
            return p, 0.0
    chunk_file = os.path.join(
        real_root, "videos/observation.images.overhead/chunk-000/file-000.mp4"
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
            frame[415:, 455:] = 0
            frames.append(frame)
        else:
            print(f"[DETECT] ffmpeg@{t}s: ret={r.returncode} got={len(r.stdout)} expected={expected}")
    return frames


def detect_episode_objects(episode_idx, real_root):
    """
    Returns {"cylinder": (ix,iy,iz), "target": (ix,iy,iz), "cyl_real_px": (cx,cy)}
    or None on failure.
    """
    try:
        import cv2, numpy as np  # noqa: F401
    except ImportError:
        print("[DETECT] cv2 not available")
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
    _BASE_X, _BASE_Y = 0.043, 0.0
    _MIN_DIST = 0.12

    def _near_base(wx, wy):
        return _math.sqrt((wx - _BASE_X)**2 + (wy - _BASE_Y)**2) < _MIN_DIST

    cyl_pts, tgt_pts = [], []
    cyl_px_pts = []
    for f in frames:
        (c_world, c_px) = _det_find_blob(f, "left", hsv_lo=_DET_HSV_LO_CYL, hsv_hi=_DET_HSV_HI_CYL)
        (t_world, _   ) = _det_find_blob(f, "right")
        if c_world is not None and not _near_base(*c_world):
            cyl_pts.append(c_world)
            cyl_px_pts.append(c_px)
        if t_world is not None and not _det_in_excl(*t_world) and not _near_base(*t_world):
            tgt_pts.append(t_world)

    if not cyl_pts:
        print(f"[DETECT] Ep{episode_idx}: cylinder not detected")
        return None
    if not tgt_pts:
        print(f"[DETECT] Ep{episode_idx}: target not detected")
        return None

    cyl_x = float(np.median([p[0] for p in cyl_pts]))
    cyl_y = float(np.median([p[1] for p in cyl_pts]))
    tgt_x = float(np.median([p[0] for p in tgt_pts]))
    tgt_y = float(np.median([p[1] for p in tgt_pts]))

    # Median real pixel position of the cylinder
    cyl_px_col = int(np.median([p[0] for p in cyl_px_pts]))
    cyl_px_row = int(np.median([p[1] for p in cyl_px_pts]))

    print(f"[DETECT] Ep{episode_idx}: "
          f"cyl=({cyl_x:.3f}, {cyl_y:.3f}) real_px=({cyl_px_col},{cyl_px_row}) "
          f"[{len(cyl_pts)}/{len(frames)} frames]  "
          f"tgt=({tgt_x:.3f}, {tgt_y:.3f}) [{len(tgt_pts)}/{len(frames)} frames]")

    _PARALLAX_CAM_X = +0.021
    _PARALLAX_CAM_Y =  0.211
    _PARALLAX_CAM_H =  0.610
    _CYL_TOP_H      =  0.060
    _cyl_f = (_PARALLAX_CAM_H - _CYL_TOP_H) / _PARALLAX_CAM_H
    print(f"[DETECT] Parallax-corrected cyl (factor={_cyl_f:.4f}): "
          f"cyl=({cyl_x:.3f}, {cyl_y:.3f})  tgt unchanged=({tgt_x:.3f}, {tgt_y:.3f})")

    _X_CYL_CORRECTION = 0.00
    _X_TGT_CORRECTION = -0.01
    _Y_CYL_CORRECTION = 0.01
    _Y_TGT_CORRECTION = -0.01
    return {
        "cylinder":    (-cyl_x - _X_CYL_CORRECTION, cyl_y - _Y_CYL_CORRECTION, 0.050),
        "target":      (-tgt_x - _X_TGT_CORRECTION, tgt_y - _Y_TGT_CORRECTION, 0.060),
        "cyl_real_px": (cyl_px_col, cyl_px_row),   # ← NEW: real pixel coords
    }


def set_object_positions(env, episode_idx, real_root=None):
    pos = None
    if real_root is not None:
        pos = detect_episode_objects(episode_idx, real_root)
    if pos is None:
        print(f"[INFO]: Detection failed for episode {episode_idx} -- using randomised placement.")
        return None
    print(f"[PLACEMENT] cylinder Isaac XYZ = {pos['cylinder']}")
    print(f"[PLACEMENT] target   Isaac XYZ = {pos['target']}")
    import torch
    device = env.unwrapped.device
    for obj_name, xyz in (("cylinder", pos["cylinder"]), ("target", pos["target"])):
        obj   = env.unwrapped.scene[obj_name]
        state = obj.data.root_state_w.clone()
        state[:, :3]  = torch.tensor([xyz], dtype=torch.float32, device=device)
        state[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        state[:, 7:]  = 0.0
        obj.write_root_state_to_sim(state)
        print(f"[PLACEMENT] {obj_name} written to Isaac XYZ = {[round(v,4) for v in xyz]}")
    print(f"[INFO]: Object positions set for episode {episode_idx}.")
    return pos   # return so caller can use cyl_real_px


# ── Calibration marker drawing ────────────────────────────────────────────────

def draw_calibration_markers(sim_frame_rgb, ep,
                              cyl_sim_px_3d=None,
                              cyl_real_px=None):
    """
    Draw two markers on the SIM frame:

    MAGENTA — where the cylinder IS in SIM (from root_state_w + 3D cam projection).
              This is ground truth: if the world coords from _H_REAL are right, this
              is exactly where the cylinder should appear.

    WHITE   — the raw REAL detection pixel (cx, cy) stamped onto the SIM frame.
              If the two cameras were perfectly co-located, WHITE == MAGENTA.
              The gap between WHITE and MAGENTA = camera position error between
              the real and SIM overhead cameras.

    The line between them quantifies and visualises the misalignment.
    Returns annotated BGR image.
    """
    import cv2
    import numpy as np

    out = cv2.cvtColor(sim_frame_rgb, cv2.COLOR_RGB2BGR)
    font   = cv2.FONT_HERSHEY_SIMPLEX
    FONT_S = 0.42
    magenta = (200, 0, 255)   # BGR
    white   = (255, 255, 255)

    # ── MAGENTA: where cylinder IS in SIM (3D projection ground truth) ────────
    if cyl_sim_px_3d is not None:
        mc, mr = cyl_sim_px_3d
        cv2.drawMarker(out, (mc, mr), magenta, cv2.MARKER_CROSS, 36, 3, cv2.LINE_AA)
        cv2.circle(out, (mc, mr), 10, magenta, 1, cv2.LINE_AA)
        text = f"SIM 3D ({mc},{mr})"
        cv2.putText(out, text, (max(mc+14,2), mr-12), font, FONT_S, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(out, text, (max(mc+14,2), mr-12), font, FONT_S, magenta,  1, cv2.LINE_AA)
        print(f"[MARKER] MAGENTA(SIM 3D)  sim_px=({mc},{mr})")

    # ── WHITE: raw REAL detection pixel stamped on SIM frame ─────────────────
    if cyl_real_px is not None:
        rc, rr = cyl_real_px
        cv2.drawMarker(out, (rc, rr), white, cv2.MARKER_TILTED_CROSS, 30, 2, cv2.LINE_AA)
        cv2.circle(out, (rc, rr), 8, white, 1, cv2.LINE_AA)
        text2 = f"REAL det ({rc},{rr})"
        cv2.putText(out, text2, (max(rc+12,2), rr+4), font, FONT_S, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(out, text2, (max(rc+12,2), rr+4), font, FONT_S, white,   1, cv2.LINE_AA)
        print(f"[MARKER] WHITE(REAL det)  real_px=({rc},{rr})")

        # Arrow + distance from REAL pixel to SIM pixel — quantifies camera offset
        if cyl_sim_px_3d is not None:
            mc, mr = cyl_sim_px_3d
            err = ((mc-rc)**2 + (mr-rr)**2)**0.5
            cv2.arrowedLine(out, (rc, rr), (mc, mr), (180, 180, 180), 1,
                            cv2.LINE_AA, tipLength=0.15)
            mid = (int(round((rc+mc)/2)), int(round((rr+mr)/2)))
            err_txt = f"{err:.1f}px"
            cv2.putText(out, err_txt, (mid[0]+4, mid[1]-4), font, FONT_S,
                        (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(out, err_txt, (mid[0]+4, mid[1]-4), font, FONT_S,
                        (200,200,200), 1, cv2.LINE_AA)
            print(f"[MARKER] camera offset REAL→SIM = {err:.1f}px  "
                  f"(REAL({rc},{rr}) → SIM({mc},{mr}))")

    # Title
    title = (f"Ep {ep}  MAGENTA=SIM 3D position  "
             f"WHITE=REAL det pixel  ARROW=camera offset")
    cv2.putText(out, title, (4, 14), font, 0.35, (0, 0, 0),     2, cv2.LINE_AA)
    cv2.putText(out, title, (4, 14), font, 0.35, (255,255,255), 1, cv2.LINE_AA)

    return out


def annotate_real_frame(real_frame_bgr, cyl_real_px):
    """Draw a circle on the REAL frame at the detected cylinder pixel."""
    import cv2
    out = real_frame_bgr.copy()
    cx, cy = cyl_real_px
    cv2.circle(out, (cx, cy), 12, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(out, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    cv2.putText(out, f"cyl ({cx},{cy})", (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),   2, cv2.LINE_AA)
    cv2.putText(out, f"cyl ({cx},{cy})", (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    real_root = os.path.expanduser(
        f"~/.cache/huggingface/lerobot/{args_cli.real_repo_id}"
    )
    print('[CHECKPOINT] 4: entering main(), reading parquet', flush=True)
    df     = pd.read_parquet(f"{real_root}/data/chunk-000/file-000.parquet")
    ep_df  = df[df["episode_index"] == args_cli.episode].sort_values("frame_index")
    frames = list(ep_df.iterrows())
    print(f"[INFO]: Loaded {len(frames)} frames for episode {args_cli.episode}")

    # ── Save annotated REAL overhead frame at episode time 0 ─────────────────
    _cyl_real_px = None
    try:
        import numpy as _np_real
        _rvpath, _roffset = _det_get_video(real_root, args_cli.episode)
        if _rvpath is not None:
            _rframes = _det_read_frames(_rvpath, [_roffset])
            if _rframes:
                _rframe_bgr = _rframes[0]
                # Detect cylinder pixel in this first frame
                import cv2 as _cv2r
                (_c_world, _c_px) = _det_find_blob(
                    _rframe_bgr, "left",
                    hsv_lo=_DET_HSV_LO_CYL, hsv_hi=_DET_HSV_HI_CYL
                )
                if _c_px is not None:
                    _cyl_real_px = _c_px
                    _annotated_real = annotate_real_frame(_rframe_bgr, _c_px)
                else:
                    _annotated_real = _rframe_bgr
                _rout = (f"/home/ted/so101-sim-project/"
                         f"replay_calibrate_real_ep{args_cli.episode}_frame0.png")
                _cv2r.imwrite(_rout, _annotated_real)
                print(f"[INFO]: Saved real frame → {_rout}")
                if _c_px:
                    print(f"[INFO]: Cylinder detected at real pixel ({_c_px[0]}, {_c_px[1]})")
    except Exception as _e:
        print(f"[WARNING]: real frame save failed: {_e}")

    # ── Build Isaac Lab environment ───────────────────────────────────────────
    print('[CHECKPOINT] 5: calling parse_env_cfg', flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=None, use_fabric=True
    )
    print('[CHECKPOINT] 6: calling gym.make', flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[CHECKPOINT] 7: gym.make returned", flush=True)

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
    print('[CHECKPOINT] 8: calling env.reset()', flush=True)
    env.reset()

    for _ in range(10):
        simulation_app.update()
    set_viewport_camera()
    simulation_app.update()

    import omni.usd as _omni_usd_tex
    _tex_stage = _omni_usd_tex.get_context().get_stage()
    apply_graphpaper_texture(
        _tex_stage,
        texture_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphpaper_mat.png"),
    )
    simulation_app.update()

    # Detect object positions (also grabs cyl_real_px from video)
    _det_pos = detect_episode_objects(args_cli.episode, real_root=real_root)
    if _det_pos is not None and _cyl_real_px is None:
        _cyl_real_px = _det_pos.get("cyl_real_px")

    print("[INFO]: Ready — timeline will auto-play for frame capture.")
    print("[INFO]: Ctrl-C to quit early.")

    while simulation_app.is_running():
        if timeline.is_playing():
            if frame_idx < len(frames):

                if frame_idx == 0:
                    # Place cylinder & target at detected real positions
                    if _det_pos is not None:
                        for obj_name, xyz in (
                            ("cylinder", _det_pos["cylinder"]),
                            ("target",   _det_pos["target"]),
                        ):
                            obj   = env.unwrapped.scene[obj_name]
                            state = obj.data.root_state_w.clone()
                            state[:, :3]  = torch.tensor([xyz], dtype=torch.float32,
                                                          device=env.unwrapped.device)
                            state[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]],
                                                          device=env.unwrapped.device)
                            state[:, 7:]  = 0.0
                            obj.write_root_state_to_sim(state)
                            print(f"[PLACEMENT] {obj_name} → {[round(v,4) for v in xyz]}")

                _, row = frames[frame_idx]
                real_action = torch.tensor(
                    row["action"], dtype=torch.float32,
                    device=env.unwrapped.device,
                )
                sim_action = robot_iface.get_mapped_actions_vectorized(real_action)
                env.step(sim_action.unsqueeze(0))
                frame_idx += 1

                # ── At frame 5: capture SIM frame, draw markers, save ─────────
                if frame_idx == 5:
                    try:
                        import numpy as _np
                        import cv2 as _cv2

                        _rgb = env.unwrapped.scene["camera_overhead"].data.output["rgb"]
                        _f   = _rgb[0].cpu().numpy().astype(_np.uint8)
                        _sim_rgb = _f[..., :3] if _f.shape[-1] == 4 else _f

                        _out_dir = "/home/ted/so101-sim-project"
                        _sim_raw_path = (f"{_out_dir}/"
                                         f"replay_calibrate_sim_ep{args_cli.episode}_frame5_raw.png")
                        _cv2.imwrite(_sim_raw_path, _cv2.cvtColor(_sim_rgb, _cv2.COLOR_RGB2BGR))
                        print(f"[INFO]: Saved raw SIM frame → {_sim_raw_path}")

                        if _det_pos is not None:
                            # Ground-truth 3D projection: cylinder pos in Isaac
                            # world → camera K/R/t → SIM pixel
                            _cyl_obj = env.unwrapped.scene["cylinder"]
                            _cyl_pos_isaac = (
                                _cyl_obj.data.root_state_w[0, :3].cpu().tolist()
                            )
                            print(f"[3D_PROJ] cylinder root_state_w = "
                                  f"{[round(v,4) for v in _cyl_pos_isaac]}")
                            _cyl_sim_px_3d = project_isaac_3d_to_sim_pixel(
                                _cyl_pos_isaac, env
                            )
                            _marked = draw_calibration_markers(
                                _sim_rgb,
                                args_cli.episode,
                                cyl_sim_px_3d=_cyl_sim_px_3d,
                                cyl_real_px=_cyl_real_px,
                            )
                            _sim_out = (f"{_out_dir}/"
                                        f"replay_calibrate_sim_ep{args_cli.episode}_frame5.png")
                            _cv2.imwrite(_sim_out, _marked)
                            print(f"[INFO]: Saved annotated SIM frame → {_sim_out}")
                        else:
                            print("[WARNING]: No detection result — markers not drawn.")

                    except Exception as _e:
                        print(f"[WARNING]: SIM frame save failed: {_e}")

                # Log EEF every 20 frames
                if frame_idx % 20 == 1:
                    try:
                        robot = env.unwrapped.scene["robot"]
                        names = list(robot.data.body_names)
                        eef_pos = robot.data.body_state_w[0, len(names)-1, :3].cpu().tolist()
                        print(f"[EEF] frame={frame_idx:4d}  "
                              f"x={eef_pos[0]:.4f}  y={eef_pos[1]:.4f}  z={eef_pos[2]:.4f}",
                              flush=True)
                    except Exception as _e:
                        print(f"[EEF ERROR] frame={frame_idx} err={_e}", flush=True)

            else:
                print("[INFO]: Episode complete -- exiting.")
                import sys; sys.exit(0)

        simulation_app.update()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()