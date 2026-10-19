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


# EPISODE_POSITIONS table removed — positions detected live from video.

def set_viewport_camera():
    """Configure CameraOverhead for training data capture and switch the
    streaming viewport back to the default Perspective camera."""
    import math
    import omni.kit.viewport.utility as vp_util

    # Mat corners: X in [-0.41, 0.34], Y in [0.00, 0.41]
    # Camera centred directly above mat, looking straight down (no tilt).
    CAM_X      = -0.035   # mat centre X = (-0.41+0.34)/2
    CAM_Y      =  0.205   # mat centre Y = (0.00+0.41)/2
    CAM_Z      =  0.70    # height above mat (m)
    TILT_DEG   =  0.0     # straight down — no tilt
    FOCAL_MM   = 18.14756
    H_APERTURE = 20.995  # horizontal sensor mm
    V_APERTURE = 16.594  # vertical sensor mm (582x460 ratio)

    # Orientation: Rz(180°) only — image-right=Isaac+X, image-down=Isaac+Y.
    # No tilt quaternion needed; q = (w=0, x=0, y=0, z=1).
    qw, qx, qy, qz = 0.0, 0.0, 0.0, 1.0

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
    # Override camera prim position/orientation so it matches our target config.
    # TiledCameraCfg sets initial values; we force them here after initialisation.
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
            # Gf.Quatd(w, x, y, z) — Rz(180°): w=0, z=1
            _ops[_r_name].Set(Gf.Quatd(qw, qx, qy, qz))
            print(f"[camera] set {_r_name} → (w={qw}, x={qx}, y={qy}, z={qz})")
            break
    # Fallback: rotateXYZ op (Euler degrees)
    if "xformOp:rotateXYZ" in _ops:
        _ops["xformOp:rotateXYZ"].Set(Gf.Vec3f(0.0, 0.0, 180.0))
        print("[camera] set xformOp:rotateXYZ → (0, 0, 180)")

    print(f"[camera] overhead → pos=({CAM_X},{CAM_Y},{CAM_Z}) tilt={TILT_DEG}deg "
          f"focal={FOCAL_MM}mm h={H_APERTURE}mm v={V_APERTURE}mm")

    # Switch streaming viewport to overhead camera.
    vp.camera_path = "/World/envs/env_0/CameraOverhead"
    print("[camera] viewport → Perspective")

    PERSP_EYE    = (-0.03, 0.21, 2.0)   # TOP-DOWN: directly above mat centre
    PERSP_TARGET = (-0.03, 0.21, 0.00)  # look straight down

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
        persp_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
        if persp_prim.IsValid():
            _p_ops = {op.GetOpName(): op for op in UsdGeom.Xformable(persp_prim).GetOrderedXformOps()}
            print(f"[persp] ops on /OmniverseKit_Persp: {list(_p_ops.keys())}")
            for _n, _op in _p_ops.items():
                print(f"[persp]   {_n} = {_op.Get()}")
        else:
            print("[persp] /OmniverseKit_Persp prim not found in stage")


# ── Dynamic detection from real overhead video ─────────────────────────────
_DET_IMG_CX  = 320   # image centre col — used for left/right blob split
_DET_IMG_CY  = 240   # image centre row

# Homography-based pixel → world (replaces linear formula)
from homography_pix2world import pix2world_real as _det_pix2world

# Exclusion zone: robot body / arm (bottom-right of real image)
_DET_EXCL   = dict(x_min=-0.400, x_max=-0.185, y_min=0.325, y_max=0.490)

# HSV detection thresholds
_DET_HSV_LO     = (15,  50,  50)   # yellow/orange cup
_DET_HSV_HI     = (38, 255, 255)
_DET_HSV_LO_CYL = (35,  40,  40)   # green cylinder
_DET_HSV_HI_CYL = (85, 255, 255)

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
    for pat in [
        f"videos/observation.images.overhead/chunk-000/episode_{ep_idx:06d}.mp4",
        f"videos/observation.images.overhead/chunk-{ep_idx // 500:03d}/episode_{ep_idx:06d}.mp4",
    ]:
        p = os.path.join(real_root, pat)
        if os.path.exists(p):
            return p, 0.0
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

    Returns {"cylinder": (x,y,z), "target": (x,y,z)} or None on failure.
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
    _BASE_X, _BASE_Y = 0.03, 0.0
    _MIN_DIST = 0.12

    def _near_base(wx, wy):
        return _math.sqrt((wx - _BASE_X)**2 + (wy - _BASE_Y)**2) < _MIN_DIST

    cyl_pts, tgt_pts = [], []
    for f in frames:
        c = _det_find_blob(f, "left", hsv_lo=_DET_HSV_LO_CYL, hsv_hi=_DET_HSV_HI_CYL)
        t = _det_find_blob(f, "right")
        if c is not None and not _near_base(*c):
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

    # ── Parallax correction ────────────────────────────────────────────────────
    # pix2world_real is calibrated at z=0 (real mat surface). The camera sees
    # the TOP face of the cylinder/target (h_top metres above mat), so the
    # returned XY is displaced AWAY from the camera nadir relative to the true
    # base position.  Correct by scaling the distance from nadir by
    # (H - h_top) / H, which pulls the detected position back toward centre.
    #
    # Derivation: ray from camera (nadir, H) through top-face pixel hits z=0
    # at X_det = nadir + (X_base - nadir)*H/(H-h_top)
    # → X_base = nadir + (X_det - nadir)*(H-h_top)/H
    #
    # _PARALLAX_CAM_H: real camera height above mat (m).
    # Value from homography_pix2world.py calibration: "Camera at Z=0.600m".
    _PARALLAX_CAM_X = -0.035   # camera nadir X in real-world coords
    _PARALLAX_CAM_Y =  0.205   # camera nadir Y in real-world coords
    _PARALLAX_CAM_H =  0.600   # real camera height above mat surface (m)
                               # confirmed in homography_pix2world.py: "Camera at Z=0.600m"
    _CYL_TOP_H      =  0.060   # cylinder height above mat surface (m)
    _TGT_TOP_H      =  0.060   # target/cup height above mat surface (m)
    _cyl_f = (_PARALLAX_CAM_H - _CYL_TOP_H) / _PARALLAX_CAM_H   # = 0.900
    _tgt_f = (_PARALLAX_CAM_H - _TGT_TOP_H) / _PARALLAX_CAM_H
    cyl_x = _PARALLAX_CAM_X + (cyl_x - _PARALLAX_CAM_X) * _cyl_f
    cyl_y = _PARALLAX_CAM_Y + (cyl_y - _PARALLAX_CAM_Y) * _cyl_f
    tgt_x = _PARALLAX_CAM_X + (tgt_x - _PARALLAX_CAM_X) * _tgt_f
    tgt_y = _PARALLAX_CAM_Y + (tgt_y - _PARALLAX_CAM_Y) * _tgt_f
    print(f"[DETECT] Parallax-corrected (factor={_cyl_f:.4f}): "
          f"cyl=({cyl_x:.3f}, {cyl_y:.3f})  tgt=({tgt_x:.3f}, {tgt_y:.3f})")
    # ─────────────────────────────────────────────────────────────────────────

    # X correction: objects appear ~4 cm left of H_REAL commanded position in SIM.
    # Shift Isaac_X by -0.04 m to compensate (moves object rightward in SIM).
    _X_CORRECTION = 0.04
    return {
        "cylinder": (-cyl_x - _X_CORRECTION, cyl_y, 0.050),
        "target":   (-tgt_x - _X_CORRECTION, tgt_y, 0.060),
    }
# ── End detection pipeline ──────────────────────────────────────────────────


def set_object_positions(env, episode_idx, real_root=None):
    """Override cylinder & target world positions.
    Must be called inside the simulation loop at frame_idx == 0,
    immediately before the first env.step(), so that write_root_state_to_sim
    is flushed to PhysX by that step rather than overwritten by pump frames.
    """
    pos = None
    if real_root is not None:
        pos = detect_episode_objects(episode_idx, real_root)
    if pos is None:
        print(f"[INFO]: Detection failed for episode {episode_idx} -- using randomised placement.")
        return
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

    # ── Pump frames so viewport is fully initialised before moving camera ─────
    # NOTE: set_object_positions is NOT called here. It is called inside the
    # loop at frame_idx == 0, immediately before the first env.step(), so
    # that the write survives into the PhysX simulation rather than being
    # overwritten by the physics ticks inside these pump frames.
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

                # ── Set object positions at frame 0, before the first step ────
                # Called here (not before pump frames) so write_root_state_to_sim
                # is flushed to PhysX by env.step() rather than overwritten.
                if frame_idx == 0:
                    set_object_positions(env, args_cli.episode, real_root=real_root)
                # ─────────────────────────────────────────────────────────────

                _, row = frames[frame_idx]
                real_action = torch.tensor(
                    row["action"], dtype=torch.float32,
                    device=env.unwrapped.device,
                )
                sim_action = robot_iface.get_mapped_actions_vectorized(real_action)

                # ── Frame-0 ACTION diagnostic (one-shot) ─────────────────────
                if frame_idx == 0:
                    import numpy as _np_act
                    print(f"[ACTION_DIAG] ── frame 0 action diagnostic ─────────")
                    print(f"[ACTION_DIAG] real_action (raw dataset): {real_action.cpu().numpy().round(4).tolist()}")
                    print(f"[ACTION_DIAG] sim_action  (after map):   {sim_action.cpu().numpy().round(4).tolist()}")
                    _ratio = (sim_action / (real_action + 1e-9)).cpu().numpy()
                    print(f"[ACTION_DIAG] sim/real ratio per joint:  {_ratio.round(4).tolist()}")
                    _pi_180 = float(_np_act.pi / 180)
                    print(f"[ACTION_DIAG] pi/180 = {_pi_180:.6f}  -- if ratio ≈ this, function converts deg→rad")
                    print(f"[ACTION_DIAG] if ratio ≈ 1.0, function passes through unchanged")
                # ─────────────────────────────────────────────────────────────

                env.step(sim_action.unsqueeze(0))
                frame_idx += 1

                # ── Frame-0 joint diagnostic ──────────────────────────────────
                if frame_idx == 1:
                    try:
                        import numpy as _np_diag
                        _robot = env.unwrapped.scene["robot"]
                        _body_names = list(_robot.data.body_names)
                        _joint_names = list(_robot.data.joint_names)
                        _sim_joints  = _robot.data.joint_pos[0].cpu().numpy()
                        _real_state  = _np_diag.array(row["observation.state"], dtype=_np_diag.float32)
                        _real_joints = _real_state[:len(_sim_joints)]
                        _delta       = _sim_joints - _real_joints
                        print(f"[JOINT_CMP] ── frame 0 diagnostic ──────────────────")
                        print(f"[JOINT_CMP] joint_names : {_joint_names}")
                        print(f"[JOINT_CMP] real_state  : {_real_joints.round(3).tolist()}")
                        print(f"[JOINT_CMP] sim_joints  : {_sim_joints.round(3).tolist()}")
                        print(f"[JOINT_CMP] delta(sim-real): {_delta.round(3).tolist()}")
                        _eef = _robot.data.body_state_w[0, -1, :3].cpu().numpy()
                        print(f"[JOINT_CMP] EEF world pos: x={_eef[0]:.4f}  y={_eef[1]:.4f}  z={_eef[2]:.4f}")
                    except Exception as _e:
                        print(f"[JOINT_CMP] error: {_e}")
                # ─────────────────────────────────────────────────────────────

                # Log EEF position every 20 frames
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

                if frame_idx == 1 or frame_idx == 5:
                    try:
                        import numpy as _np; from PIL import Image as _I
                        _rgb = env.unwrapped.scene["camera_overhead"].data.output["rgb"]
                        _f = _rgb[0].cpu().numpy().astype(_np.uint8)
                        _frame_rgb = _f[..., :3] if _f.shape[-1] == 4 else _f
                        _I.fromarray(_frame_rgb).save(
                            f"/home/ted/so101-sim-project/sim_overhead_ep{args_cli.episode}_frame{frame_idx}.png")
                        print(f"[INFO]: Saved sim_overhead_ep{args_cli.episode}_frame{frame_idx}.png")
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