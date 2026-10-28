"""record_sim_episodes.py — Record Isaac Lab sim episodes as a LeRobot v3.0 dataset.

For each real episode in --real_repo_id, replays the arm movements in Isaac Lab,
captures every sim camera frame, and writes a new LeRobot v3.0 dataset under
~/.cache/huggingface/lerobot/{sim_out_id}/.

Video format matches the real dataset exactly:
  - ONE video file per chunk (chunk-000/file-000.mp4) with ALL episode frames
    concatenated in episode order inside that file.
  - meta/episodes parquet has a  videos/observation.images.overhead/from_timestamp
    column recording the absolute seek position (seconds) where each episode
    starts inside the chunk video file.  This is how LeRobot and replay_episodes.py
    locate a specific episode inside the shared video.

Usage:
    ./isaaclab.sh -p record_sim_episodes.py \\
        --real_repo_id Screener2/FullGrid_20261007_095459 \\
        --sim_out_id   Screener2/FullGrid_20261007_sim

    --episodes 0          test on a single episode
    --episodes 0,1,2      test on a few episodes
    (omit)                record all episodes

    --headless            suppress the viewport window
    --overwrite           delete and recreate the output dataset if it exists
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import warnings

from isaaclab.app import AppLauncher

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Record sim episodes as LeRobot dataset.")
parser.add_argument("--task",         type=str, default="Lerobot-So101-Teleop-Cylinder-To-Target")
parser.add_argument("--real_repo_id", type=str, default="Screener2/FullGrid_20261007_095459")
parser.add_argument("--sim_out_id",   type=str, default="Screener2/FullGrid_20261007_sim")
parser.add_argument("--episodes",     type=str, default="all",
                    help="Comma-separated episode indices or 'all'")
parser.add_argument("--overwrite",    action="store_true")
AppLauncher.add_app_launcher_args(parser)

# Parse user flags before AppLauncher consumes sys.argv.
_real_repo_id = "Screener2/FullGrid_20261007_095459"
_sim_out_id   = "Screener2/FullGrid_20261007_sim"
_episodes_arg = "all"
_overwrite    = False
for _i, _a in enumerate(sys.argv[1:], 1):
    if _a == "--real_repo_id" and _i + 1 < len(sys.argv):
        _real_repo_id = sys.argv[_i + 1]
    elif _a == "--sim_out_id" and _i + 1 < len(sys.argv):
        _sim_out_id = sys.argv[_i + 1]
    elif _a == "--episodes" and _i + 1 < len(sys.argv):
        _episodes_arg = sys.argv[_i + 1]
    elif _a == "--overwrite":
        _overwrite = True

args_cli = parser.parse_args([
    "--livestream", "2",
    "--enable_cameras",
    "--real_repo_id", _real_repo_id,
    "--sim_out_id",   _sim_out_id,
    "--episodes",     _episodes_arg,
    "--kit_args",
    "--ext-folder /home/ted/isaac-sim/extsDeprecated "
    "--enable isaacsim.core.utils "
    "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false",
])
args_cli.device    = "cpu"
args_cli.overwrite = _overwrite

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import gymnasium as gym
import numpy as np
import pandas as pd
import torch

import isaaclab_tasks                          # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks                 # noqa: F401
from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface

# ── Dataset constants ─────────────────────────────────────────────────────────
CHUNKS_SIZE  = 1000    # episodes per chunk (matches real dataset behaviour)
FPS          = 30
# Sim robot starts in an extended default pose; real episodes start contracted.
# Skip the first second of every sim episode (physics still steps, nothing recorded).
SKIP_FRAMES  = FPS     # = 30 frames = 1 s
IMG_H, IMG_W = 480, 640
VIDEO_KEY    = "observation.images.overhead"
TS_COL       = f"videos/{VIDEO_KEY}/from_timestamp"   # absolute seek position in chunk video
CHUNK_COL    = f"videos/{VIDEO_KEY}/chunk_index"      # which chunk file holds this episode
FILE_COL     = f"videos/{VIDEO_KEY}/file_index"       # file index within the chunk
JOINT_NAMES  = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos",   "wrist_roll.pos",    "gripper.pos",
]
TASK_INDEX   = 0


# ─────────────────────────────────────────────────────────────────────────────
# Parquet / meta helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_all_real_episodes(real_root: str) -> pd.DataFrame:
    pq_files = sorted(glob.glob(f"{real_root}/data/**/*.parquet", recursive=True))
    if not pq_files:
        raise RuntimeError(f"No parquet files found under {real_root}/data/")
    chunks = [pd.read_parquet(p) for p in pq_files]
    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    print(f"[DATA] Loaded {len(df)} rows from {len(pq_files)} parquet file(s)")
    return df


def write_data_parquet(rows: list[dict], chunk_dir: str, file_idx: int):
    os.makedirs(chunk_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in ["frame_index", "episode_index", "index", "task_index"]:
        df[col] = df[col].astype("int64")
    df["timestamp"] = df["timestamp"].astype("float32")
    path = os.path.join(chunk_dir, f"file-{file_idx:03d}.parquet")
    df.to_parquet(path, index=False)
    print(f"[PARQUET] {len(df)} rows → {path}")
    return path


def compute_ep_stats(ep_rows: list[dict]) -> dict:
    """Compute per-episode statistics for all features (matches real dataset columns)."""
    action = np.array([r["action"] for r in ep_rows], dtype=np.float32)          # (N, 6)
    state  = np.array([r["observation.state"] for r in ep_rows], dtype=np.float32)  # (N, 6)
    ts     = np.array([r["timestamp"] for r in ep_rows], dtype=np.float32).reshape(-1, 1)
    fi     = np.array([r["frame_index"] for r in ep_rows], dtype=np.float32).reshape(-1, 1)
    ei     = np.array([r["episode_index"] for r in ep_rows], dtype=np.float32).reshape(-1, 1)
    idx    = np.array([r["index"] for r in ep_rows], dtype=np.float32).reshape(-1, 1)
    ti     = np.array([r["task_index"] for r in ep_rows], dtype=np.float32).reshape(-1, 1)
    n      = len(ep_rows)

    def _s(arr):  # arr: (N, D) → dict of per-dim stat lists
        return {
            "min":   arr.min(axis=0).tolist(),
            "max":   arr.max(axis=0).tolist(),
            "mean":  arr.mean(axis=0).tolist(),
            "std":   arr.std(axis=0).clip(min=0.0).tolist(),
            "count": [int(n)] * arr.shape[1],
            "q01":   np.percentile(arr,  1, axis=0).tolist(),
            "q10":   np.percentile(arr, 10, axis=0).tolist(),
            "q50":   np.percentile(arr, 50, axis=0).tolist(),
            "q90":   np.percentile(arr, 90, axis=0).tolist(),
            "q99":   np.percentile(arr, 99, axis=0).tolist(),
        }

    # Image stats use ImageNet normalization constants (per channel)
    img = {
        "min":   [0.0, 0.0, 0.0],
        "max":   [1.0, 1.0, 1.0],
        "mean":  [0.485, 0.456, 0.406],
        "std":   [0.229, 0.224, 0.225],
        "count": [int(n), int(n), int(n)],
        "q01":   [0.0, 0.0, 0.0],
        "q10":   [0.0, 0.0, 0.0],
        "q50":   [0.485, 0.456, 0.406],
        "q90":   [1.0, 1.0, 1.0],
        "q99":   [1.0, 1.0, 1.0],
    }

    return {
        "action":                          _s(action),
        "observation.state":               _s(state),
        "observation.images.overhead":     img,
        "timestamp":                       _s(ts),
        "frame_index":                     _s(fi),
        "episode_index":                   _s(ei),
        "index":                           _s(idx),
        "task_index":                      _s(ti),
    }


def write_episodes_meta(episode_list: list[dict], out_root: str):
    """Write meta/episodes/chunk-N/file-N.parquet.

    All columns are already computed in episode_list; this function just
    groups them by chunk and serialises to parquet.
    """
    chunk_eps: dict[int, list[dict]] = {}
    for ep in episode_list:
        c = int(ep["episode_index"]) // CHUNKS_SIZE
        chunk_eps.setdefault(c, []).append(ep)

    for c_idx, eps in sorted(chunk_eps.items()):
        ep_dir = os.path.join(out_root, "meta", "episodes", f"chunk-{c_idx:03d}")
        os.makedirs(ep_dir, exist_ok=True)
        ep_df = pd.DataFrame(eps)
        ep_df = ep_df.sort_values("episode_index").reset_index(drop=True)

        # Cast scalar integer / float columns to the right dtype
        for col in ["episode_index", "length",
                    "dataset_from_index", "dataset_to_index",
                    "data/chunk_index", "data/file_index",
                    CHUNK_COL, FILE_COL,
                    "meta/episodes/chunk_index", "meta/episodes/file_index"]:
            if col in ep_df.columns:
                ep_df[col] = ep_df[col].astype("int64")
        for col in [TS_COL, f"videos/{VIDEO_KEY}/to_timestamp"]:
            if col in ep_df.columns:
                ep_df[col] = ep_df[col].astype("float64")

        path = os.path.join(ep_dir, f"file-{c_idx:03d}.parquet")
        ep_df.to_parquet(path, index=False)
        print(f"[META] episodes parquet → {path}")
        print(f"       columns ({len(ep_df.columns)}): {list(ep_df.columns)}")


def write_tasks_parquet(out_root: str, task: str = "Pick cylinder to target"):
    meta_dir = os.path.join(out_root, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    df = pd.DataFrame([{"task_index": 0, "task": task}])
    df["task_index"] = df["task_index"].astype("int64")
    path = os.path.join(meta_dir, "tasks.parquet")
    df.to_parquet(path, index=False)
    print(f"[META] tasks.parquet → {path}")


def write_stats_json(out_root: str, all_rows: list[dict]):
    """Compute and write meta/stats.json from recorded data."""
    import numpy as np
    action = np.array([r["action"] for r in all_rows], dtype=np.float32)
    state  = np.array([r["observation.state"] for r in all_rows], dtype=np.float32)
    stats = {
        "action": {
            "mean": action.mean(axis=0).tolist(),
            "std":  action.std(axis=0).clip(min=1e-6).tolist(),
            "min":  action.min(axis=0).tolist(),
            "max":  action.max(axis=0).tolist(),
        },
        "observation.state": {
            "mean": state.mean(axis=0).tolist(),
            "std":  state.std(axis=0).clip(min=1e-6).tolist(),
            "min":  state.min(axis=0).tolist(),
            "max":  state.max(axis=0).tolist(),
        },
        "observation.images.overhead": {
            # ImageNet normalisation (matches use_imagenet_stats=True in lerobot-train)
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225],
            "min":  [0.0, 0.0, 0.0],
            "max":  [1.0, 1.0, 1.0],
        },
    }
    path = os.path.join(out_root, "meta", "stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[META] stats.json → {path}")


def write_info_json(out_root: str, total_episodes: int, total_frames: int):
    info = {
        "codebase_version": "v3.0",
        "fps": FPS,
        "features": {
            "action":            {"dtype": "float32", "names": JOINT_NAMES, "shape": [6]},
            "observation.state": {"dtype": "float32", "names": JOINT_NAMES, "shape": [6]},
            VIDEO_KEY: {
                "dtype": "video",
                "names": ["height", "width", "channel"],
                "shape": [IMG_H, IMG_W, 3],
                "info": {
                    "video.codec":   "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.fps":     FPS,
                    "video.crf":     30,
                    "video.g":       2,
                    "video.preset":  12,
                },
            },
            "timestamp":     {"dtype": "float32", "names": None, "shape": [1]},
            "frame_index":   {"dtype": "int64",   "names": None, "shape": [1]},
            "episode_index": {"dtype": "int64",   "names": None, "shape": [1]},
            "index":         {"dtype": "int64",   "names": None, "shape": [1]},
            "task_index":    {"dtype": "int64",   "names": None, "shape": [1]},
        },
        "total_episodes": total_episodes,
        "total_frames":   total_frames,
        "total_tasks":    1,
        "chunks_size":    CHUNKS_SIZE,
        "data_path":      "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path":     f"videos/{VIDEO_KEY}/chunk-{{chunk_index:03d}}/file-{{file_index:03d}}.mp4",
        "robot_type":     "so_follower",
        "splits":         {"train": f"0:{total_episodes}"},
    }
    meta_dir = os.path.join(out_root, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    path = os.path.join(meta_dir, "info.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[META] info.json → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Video encoder
# One AV1VideoWriter per CHUNK — all episodes in the chunk are piped into the
# same ffmpeg process, producing one concatenated file (matching real dataset).
# ─────────────────────────────────────────────────────────────────────────────

class AV1VideoWriter:
    """Pipe numpy RGB frames into one ffmpeg AV1 process for an entire chunk."""

    def __init__(self, path: str, fps: int = 30, width: int = 640, height: int = 480):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v",     "libsvtav1",
            "-crf",     "30",
            "-g",       "2",
            "-preset",  "12",
            "-pix_fmt", "yuv420p",
            path,
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        self._path  = path
        self._n     = 0           # total frames written to this file
        self._fps   = fps
        # cumulative duration of episodes already written to this file
        self._chunk_duration_s = 0.0

    @property
    def current_offset_s(self) -> float:
        """Absolute timestamp (seconds) where the NEXT episode will start."""
        return self._chunk_duration_s

    def write(self, frame_rgb: np.ndarray):
        assert frame_rgb.dtype == np.uint8
        assert frame_rgb.shape == (480, 640, 3), f"Unexpected shape {frame_rgb.shape}"
        self._proc.stdin.write(frame_rgb.tobytes())
        self._n += 1

    def end_episode(self, n_frames: int):
        """Call after finishing one episode so offset tracking stays correct."""
        self._chunk_duration_s += n_frames / self._fps

    def close(self):
        # communicate() handles stdin close internally — do NOT close stdin
        # separately first or Python 3.12 raises "flush of closed file".
        _, stderr = self._proc.communicate()
        rc = self._proc.returncode
        if rc != 0:
            print(f"[FFMPEG] ERROR (rc={rc}) encoding {self._path}:\n{stderr.decode()}")
        else:
            print(f"[VIDEO] {self._n} frames → {self._path}")
        return rc == 0


# ─────────────────────────────────────────────────────────────────────────────
# Isaac Lab helpers
# ─────────────────────────────────────────────────────────────────────────────

def capture_frame(obs: dict) -> np.ndarray | None:
    """Extract the overhead RGB frame from the env's observation dict.

    We read from obs['visual']['rgb_overhead'] rather than directly from
    the camera sensor buffer — this is exactly the image the policy will
    see during inference, and it avoids stale-buffer / render-lag issues.
    The env step already flushed the camera before returning obs.
    """
    try:
        rgb = obs["visual"]["rgb_overhead"]   # (1, H, W, 3) or (H, W, 3)
        if rgb.ndim == 4:
            rgb = rgb[0]
        frame = rgb.cpu().numpy()
        if frame.dtype != np.uint8:
            # float in [0,1] → uint8
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        return frame
    except Exception as e:
        print(f"[CAMERA] Capture from obs failed: {e}")
        return None


def print_camera_pose(env):
    """Diagnostic: print camera world position so we can verify it's aimed at the robot."""
    try:
        cam = env.unwrapped.scene["camera_overhead"]
        pos  = cam.data.pos_w[0]
        quat = cam.data.quat_w_world[0]
        print(f"[DEBUG] camera_overhead  pos={[round(float(v),3) for v in pos]}"
              f"  quat_wxyz={[round(float(v),3) for v in quat]}")
    except Exception as e:
        print(f"[DEBUG] Could not read camera pose: {e}")


def clear_reset_buf(env):
    try:
        u = env.unwrapped
        if hasattr(u, "reset_buf"):
            u.reset_buf[:] = 0
        elif hasattr(u, "_reset_buf"):
            u._reset_buf[:] = 0
        elif hasattr(u, "episode_length_buf"):
            u.episode_length_buf[:] = 0
        if hasattr(u, "termination_manager") and hasattr(u.termination_manager, "reset"):
            u.termination_manager.reset()
    except Exception as e:
        print(f"[WARN] Could not clear reset_buf: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Cylinder / cup detection + placement
# (ported from replay_episodes.py — must stay in sync)
# ─────────────────────────────────────────────────────────────────────────────

# ── Real-camera pixel→world homography ───────────────────────────────────────
# Origin: top-left corner of the real overhead image.
_H_REAL = np.array([
    [+1.0641627654969605e-03, +4.4363254411092787e-05, +0.0000000000000000e+00],
    [+0.0000000000000000e+00, +1.1090813850672171e-03, +0.0000000000000000e+00],
    [+0.0000000000000000e+00, +1.3048009790061474e-04, +1.0000000000000000e+00],
], dtype=np.float64)
_H_REAL_INV = np.linalg.inv(_H_REAL)

# ── HSV colour gates ─────────────────────────────────────────────────────────
_CYL_HSV_LO = np.array(( 35,  40,  40), dtype=np.uint8)   # green cylinder
_CYL_HSV_HI = np.array(( 85, 255, 255), dtype=np.uint8)
_CUP_HSV_LO = np.array(( 15,  50,  50), dtype=np.uint8)   # yellow/orange cup
_CUP_HSV_HI = np.array(( 38, 255, 255), dtype=np.uint8)

# ── Real-world → Isaac coordinate offsets ───────────────────────────────────
REAL_TO_ISAAC_X_OFFSET_M = +0.31
REAL_TO_ISAAC_Y_OFFSET_M = +0.090


def pix2world_real(col: float, row: float):
    """Real-camera pixel (col, row) → world (X, Y) in metres."""
    import cv2 as _cv2_h
    pt  = np.array([[[float(col), float(row)]]], dtype=np.float32)
    out = _cv2_h.perspectiveTransform(pt, _H_REAL.astype(np.float32))
    return float(out[0, 0, 0]), float(out[0, 0, 1])


_pix2world = pix2world_real  # internal alias


def real_world_to_isaac(wx, wy):
    """_H_REAL world coords → Isaac world coords (Rz180 + calibration offset)."""
    return (
        float(-wx + REAL_TO_ISAAC_X_OFFSET_M),
        float( wy + REAL_TO_ISAAC_Y_OFFSET_M),
    )


def detect_cylinder_in_real(real_root, episode_idx):
    """
    Read real overhead video frame 0 for *episode_idx*, run HSV blob detection,
    apply parallax correction, and return cylinder + cup coordinates.

    Returns:
        (frame, px, py, wx, wy, cyl_bx, cyl_by, cyl_bw, cyl_bh,
         cup_px, cup_py, cup_wx, cup_wy, cup_bx, cup_by, cup_bw, cup_bh)
    Raises RuntimeError if the cylinder cannot be found.
    """
    import cv2

    # ── Locate the video file ────────────────────────────────────────────────
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
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ── Cylinder (green) detection ───────────────────────────────────────────
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
    _dil = 4
    cyl_bx += _dil; cyl_by += _dil; cyl_bw -= 2 * _dil; cyl_bh -= 2 * _dil
    print(f"[REAL] Cylinder pixel centroid: ({px}, {py})")

    # ── Parallax correction (anchor-corner method, 1.5 cm = 14 px) ──────────
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
    print(f"[REAL] Cylinder world: ({wx:+.4f}, {wy:.4f}) m  [parallax-corrected]")

    # ── Cup (yellow) detection in right half ─────────────────────────────────
    cup_px, cup_py, cup_wx, cup_wy = None, None, None, None
    cup_bx, cup_by, cup_bw, cup_bh = None, None, None, None
    try:
        right_hsv = hsv.copy()
        right_hsv[:, :w // 2] = 0
        cup_mask  = cv2.inRange(right_hsv, _CUP_HSV_LO, _CUP_HSV_HI)
        cup_mask  = cv2.morphologyEx(cup_mask, cv2.MORPH_OPEN,   k)
        cup_mask  = cv2.morphologyEx(cup_mask, cv2.MORPH_DILATE, k)
        cup_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cup_contours:
            cc = max(cup_contours, key=cv2.contourArea)
            if cv2.contourArea(cc) >= 50:
                Mc = cv2.moments(cc)
                if Mc["m00"] != 0:
                    cup_px = int(Mc["m10"] / Mc["m00"])
                    cup_py = int(Mc["m01"] / Mc["m00"])
                    cup_bx, cup_by, cup_bw, cup_bh = cv2.boundingRect(cc)
                    cup_bx += _dil; cup_by += _dil; cup_bw -= 2 * _dil; cup_bh -= 2 * _dil
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
                    print(f"[REAL] Cup world: ({cup_wx:+.4f}, {cup_wy:.4f}) m  [parallax-corrected]")
    except Exception as _cup_e:
        print(f"[REAL] Cup detection failed: {_cup_e}")

    return (frame, px, py, wx, wy,
            cyl_bx, cyl_by, cyl_bw, cyl_bh,
            cup_px, cup_py, cup_wx, cup_wy,
            cup_bx, cup_by, cup_bw, cup_bh)


def place_cylinder_at_coords(env, wx, wy):
    """
    Move the SIM cylinder to real-world coords (wx, wy).
    Writes PhysX pose (multi-API fallback) + USD authored translate.
    """
    import warnings as _w
    from pxr import UsdGeom, Gf
    import omni.usd

    _ix, _iy = real_world_to_isaac(wx, wy)
    xyz    = (_ix, _iy, 0.059)
    device = env.unwrapped.device
    obj    = env.unwrapped.scene["cylinder"]

    pose_t = torch.tensor([[_ix, _iy, 0.059, 1.0, 0.0, 0.0, 0.0]],
                          dtype=torch.float32, device=device)
    vel_t  = torch.zeros(1, 6, dtype=torch.float32, device=device)

    physx_ok = False
    try:
        obj.write_root_link_pose_to_sim(pose_t)
        obj.write_root_com_velocity_to_sim(vel_t)
        physx_ok = True
        print("[PLACE] write_root_link_pose_to_sim() succeeded")
    except Exception as _e1:
        print(f"[PLACE] write_root_link_pose_to_sim failed ({_e1})")

    if not physx_ok:
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore", DeprecationWarning)
                _view = obj.root_view
            try:
                _view.set_transforms(pose_t.cpu().numpy(), indices=None)
                _view.set_velocities(vel_t.cpu().numpy(), indices=None)
                physx_ok = True
                print("[PLACE] root_view.set_transforms() succeeded")
            except Exception as _e2a:
                pos_np = pose_t[:, :3].cpu().numpy()
                rot_np = pose_t[:, 3:].cpu().numpy()
                _view.set_world_poses(positions=pos_np, orientations=rot_np, indices=None)
                physx_ok = True
                print("[PLACE] root_view.set_world_poses() succeeded")
        except Exception as _e2:
            print(f"[PLACE] root_view access failed ({_e2})")

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

    # ── USD authored translate (keeps RTX render in sync) ────────────────────
    stage = omni.usd.get_context().get_stage()
    try:
        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            try:
                _prim_paths = obj.root_view.prim_paths
            except AttributeError:
                _prim_paths = obj.root_physx_view.prim_paths
        prim_path = "/".join(_prim_paths[0].split("/")[:5])
    except Exception:
        prim_path = obj.cfg.prim_path.replace("{ENV_IDX}", "0")
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        xf   = UsdGeom.Xformable(prim)
        t_op = next((op for op in xf.GetOrderedXformOps()
                     if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
        if t_op is None:
            t_op = xf.AddTranslateOp()
        t_op.Set(Gf.Vec3d(*xyz))
        print(f"[PLACE] Cylinder → {prim_path}  Isaac=({_ix:+.4f}, {_iy:.4f}, 0.059)")
    else:
        print(f"[PLACE] WARNING: cylinder prim not found at {prim_path}")


def place_cup_at_coords(env, wx, wy):
    """
    Move the SIM target cup to real-world coords (wx, wy).
    Mirrors place_cylinder_at_coords() but targets scene["target"] at z=0.060.
    """
    import warnings as _w
    from pxr import UsdGeom, Gf
    import omni.usd

    _ix, _iy = real_world_to_isaac(wx, wy)
    xyz    = (_ix, _iy, 0.060)
    device = env.unwrapped.device
    obj    = env.unwrapped.scene["target"]

    pose_t = torch.tensor([[_ix, _iy, 0.060, 1.0, 0.0, 0.0, 0.0]],
                          dtype=torch.float32, device=device)
    vel_t  = torch.zeros(1, 6, dtype=torch.float32, device=device)

    physx_ok = False
    try:
        obj.write_root_link_pose_to_sim(pose_t)
        obj.write_root_com_velocity_to_sim(vel_t)
        physx_ok = True
        print(f"[CUP] PhysX (4.x API) → Isaac=({_ix:+.4f}, {_iy:.4f}, 0.060)")
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
            print(f"[CUP] PhysX (legacy API) → Isaac=({_ix:+.4f}, {_iy:.4f}, 0.060)")
        except Exception as _pe2:
            print(f"[CUP] All PhysX APIs failed: {_pe2}")

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
        t_op = next((op for op in xf.GetOrderedXformOps()
                     if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
        if t_op is None:
            t_op = xf.AddTranslateOp()
        t_op.Set(Gf.Vec3d(*xyz))
        print(f"[CUP] USD translate → {prim_path}  Isaac=({_ix:+.4f}, {_iy:.4f}, 0.060)")
    else:
        print(f"[CUP] WARNING: target prim not found at {prim_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    real_root = os.path.expanduser(
        f"~/.cache/huggingface/lerobot/{args_cli.real_repo_id}"
    )
    out_root = os.path.expanduser(
        f"~/.cache/huggingface/lerobot/{args_cli.sim_out_id}"
    )

    print("=" * 70)
    print("  record_sim_episodes.py")
    print(f"  REAL : {real_root}")
    print(f"  SIM  : {out_root}")
    print("=" * 70)

    if os.path.exists(out_root):
        if _overwrite:
            shutil.rmtree(out_root)
            print(f"[OUT] Deleted existing dataset at {out_root}")
        else:
            raise RuntimeError(
                f"Output dataset already exists: {out_root}\n"
                "Use --overwrite to delete and recreate it."
            )
    os.makedirs(out_root, exist_ok=True)

    # ── Load real episodes ────────────────────────────────────────────────────
    real_df = load_all_real_episodes(real_root)
    all_episode_indices = sorted(real_df["episode_index"].unique().tolist())

    if _episodes_arg == "all":
        episode_indices = all_episode_indices
    else:
        episode_indices = [int(x.strip()) for x in _episodes_arg.split(",")]
        missing = [e for e in episode_indices if e not in all_episode_indices]
        if missing:
            raise RuntimeError(f"Episode(s) not in real dataset: {missing}")

    print(f"[PLAN] Recording {len(episode_indices)} episode(s): {episode_indices}")

    # ── Init Isaac Lab env ────────────────────────────────────────────────────
    print("\n[SIM] Initialising Isaac Lab …")
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
        port="dummy", id="replay", cameras=cameras, fps=FPS, kind="leader",
    )
    env.reset()
    for _ in range(10):
        simulation_app.update()

    # ── Set camera via USD stage — same method as replay_episodes ─────────────
    # Directly manipulating the USD prim's xform is what replay_episodes uses
    # and is the approach that actually produces a top-down view.
    # _cam_cfg.offset.rot is stored in scipy [x,y,z,w] convention; Gf.Quatd takes (w,x,y,z).
    try:
        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        _CAM_PATH = "/World/envs/env_0/CameraOverhead"
        cam_prim = stage.GetPrimAtPath(_CAM_PATH)
        if cam_prim.IsValid():
            xf  = UsdGeom.Xformable(cam_prim)
            ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
            # Position: from env_cfg
            _cp = env_cfg.scene.camera_overhead.offset.pos
            for t in ("xformOp:translate", "xformOp:translation"):
                if t in ops:
                    ops[t].Set(Gf.Vec3d(float(_cp[0]), float(_cp[1]), float(_cp[2]))); break
            # Rotation: _cam_cfg.offset.rot is scipy [x,y,z,w]; Gf.Quatd(w,x,y,z)
            _rot = env_cfg.scene.camera_overhead.offset.rot
            qx, qy, qz, qw = float(_rot[0]), float(_rot[1]), float(_rot[2]), float(_rot[3])
            print(f"[CAM] USD prim rot: Gf.Quatd(w={qw:.5f}, x={qx:.5f}, y={qy:.5f}, z={qz:.5f})")
            for r in ("xformOp:orient", "xformOp:orientation"):
                if r in ops:
                    ops[r].Set(Gf.Quatd(qw, qx, qy, qz)); break
            if "xformOp:rotateXYZ" in ops:
                ops["xformOp:rotateXYZ"].Set(Gf.Vec3f(0.0, 0.0, 180.0))
            simulation_app.update()
            print(f"[CAM] USD camera prim set at {_CAM_PATH}")
        else:
            print(f"[CAM] WARNING: {_CAM_PATH} not found in stage")
    except Exception as _cam_e:
        print(f"[CAM] USD camera setup failed (continuing): {_cam_e}")

    # ── Apply graphpaper mat texture — same as replay_episodes ───────────────
    # Looks for graphpaper_mat.png in the same directory as this script.
    try:
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _tex_path   = os.path.join(_script_dir, "graphpaper_mat.png")
        if not os.path.isfile(_tex_path):
            # Fallback: beside replay_episodes.py (common project dir)
            _replay_dir = os.path.dirname(os.path.abspath(
                __import__("shutil").which("replay_episodes.py") or _tex_path))
            _tex_path = os.path.join(_script_dir, "graphpaper_mat.png")
        sys.path.insert(0, _script_dir)
        from apply_graphpaper_texture import apply_graphpaper_texture as _apply_tex
        import omni.usd as _omni_usd_tex
        _tex_stage = _omni_usd_tex.get_context().get_stage()
        _apply_tex(_tex_stage, texture_path=_tex_path)
        simulation_app.update()
        print(f"[SIM] Graphpaper mat texture applied from {_tex_path}")
    except Exception as _tex_e:
        import traceback as _tb2
        print(f"[SIM] apply_graphpaper_texture failed (continuing): {_tex_e}")
        _tb2.print_exc()

    # ── Recolour robot arm to match real SO-101 appearance ───────────────────
    # Creates a fresh UsdPreviewSurface material and binds it to every Mesh
    # prim under the robot root — more reliable than patching MDL input names.
    try:
        import omni.usd as _omni_usd_rc
        _rc_stage = _omni_usd_rc.get_context().get_stage()
        sys.path.insert(0, _script_dir)   # _script_dir already set by texture block
        from apply_robot_recolor import apply_robot_recolor as _apply_recolor
        _n_recoloured = _apply_recolor(_rc_stage)
        if _n_recoloured:
            simulation_app.update()
    except Exception as _rc_e:
        import traceback as _tb_rc
        print(f"[SIM] Robot recolour failed (continuing): {_rc_e}")
        _tb_rc.print_exc()

    from omni.timeline import get_timeline_interface
    timeline = get_timeline_interface()
    timeline.play()

    # ── Recording state ───────────────────────────────────────────────────────
    # CHUNKS_SIZE episodes per chunk → all 15 real episodes fit in chunk-000.
    # The video writer for each chunk stays open across all episodes in that
    # chunk, producing one concatenated MP4 file (same as real dataset).

    all_rows:     list[dict] = []
    episode_meta: list[dict] = []
    global_index  = 0

    # Build chunk groups: {chunk_idx: [ep_idx, ...]}
    chunk_groups: dict[int, list[int]] = {}
    for ep_idx in episode_indices:
        c = ep_idx // CHUNKS_SIZE
        chunk_groups.setdefault(c, []).append(ep_idx)

    # ── Process chunk by chunk ────────────────────────────────────────────────
    for chunk_idx in sorted(chunk_groups.keys()):
        eps_in_chunk = sorted(chunk_groups[chunk_idx])

        video_dir  = os.path.join(out_root, "videos", VIDEO_KEY, f"chunk-{chunk_idx:03d}")
        video_path = os.path.join(video_dir, f"file-{chunk_idx:03d}.mp4")
        writer = AV1VideoWriter(video_path, fps=FPS, width=IMG_W, height=IMG_H)
        print(f"\n[CHUNK {chunk_idx}] {len(eps_in_chunk)} episode(s) → {video_path}")

        for ep_idx in eps_in_chunk:
            ep_df    = real_df[real_df["episode_index"] == ep_idx].sort_values("frame_index")
            frames   = list(ep_df.iterrows())
            n_frames = len(frames)

            # Record where this episode starts inside the chunk video file
            ep_from_ts = writer.current_offset_s
            print(f"\n  [EP {ep_idx}] {n_frames} frames, video offset = {ep_from_ts:.3f}s")

            env.reset()
            for _ in range(20):          # more warm-up to fully hydrate renderer
                simulation_app.update()
            clear_reset_buf(env)

            # ── Detect cylinder + cup from real video, place in sim ───────────
            # TGS warm-start clearing: place → flush stale contacts → re-place.
            # The double-place is essential: the first call exposes any stale
            # warm-starts from the reset; the sim.step() flushes them; the
            # second call sets the position cleanly.
            _det_wx, _det_wy, _det_cup_wx, _det_cup_wy = None, None, None, None
            try:
                (_, _det_px, _det_py, _det_wx, _det_wy,
                 _det_cbx, _det_cby, _det_cbw, _det_cbh,
                 _det_cup_px, _det_cup_py, _det_cup_wx, _det_cup_wy,
                 _det_cup_bx, _det_cup_by, _det_cup_bw, _det_cup_bh) = \
                    detect_cylinder_in_real(real_root, ep_idx)
            except Exception as _det_e:
                print(f"[DETECT] Ep {ep_idx}: detection failed ({_det_e}) — using default positions")

            if _det_wx is not None:
                # Step 1: place (exposes stale warm-starts)
                place_cylinder_at_coords(env, _det_wx, _det_wy)
                if _det_cup_wx is not None:
                    place_cup_at_coords(env, _det_cup_wx, _det_cup_wy)
                # Step 2: flush TGS stale contacts
                env.unwrapped.sim.step(render=False)
                # Step 3: re-place into clean state
                place_cylinder_at_coords(env, _det_wx, _det_wy)
                if _det_cup_wx is not None:
                    place_cup_at_coords(env, _det_cup_wx, _det_cup_wy)
                _ix_log, _iy_log = real_world_to_isaac(_det_wx, _det_wy)
                print(f"[PLACE] Ep {ep_idx}: cylinder at Isaac=({_ix_log:+.4f}, {_iy_log:.4f})")

            ep_rows: list[dict] = []
            recorded = 0   # frames actually written (after SKIP_FRAMES skipped)

            for frame_pos, (_, row) in enumerate(frames):
                real_action = torch.tensor(
                    row["action"], dtype=torch.float32, device=env.unwrapped.device
                )
                sim_action = robot_iface.get_mapped_actions_vectorized(real_action)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    obs, _, _, _, _ = env.step(sim_action.unsqueeze(0))

                simulation_app.update()
                env.unwrapped.sim.render()

                # Skip the first SKIP_FRAMES frames — sim robot starts in an
                # extended default pose while real episodes start contracted.
                # Physics still advances so the replay trajectory stays correct.
                if frame_pos < SKIP_FRAMES:
                    continue

                # Print camera world pose on the first recorded frame.
                if recorded == 0:
                    print_camera_pose(env)

                frame_rgb = capture_frame(obs)
                if frame_rgb is None:
                    frame_rgb = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

                # ── Debug: save first recorded frame of first episode as PNG ─
                if ep_idx == episode_indices[0] and recorded == 0:
                    import imageio
                    dbg_path = os.path.join(out_root, "debug_frame0.png")
                    imageio.imwrite(dbg_path, frame_rgb)
                    print(f"[DEBUG] First recorded frame saved → {dbg_path}  "
                          f"shape={frame_rgb.shape}  "
                          f"min={frame_rgb.min()}  max={frame_rgb.max()}")
                # ─────────────────────────────────────────────────────────────

                writer.write(frame_rgb)

                ep_rows.append({
                    "action":            list(map(float, row["action"])),
                    "observation.state": list(map(float, row["observation.state"])),
                    "timestamp":         float(recorded) / FPS,
                    "frame_index":       int(recorded),
                    "episode_index":     int(ep_idx),
                    "index":             int(global_index),
                    "task_index":        TASK_INDEX,
                })
                global_index += 1
                recorded += 1

                n_rec = n_frames - SKIP_FRAMES
                if recorded % 100 == 0 or recorded == n_rec:
                    print(f"    frame {recorded}/{n_rec}", flush=True)

            # Notify writer of frames actually written (not total real frames)
            writer.end_episode(recorded)

            all_rows.extend(ep_rows)
            ep_stats = compute_ep_stats(ep_rows)
            ep_to_ts = ep_from_ts + recorded / FPS   # recorded frames only, not n_frames

            ep_entry: dict = {
                "episode_index":    int(ep_idx),
                "tasks":            ["Pick cylinder to target"],
                "length":           int(recorded),   # actual rows written, not pre-skip total
                # data parquet location
                "data/chunk_index": chunk_idx,
                "data/file_index":  chunk_idx,
                # dataset-global row range (filled in below after all episodes recorded)
                "dataset_from_index": 0,
                "dataset_to_index":   0,
                # video location + timestamp window
                TS_COL:                              ep_from_ts,
                f"videos/{VIDEO_KEY}/to_timestamp":  ep_to_ts,
                CHUNK_COL:                           chunk_idx,
                FILE_COL:                            chunk_idx,
                # meta/episodes file location (always chunk-000 / file-000)
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index":  0,
            }
            # Per-episode stats for every feature
            for feat, stat_dict in ep_stats.items():
                for stat_name, val in stat_dict.items():
                    ep_entry[f"stats/{feat}/{stat_name}"] = val

            episode_meta.append(ep_entry)

        # Close the video file for this chunk (all episodes concatenated inside)
        writer.close()

    # ── Write parquet chunks (episode-based chunking, same as video chunks) ────
    # CHUNKS_SIZE is episodes-per-chunk, not rows-per-chunk.  All rows that
    # belong to episodes in chunk-N go into data/chunk-N/file-N.parquet.
    print(f"\n[PARQUET] Writing {len(all_rows)} total rows …")
    chunk_row_groups: dict[int, list[dict]] = {}
    for row in all_rows:
        c_idx = int(row["episode_index"]) // CHUNKS_SIZE
        chunk_row_groups.setdefault(c_idx, []).append(row)

    for c_idx, rows in sorted(chunk_row_groups.items()):
        data_dir = os.path.join(out_root, "data", f"chunk-{c_idx:03d}")
        write_data_parquet(rows, data_dir, file_idx=c_idx)

    # ── Fill in global dataset row indices ───────────────────────────────────
    # Sort by episode_index so cumulative sum is in the right order.
    episode_meta.sort(key=lambda x: x["episode_index"])
    cumulative = 0
    for ep in episode_meta:
        ep["dataset_from_index"] = cumulative
        cumulative += ep["length"]
        ep["dataset_to_index"]   = cumulative

    # ── Write metadata ────────────────────────────────────────────────────────
    write_tasks_parquet(out_root)
    write_episodes_meta(episode_meta, out_root)
    write_stats_json(out_root, all_rows)
    write_info_json(out_root, len(episode_indices), len(all_rows))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RECORDING COMPLETE")
    print(f"  Episodes  : {len(episode_indices)}")
    print(f"  Frames    : {len(all_rows)}")
    print(f"  Dataset   : {out_root}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()