"""calibrate_sim_homography_v3.py — Fix: read SIM camera from env.step() obs dict.

CHANGE vs v2
------------
v2 read the overhead frame via:
    env.unwrapped.scene["camera_overhead"].data.output["rgb"]
This returned the wrong buffer (a perspective/streaming camera view).

v3 fixes this in two ways:

  1.  _settle() now captures and returns the obs dict from the LAST env.step()
      call.  The overhead RGB is extracted from that dict, which is what the
      policy actually sees during training.

  2.  _find_rgb_in_obs() walks the obs tree on first call, prints every tensor
      it finds (shape + dtype), then returns the first one whose shape looks
      like a camera frame (H×W×3 or 1×H×W×3).  The key is auto-detected so
      you don't have to hard-code it — but it is printed so you can confirm.

  3.  Fallback: if nothing is found in obs, the script retries the direct
      scene-camera read and also tries common key variants ("rgba", "rgb_overhead").

  4.  Post-settle extra steps: after teleporting the cylinder the script runs
      a few extra no-move steps so the TiledCamera buffer has time to flush.

USAGE
-----
    ./isaaclab.sh -p calibrate_sim_homography_v3.py

Everything else (grid layout, homography fitting, output file) is unchanged.
"""

import argparse
import os
import sys
import datetime

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Calibrate _H_SIM using the scene cylinder.")
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

import isaaclab_tasks                          # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks                 # noqa: F401

# ── Tunable constants ─────────────────────────────────────────────────────────

OUT_DIR = os.path.expanduser("~/so101-sim-project/calib_sim")
os.makedirs(OUT_DIR, exist_ok=True)

_MARKER_Z    = 0.020   # Z of cylinder on the mat surface (m)
_HIDDEN_Z    = -0.50   # Z while "hidden" below floor (m)

_SETTLE_STEPS   = 20   # physics steps after teleport before reading camera
_EXTRA_STEPS    = 5    # additional no-move steps for TiledCamera buffer flush

# ── HSV detection for the GREEN cylinder ─────────────────────────────────────
_HSV_LO = np.array(( 35,  50,  50), dtype=np.uint8)
_HSV_HI = np.array(( 85, 255, 255), dtype=np.uint8)

_DIFF_THRESH  = 15
_MIN_AREA     = 30

# Calibration grid in WORLD coordinates (metres).
_GRID_WX = [-0.25, -0.10,  0.00, +0.15, +0.30]
_GRID_WY = [ 0.10,  0.19,  0.28,  0.37]

_RANSAC_THRESH = 0.010
_MIN_POINTS    = 6
_DUP_THRESH    = 8.0


# ── Cylinder helper ───────────────────────────────────────────────────────────

def _setup_cylinder(env, device):
    """Return move(ix, iy, iz) that teleports the existing cylinder."""
    cylinder = None
    for key in ("cylinder", "Cylinder", "cylinder_object", "rigid_object"):
        try:
            cylinder = env.unwrapped.scene[key]
            print(f"[CAL] Found cylinder under scene key: '{key}'")
            break
        except (KeyError, AttributeError):
            pass
    if cylinder is None:
        raise RuntimeError(
            "Could not find the cylinder in the scene.  "
            "Run the env and check env.unwrapped.scene.keys()."
        )

    def move(ix: float, iy: float, iz: float):
        root_state = cylinder.data.default_root_state.clone()
        root_state[0, 0] = float(ix)
        root_state[0, 1] = float(iy)
        root_state[0, 2] = float(iz)
        cylinder.write_root_pose_to_sim(root_state[:, :7])
        cylinder.write_root_velocity_to_sim(
            torch.zeros(1, 6, dtype=torch.float32, device=device)
        )

    return move


# ── Camera reading ────────────────────────────────────────────────────────────

_RGB_OBS_KEY  = None   # set on first successful find
_RGB_OBS_PATH = None   # dotted path for printing


def _walk_obs(obj, prefix=""):
    """Yield (dotted_path, tensor) for every tensor/array in the obs tree."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_obs(v, f"{prefix}{k}.")
    elif hasattr(obj, "items"):          # dict-like
        for k, v in obj.items():
            yield from _walk_obs(v, f"{prefix}{k}.")
    elif hasattr(obj, 'shape'):          # tensor or ndarray
        yield prefix.rstrip("."), obj
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_obs(v, f"{prefix}{i}.")


def _looks_like_camera(t) -> bool:
    """Return True if tensor shape looks like an image (H×W×3 or 1×H×W×3)."""
    s = tuple(t.shape)
    if len(s) == 3 and s[2] in (3, 4) and s[0] > 4 and s[1] > 4:
        return True
    if len(s) == 4 and s[0] == 1 and s[3] in (3, 4) and s[1] > 4 and s[2] > 4:
        return True
    return False


def _tensor_to_rgb(t) -> np.ndarray:
    """Convert any camera-shaped tensor to H×W×3 uint8 ndarray."""
    if hasattr(t, 'cpu'):
        arr = t.cpu().numpy()
    else:
        arr = np.asarray(t)
    if arr.ndim == 4:
        arr = arr[0]                     # drop batch dim
    arr = arr[..., :3]                   # drop alpha if present
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def _find_and_cache_rgb_key(obs):
    """Walk obs once, print structure, cache the best camera-shaped key."""
    global _RGB_OBS_KEY, _RGB_OBS_PATH
    print("\n[CAL] === Observation structure ===")
    candidates = []
    for path, tensor in _walk_obs(obs):
        s = tuple(tensor.shape) if hasattr(tensor, 'shape') else "?"
        if hasattr(tensor, 'dtype'):
            print(f"  {path}: shape={s}  dtype={tensor.dtype}")
        else:
            print(f"  {path}: shape={s}")
        if hasattr(tensor, 'shape') and _looks_like_camera(tensor):
            candidates.append((path, tensor))
    print("[CAL] === End observation structure ===\n")

    if not candidates:
        print("[CAL] WARNING: no camera-shaped tensor found in obs dict!")
        return

    # Prefer keys that mention "overhead" or "rgb"; otherwise take the first
    for path, tensor in candidates:
        if "overhead" in path.lower() or ("rgb" in path.lower() and "rgb" in path.lower()):
            _RGB_OBS_PATH = path
            # Build accessor key list
            _RGB_OBS_KEY  = path.split(".")
            print(f"[CAL] Using obs key: '{path}'  shape={tuple(tensor.shape)}")
            return
    # Fallback: first candidate
    path, tensor = candidates[0]
    _RGB_OBS_PATH = path
    _RGB_OBS_KEY  = path.split(".")
    print(f"[CAL] Using obs key (fallback): '{path}'  shape={tuple(tensor.shape)}")


def _get_tensor_from_obs(obs, key_list):
    """Navigate obs[key_list[0]][key_list[1]]... tolerating int keys."""
    obj = obs
    for k in key_list:
        try:
            obj = obj[k]
        except (KeyError, TypeError, IndexError):
            try:
                obj = obj[int(k)]
            except Exception:
                return None
    return obj


def _read_rgb_from_obs(obs) -> np.ndarray | None:
    """Read the overhead camera frame from the obs dict returned by env.step()."""
    global _RGB_OBS_KEY, _RGB_OBS_PATH

    # First call: walk and cache
    if _RGB_OBS_KEY is None:
        _find_and_cache_rgb_key(obs)
    if _RGB_OBS_KEY is None:
        return None

    tensor = _get_tensor_from_obs(obs, _RGB_OBS_KEY)
    if tensor is None:
        return None
    return _tensor_to_rgb(tensor)


def _read_rgb_from_scene(env) -> np.ndarray | None:
    """Fallback: read directly from the scene TiledCamera, trying several keys."""
    camera_keys = ["camera_overhead", "overhead_camera", "tiled_camera", "camera"]
    rgb_output_keys = ["rgb", "rgba", "rgb_overhead"]
    for ck in camera_keys:
        try:
            cam = env.unwrapped.scene[ck]
        except (KeyError, AttributeError):
            continue
        for rk in rgb_output_keys:
            try:
                tensor = cam.data.output[rk]
                print(f"[CAL] Scene camera: [{ck}].data.output['{rk}']  shape={tuple(tensor.shape)}")
                return _tensor_to_rgb(tensor)
            except (KeyError, AttributeError, TypeError):
                continue
    return None


def _settle(env, move_fn, ix, iy, iz, n_steps, zero_action):
    """Teleport cylinder then step n times; return (obs, rgb_ndarray).

    After the main settle steps, a few extra no-move steps give the
    TiledCamera buffer time to flush.
    """
    obs = None
    for _ in range(n_steps):
        move_fn(ix, iy, iz)
        obs, *_ = env.step(zero_action)

    # Extra flush steps (cylinder stays in place)
    for _ in range(_EXTRA_STEPS):
        obs, *_ = env.step(zero_action)

    # Try obs dict first
    rgb = _read_rgb_from_obs(obs)
    if rgb is None or rgb.mean() < 2.0:
        # Suspiciously dark or failed — try scene fallback
        rgb_fb = _read_rgb_from_scene(env)
        if rgb_fb is not None:
            print("[CAL] Using scene-camera fallback for this frame.")
            rgb = rgb_fb

    return obs, rgb


# ── Detection helpers ─────────────────────────────────────────────────────────

def _detect_blob_diff(rgb_fg, rgb_bg):
    fg16  = rgb_fg.astype(np.int16)
    bg16  = rgb_bg.astype(np.int16)
    diff  = np.abs(fg16 - bg16).max(axis=2).astype(np.uint8)
    _, mask = cv2.threshold(diff, _DIFF_THRESH, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ctrs:
        return None, None, mask
    c = max(ctrs, key=cv2.contourArea)
    if cv2.contourArea(c) < _MIN_AREA:
        return None, None, mask
    M = cv2.moments(c)
    if M["m00"] <= 0:
        return None, None, mask
    return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"]), mask


def _refine_with_hsv(rgb_fg, diff_mask):
    bgr      = cv2.cvtColor(rgb_fg, cv2.COLOR_RGB2BGR)
    hsv      = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, _HSV_LO, _HSV_HI)
    combined = cv2.bitwise_and(hsv_mask, diff_mask)
    ctrs, _  = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ctrs:
        return None, None
    c = max(ctrs, key=cv2.contourArea)
    if cv2.contourArea(c) < _MIN_AREA:
        return None, None
    M = cv2.moments(c)
    if M["m00"] <= 0:
        return None, None
    return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])


# ── Image saving ──────────────────────────────────────────────────────────────

def _save_diagnostic_diff(rgb_bg, rgb_fg, diff_mask, path_prefix):
    bgr_bg = cv2.cvtColor(rgb_bg, cv2.COLOR_RGB2BGR)
    bgr_fg = cv2.cvtColor(rgb_fg, cv2.COLOR_RGB2BGR)
    diff_c = cv2.applyColorMap(
        diff_mask if diff_mask is not None else np.zeros_like(bgr_bg[:, :, 0]),
        cv2.COLORMAP_HOT)
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = [("Background (cylinder hidden)", bgr_bg),
              ("Foreground (cylinder on mat)",  bgr_fg),
              ("Diff mask (bright=changed)",    diff_c)]
    panels = []
    for lbl, img in labels:
        p = img.copy()
        cv2.putText(p, lbl, (6, 18), font, 0.5, (0,   0,   0), 3, cv2.LINE_AA)
        cv2.putText(p, lbl, (6, 18), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        panels.append(p)
    cv2.imwrite(path_prefix + "_diag3.png", np.hstack(panels))
    cv2.imwrite(path_prefix + "_bg.png",   bgr_bg)
    cv2.imwrite(path_prefix + "_fg.png",   bgr_fg)
    print(f"[CAL] Diagnostic → {path_prefix}_diag3.png")


def _save_frame(rgb_np, diff_mask, col, row, wx, wy, method, pt_idx, path):
    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    if diff_mask is not None:
        overlay = bgr.copy()
        overlay[diff_mask > 0] = (200, 255, 255)
        bgr = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)
    if col is not None:
        ci = (int(round(col)), int(round(row)))
        cv2.circle(bgr, ci, 10, (255, 255, 0), 2)
        cv2.drawMarker(bgr, ci, (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"pt{pt_idx:02d}  world=({wx:+.3f},{wy:.3f})m",
        (f"blob=({col:.1f},{row:.1f}) [{method}]" if col is not None else "blob=NONE"),
    ]
    for i, line in enumerate(lines):
        cv2.putText(bgr, line, (8, 20 + i*22), font, 0.55, (0,   0,   0), 3, cv2.LINE_AA)
        cv2.putText(bgr, line, (8, 20 + i*22), font, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, bgr)


# ── Homography output ─────────────────────────────────────────────────────────

def _write_calibrated_module(H, src_pts, dst_pts, path):
    def _fmt(m):
        return "\n".join(
            "    [{:+.8e}, {:+.8e}, {:+.8e}],".format(*r) for r in m
        )
    today = datetime.date.today().isoformat()
    n     = len(src_pts)
    pt_table = "\n".join(
        f"  pt{i+1:02d}  SIM px ({c:6.1f},{r:6.1f})  world ({wx:+.3f},{wy:.3f})"
        for i, ((c, r), (wx, wy)) in enumerate(zip(src_pts, dst_pts))
    )
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from homography_pix2world import _H_REAL
    except Exception:
        _H_REAL = np.array([
            [+1.10696274e-03, +5.09168491e-07, -3.30000013e-01],
            [-4.20778601e-07, +8.31701143e-04, +4.99999989e-03],
            [-8.41557221e-05, +1.32766194e-04, +1.00000000e+00],
        ])
    H_inv = np.linalg.inv(H)
    _H_REAL_INV = np.linalg.inv(_H_REAL)

    content = f'''"""
Drop-in replacement for homography_pix2world.py.

  _H_REAL — unchanged (from existing calibration)
  _H_SIM  — empirically calibrated {today} from {n} cylinder detections
            (frame-differencing + HSV refine, Z={_MARKER_Z:.3f} m)

Calibration points:
{pt_table}
"""

import numpy as np
import cv2

_H_REAL = np.array([
{_fmt(_H_REAL)}
], dtype=np.float64)
_H_REAL_INV = np.linalg.inv(_H_REAL)

_H_SIM = np.array([
{_fmt(H)}
], dtype=np.float64)
_H_SIM_INV = np.linalg.inv(_H_SIM)


def pix2world_real(col: float, row: float):
    pt = np.array([[[float(col), float(row)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_REAL.astype(np.float32))
    return float(out[0,0,0]), float(out[0,0,1])

def pix2world_sim(col: float, row: float):
    pt = np.array([[[float(col), float(row)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_SIM.astype(np.float32))
    return float(out[0,0,0]), float(out[0,0,1])

def world2pix_real(wx: float, wy: float):
    pt = np.array([[[float(wx), float(wy)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_REAL_INV.astype(np.float32))
    return float(out[0,0,0]), float(out[0,0,1])

def world2pix_sim(wx: float, wy: float):
    pt = np.array([[[float(wx), float(wy)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, _H_SIM_INV.astype(np.float32))
    return float(out[0,0,0]), float(out[0,0,1])

_det_pix2world = pix2world_real
pix2world      = pix2world_real

if __name__ == "__main__":
    print("SIM centre (320,240) →", pix2world_sim(320, 240))
'''
    with open(path, "w") as f:
        f.write(content)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  calibrate_sim_homography_v3.py — obs-dict camera reading")
    print("=" * 70)

    # ── Environment ───────────────────────────────────────────────────────────
    try:
        env_cfg = parse_env_cfg(args_cli.task, use_gpu=False, num_envs=1, use_fabric=True)
    except TypeError:
        try:
            env_cfg = parse_env_cfg(args_cli.task, num_envs=1)
        except TypeError:
            env_cfg = parse_env_cfg(args_cli.task)
            env_cfg.scene.num_envs = 1
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    device = env.unwrapped.device
    try:
        n_act = env.unwrapped.action_space.shape[-1]
    except Exception:
        n_act = 6
    zero_action = torch.zeros(1, n_act, dtype=torch.float32, device=device)

    # Print scene keys so we can verify the cylinder key
    try:
        scene_keys = list(env.unwrapped.scene._entities.keys())
        print(f"[CAL] Scene entity keys: {scene_keys}")
    except Exception:
        try:
            scene_keys = list(vars(env.unwrapped.scene).keys())
            print(f"[CAL] Scene attr keys: {scene_keys}")
        except Exception:
            print("[CAL] Could not list scene keys.")

    # Walk initial obs to find the camera key (printed to console)
    print("[CAL] Walking initial obs to find camera tensor …")
    _find_and_cache_rgb_key(obs)

    # ── Get move function for the cylinder ───────────────────────────────────
    move_marker = _setup_cylinder(env, device)

    # ── Warm up ───────────────────────────────────────────────────────────────
    print(f"\n[CAL] Warming up (20 steps, cylinder hidden at Z={_HIDDEN_Z}) …")
    _, rgb_warmup = _settle(env, move_marker, 0.0, 0.20, _HIDDEN_Z, 20, zero_action)
    cv2.imwrite(os.path.join(OUT_DIR, "warmup_frame.png"),
                cv2.cvtColor(rgb_warmup, cv2.COLOR_RGB2BGR))
    print(f"[CAL] Warm-up frame saved → {OUT_DIR}/warmup_frame.png")
    print(f"[CAL] Warm-up frame: mean={rgb_warmup.mean():.1f}  shape={rgb_warmup.shape}")

    # ── Capture background (cylinder hidden) ─────────────────────────────────
    print("[CAL] Capturing background frame …")
    _, rgb_bg = _settle(env, move_marker, 0.0, 0.20, _HIDDEN_Z, _SETTLE_STEPS, zero_action)
    cv2.imwrite(os.path.join(OUT_DIR, "diagnostic_bg.png"),
                cv2.cvtColor(rgb_bg, cv2.COLOR_RGB2BGR))
    print(f"[CAL] Background mean brightness: {rgb_bg.mean():.1f}")

    # ── Diagnostic: place cylinder at a known visible position ────────────────
    diag_ix, diag_iy = 0.20, 0.30   # world (-0.20, 0.30)
    print(f"[CAL] Diagnostic: cylinder at Isaac({diag_ix},{diag_iy},{_MARKER_Z}) …")
    _, rgb_diag = _settle(env, move_marker, diag_ix, diag_iy, _MARKER_Z,
                          _SETTLE_STEPS, zero_action)
    _, _, diff_diag = _detect_blob_diff(rgb_diag, rgb_bg)
    prefix = os.path.join(OUT_DIR, "diagnostic")
    _save_diagnostic_diff(rgb_bg, rgb_diag, diff_diag, prefix)

    n_diff_px = int(np.sum(diff_diag > 0)) if diff_diag is not None else 0
    print(f"[CAL] Diff pixels at diagnostic position: {n_diff_px}")
    if n_diff_px < _MIN_AREA:
        print("[CAL] !! STILL ZERO diff pixels — camera buffer not updating.")
        print("[CAL] Check warmup_frame.png: is it the overhead mat view or something else?")
        print("[CAL] Also check obs structure printed above for the correct key.")
        print("[CAL] Proceeding anyway — all grid points will likely fail.")
    else:
        print("[CAL] OK — cylinder is visible. Proceeding with grid sweep.")

    # Return to hidden, fresh background
    _, rgb_bg = _settle(env, move_marker, 0.0, 0.20, _HIDDEN_Z, _SETTLE_STEPS, zero_action)

    # ── Grid sweep ────────────────────────────────────────────────────────────
    src_pts  = []
    dst_pts  = []
    failures = []
    total_pts   = len(_GRID_WX) * len(_GRID_WY)
    pt_idx      = 0
    _last_by_wy = {}

    for wx in _GRID_WX:
        for wy in _GRID_WY:
            pt_idx += 1
            ix = -wx
            iy =  wy
            iz = _MARKER_Z

            print(f"[CAL] Point {pt_idx:02d}/{total_pts}  "
                  f"world=({wx:+.3f},{wy:.3f})m  "
                  f"Isaac=({ix:+.3f},{iy:.3f},{iz:.3f})m")

            _, rgb_fg = _settle(env, move_marker, ix, iy, iz, _SETTLE_STEPS, zero_action)
            col_d, row_d, diff_mask = _detect_blob_diff(rgb_fg, rgb_bg)

            col, row, method = None, None, "none"
            if col_d is not None:
                col_h, row_h = _refine_with_hsv(rgb_fg, diff_mask)
                if col_h is not None:
                    col, row, method = col_h, row_h, "hsv"
                else:
                    col, row, method = col_d, row_d, "diff"

            # Hide, then refresh background
            _, rgb_bg = _settle(env, move_marker, 0.0, 0.20, _HIDDEN_Z,
                                _SETTLE_STEPS, zero_action)

            # Duplicate guard
            is_dup = False
            if col is not None and wy in _last_by_wy:
                prev_c, prev_r = _last_by_wy[wy]
                if np.hypot(col - prev_c, row - prev_r) < _DUP_THRESH:
                    print(f"  !! DUPLICATE ({col:.0f},{row:.0f}) ≈ prev X — skipping.")
                    is_dup = True
                    col, row = None, None

            _stem = (f"grid_pt_{pt_idx:02d}_wx{wx:+.3f}_wy{wy:.3f}"
                     .replace("+", "p").replace("-", "m").replace(".", "d"))
            _save_frame(rgb_fg, diff_mask, col, row, wx, wy, method, pt_idx,
                        os.path.join(OUT_DIR, _stem + ".png"))

            if col is not None:
                print(f"  blob ({col:.1f}, {row:.1f})  [{method}]")
                _last_by_wy[wy] = (col, row)
                src_pts.append([col, row])
                dst_pts.append([wx, wy])
            else:
                if not is_dup:
                    print("  !! BLOB NOT FOUND")
                failures.append((wx, wy))

    # ── Fit homography ────────────────────────────────────────────────────────
    print(f"\n[CAL] Detections: {len(src_pts)} / {total_pts}")
    if failures:
        print(f"[CAL] Failed: {failures}")

    if len(src_pts) < _MIN_POINTS:
        print(f"[CAL] ERROR: need ≥ {_MIN_POINTS} points, got {len(src_pts)}.")
        env.close(); simulation_app.close(); return

    src_np = np.array(src_pts, dtype=np.float64)
    dst_np = np.array(dst_pts, dtype=np.float64)

    H_new, mask = cv2.findHomography(src_np, dst_np, cv2.RANSAC, _RANSAC_THRESH)
    if H_new is None:
        print("[CAL] findHomography failed — degenerate point layout?")
        env.close(); simulation_app.close(); return

    inliers = int(mask.sum()) if mask is not None else len(src_pts)
    print(f"[CAL] RANSAC inliers: {inliers}/{len(src_pts)}")

    print("\n[CAL] Residuals:")
    max_err = 0.0
    for i, ((c, r), (wx, wy)) in enumerate(zip(src_pts, dst_pts)):
        h    = H_new @ np.array([c, r, 1.0])
        wx_p = h[0] / h[2]
        wy_p = h[1] / h[2]
        err  = np.hypot(wx_p - wx, wy_p - wy)
        max_err = max(max_err, err)
        in_flag = mask is not None and bool(mask[i])
        tag = "  ✓" if in_flag else "  ✗ outlier"
        if in_flag:
            print(f"  pt{i+1:02d} SIM({c:.1f},{r:.1f}) → "
                  f"pred({wx_p:+.4f},{wy_p:.4f}) truth({wx:+.4f},{wy:.4f}) "
                  f"err={err*100:.2f}cm{tag}")
    print(f"[CAL] Max residual: {max_err*100:.2f} cm")

    out_py = os.path.join(OUT_DIR, "homography_pix2world_calibrated.py")
    _write_calibrated_module(H_new, src_pts, dst_pts, out_py)
    print(f"\n[CAL] Written → {out_py}")
    print("[CAL] If residuals look good (<1 cm), copy into place:")
    print(f"[CAL]   cp {out_py} ~/so101-sim-project/source/sim_to_real_so101/homography_pix2world.py")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()