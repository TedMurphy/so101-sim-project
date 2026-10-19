#!/usr/bin/env python3
"""
compare_placement.py  —  sim vs real side-by-side with world-grid overlay.

Usage (from the project scripts directory so homography_pix2world is importable):
  python3 compare_placement.py --episode 7

Output: compare_ep<N>.png  (saved to CWD)
"""

import argparse
import os
import subprocess
import sys
import numpy as np
import cv2

# ── import homography helpers ────────────────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from homography_pix2world import (
    pix2world_real,
    world2pix_real,
    world2pix_sim,
)

# ── config ───────────────────────────────────────────────────────────────────
REAL_REPO_ID       = "Screener2/V4_pnp_44_20260720_141907"
HF_CACHE           = os.path.expanduser(f"~/.cache/huggingface/lerobot/{REAL_REPO_ID}")
SIM_FRAME_TEMPLATE = os.path.expanduser(
    "~/so101-sim-project/sim_overhead_ep{ep}_frame5.png"
)

# HSV detection (same as replay_interactive.py)
_DET_IMG_CX      = 320
_DET_HSV_LO_CYL  = (35,  40,  40)
_DET_HSV_HI_CYL  = (85, 255, 255)
_DET_HSV_LO_TGT  = (15,  50,  50)
_DET_HSV_HI_TGT  = (38, 255, 255)
_MIN_BLOB_AREA   = 50

# Grid spacing / extents (metres)
GRID_DX  = 0.05   # 5 cm
GRID_DY  = 0.05
GRID_X_MIN, GRID_X_MAX = -0.42,  0.36
GRID_Y_MIN, GRID_Y_MAX = -0.03,  0.43

# Colours (BGR)
COL_GRID_X   = (210, 100,   0)   # blue  (world-X lines = constant X)
COL_GRID_Y   = (0,   210, 230)   # yellow (world-Y lines = constant Y)
COL_CYL      = (60,  220,  80)   # green
COL_TGT      = (0,   200, 255)   # yellow-orange
COL_LABEL    = (255, 255, 255)
PANEL_W      = 780
PANEL_H      = 520
TITLE_H      = 44


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_video_path(real_root, ep_idx):
    for pat in [
        f"videos/observation.images.overhead/chunk-000/episode_{ep_idx:06d}.mp4",
        f"videos/observation.images.overhead/chunk-{ep_idx // 500:03d}/episode_{ep_idx:06d}.mp4",
    ]:
        p = os.path.join(real_root, pat)
        if os.path.exists(p):
            return p, 0.0
    chunk = os.path.join(real_root, "videos/observation.images.overhead/chunk-000/file-000.mp4")
    if os.path.exists(chunk):
        try:
            import pandas as _pd
            meta = _pd.read_parquet(
                os.path.join(real_root, "meta/episodes/chunk-000/file-000.parquet"))
            row = meta[meta["episode_index"] == ep_idx]
            if not row.empty:
                ts = float(row["videos/observation.images.overhead/from_timestamp"].iloc[0])
                return chunk, ts
        except Exception as e:
            print(f"[WARN] meta read failed: {e}")
        return chunk, 0.0
    return None, 0.0


def _read_frame(vpath, t_s, target_w=640, target_h=480):
    """Extract one BGR frame, resize to target_w×target_h."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", vpath],
        capture_output=True, text=True)
    native_w, native_h = map(int, probe.stdout.strip().split(","))
    print(f"[compare] video native resolution: {native_w}×{native_h}")

    cmd = ["ffmpeg", "-ss", str(t_s), "-i", vpath,
           "-frames:v", "1",
           "-vf", f"scale={target_w}:{target_h}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True)
    expected = target_w * target_h * 3
    if r.returncode == 0 and len(r.stdout) == expected:
        return np.frombuffer(r.stdout, dtype=np.uint8).reshape(target_h, target_w, 3).copy()
    print(f"[compare] ffmpeg failed ret={r.returncode} got={len(r.stdout)} exp={expected}")
    return None


def _detect_blobs(frame):
    """Return raw detected world positions (before X-negate) or None."""
    import math
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def _largest_blob(mask_in):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(mask_in, cv2.MORPH_OPEN,   k)
        m = cv2.morphologyEx(m,       cv2.MORPH_DILATE, k)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < _MIN_BLOB_AREA:
            return None
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        return int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

    # cylinder: green, LEFT half
    cyl_mask = cv2.inRange(hsv, np.array(_DET_HSV_LO_CYL), np.array(_DET_HSV_HI_CYL))
    cyl_mask[:, _DET_IMG_CX:] = 0
    cp = _largest_blob(cyl_mask)

    # target: yellow, RIGHT half
    tgt_mask = cv2.inRange(hsv, np.array(_DET_HSV_LO_TGT), np.array(_DET_HSV_HI_TGT))
    tgt_mask[:, :_DET_IMG_CX] = 0
    tp = _largest_blob(tgt_mask)

    BASE_X, BASE_Y, MIN_DIST = -0.03, 0.0, 0.12
    results = {}
    if cp:
        wx, wy = pix2world_real(*cp)
        if math.sqrt((wx - BASE_X)**2 + (wy - BASE_Y)**2) >= MIN_DIST:
            results["cylinder_raw"] = (wx, wy)
            results["cylinder_px"]  = cp
    if tp:
        wx, wy = pix2world_real(*tp)
        if math.sqrt((wx - BASE_X)**2 + (wy - BASE_Y)**2) >= MIN_DIST:
            results["target_raw"] = (wx, wy)
            results["target_px"]  = tp
    return results


def _polyline_clipped(img, pts, color, thickness=1):
    """Draw a polyline, skipping out-of-bounds segments."""
    h, w = img.shape[:2]
    pts = [(x, y) for x, y in pts if -20 <= x < w+20 and -20 <= y < h+20]
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i+1], color, thickness, cv2.LINE_AA)


def draw_grid(img, w2p_fn):
    """Overlay world-coordinate grid using the given world→pixel function."""
    h_img, w_img = img.shape[:2]

    # World-Y lines: constant Y, vary X  →  roughly horizontal
    ys = np.arange(GRID_Y_MIN, GRID_Y_MAX + 1e-9, GRID_DY)
    xs_dense = np.arange(GRID_X_MIN, GRID_X_MAX + 1e-9, 0.01)
    for y in ys:
        pts = [tuple(int(v) for v in w2p_fn(x, y)) for x in xs_dense]
        _polyline_clipped(img, pts, COL_GRID_Y, 1)
        # label at centre of line
        mid_pt = tuple(int(v) for v in w2p_fn(0.0, y))
        if 10 < mid_pt[0] < w_img - 10 and 5 < mid_pt[1] < h_img - 5:
            cv2.putText(img, f"{y*100:.0f}", mid_pt,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, COL_GRID_Y, 1, cv2.LINE_AA)

    # World-X lines: constant X, vary Y  →  roughly vertical
    xs = np.arange(GRID_X_MIN, GRID_X_MAX + 1e-9, GRID_DX)
    ys_dense = np.arange(GRID_Y_MIN, GRID_Y_MAX + 1e-9, 0.01)
    for x in xs:
        pts = [tuple(int(v) for v in w2p_fn(x, y)) for y in ys_dense]
        _polyline_clipped(img, pts, COL_GRID_X, 1)
        # label near top of line
        label_pt = tuple(int(v) for v in w2p_fn(x, 0.01))
        if 5 < label_pt[0] < w_img - 5 and 5 < label_pt[1] < h_img - 5:
            cv2.putText(img, f"{x*100:.0f}", (label_pt[0]-8, label_pt[1]+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, COL_GRID_X, 1, cv2.LINE_AA)


def draw_object_marker(img, w2p_fn, world_xy, color, shape="circle", label=""):
    col, row = w2p_fn(*world_xy)
    col, row = int(col), int(row)
    h_img, w_img = img.shape[:2]
    if not (0 <= col < w_img and 0 <= row < h_img):
        print(f"[compare] marker {label} at ({col},{row}) outside image {w_img}×{h_img}")
        return
    if shape == "circle":
        cv2.circle(img, (col, row), 14, color, 2, cv2.LINE_AA)
        cv2.drawMarker(img, (col, row), color, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
    else:
        s = 12
        cv2.rectangle(img, (col - s, row - s), (col + s, row + s), color, 2, cv2.LINE_AA)
        cv2.drawMarker(img, (col, row), color, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
    if label:
        tx = col + 6 if col < w_img - 90 else col - 90
        ty = row - 8 if row > 20 else row + 18
        cv2.putText(img, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)


def add_legend(img):
    x0, y0 = 8, img.shape[0] - 48
    cv2.rectangle(img, (x0-2, y0-14), (x0+260, y0+34), (30,30,30), -1)
    cv2.line(img, (x0, y0),    (x0+20, y0),    COL_GRID_X, 1, cv2.LINE_AA)
    cv2.putText(img, "= world-X lines (5 cm)", (x0+24, y0+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, COL_GRID_X, 1, cv2.LINE_AA)
    cv2.line(img, (x0, y0+16), (x0+20, y0+16), COL_GRID_Y, 1, cv2.LINE_AA)
    cv2.putText(img, "= world-Y lines (5 cm)", (x0+24, y0+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, COL_GRID_Y, 1, cv2.LINE_AA)


def make_panel(img_bgr, title_str):
    """Resize to PANEL_W×PANEL_H and add title bar."""
    panel = cv2.resize(img_bgr, (PANEL_W, PANEL_H))
    top = np.zeros((TITLE_H, PANEL_W, 3), dtype=np.uint8)
    cv2.putText(top, title_str, (10, TITLE_H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([top, panel])


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=7)
    ap.add_argument("--tilt_x", type=float, default=1.0,
                    help="Camera tilt annotation (degrees, for title)")
    ap.add_argument("--tilt_y", type=float, default=-1.0)
    ap.add_argument("--out", type=str, default="",
                    help="Output path (default: compare_ep<N>.png in CWD)")
    args = ap.parse_args()

    ep = args.episode
    out_path = args.out or f"compare_ep{ep}.png"

    # ── 1. Load real frame ────────────────────────────────────────────────────
    vpath, vid_offset = _get_video_path(HF_CACHE, ep)
    if vpath is None:
        sys.exit(f"[compare] No video found for episode {ep}")

    print(f"[compare] reading real frame at t={vid_offset:.1f}s …")
    real_frame = _read_frame(vpath, vid_offset, target_w=640, target_h=480)
    if real_frame is None:
        sys.exit("[compare] Could not read real video frame")

    # ── 2. Detect objects in real frame ──────────────────────────────────────
    det = _detect_blobs(real_frame)
    cyl_raw = det.get("cylinder_raw")   # (wx, wy) before X-negate
    tgt_raw = det.get("target_raw")

    # Placed coords (after X-negate) — these are what go into sim
    cyl_placed = (-cyl_raw[0], cyl_raw[1]) if cyl_raw else None
    tgt_placed = (-tgt_raw[0], tgt_raw[1]) if tgt_raw else None

    print(f"[compare] cyl raw={cyl_raw}  placed={cyl_placed}")
    print(f"[compare] tgt raw={tgt_raw}  placed={tgt_placed}")

    # ── 3. Load sim overhead frame ───────────────────────────────────────────
    sim_path = SIM_FRAME_TEMPLATE.format(ep=ep)
    if not os.path.exists(sim_path):
        print(f"[compare] WARNING: sim frame not found at {sim_path}")
        sim_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        sim_frame = cv2.imread(sim_path)
        if sim_frame is None:
            sim_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            sim_frame = cv2.resize(sim_frame, (640, 480))

    # ── 4. Draw grids ─────────────────────────────────────────────────────────
    real_ann = real_frame.copy()
    sim_ann  = sim_frame.copy()

    draw_grid(real_ann, world2pix_real)
    draw_grid(sim_ann,  world2pix_sim)

    # ── 5. Draw object markers ────────────────────────────────────────────────
    # Real image: use RAW detected coords (before negate)
    if cyl_raw:
        lbl = f"cyl ({cyl_raw[0]:.2f},{cyl_raw[1]:.2f})"
        draw_object_marker(real_ann, world2pix_real, cyl_raw, COL_CYL, "circle", lbl)
    if tgt_raw:
        lbl = f"tgt ({tgt_raw[0]:.2f},{tgt_raw[1]:.2f})"
        draw_object_marker(real_ann, world2pix_real, tgt_raw, COL_TGT, "square", lbl)

    # Sim image: use PLACED coords (negated X)
    if cyl_placed:
        lbl = f"cyl ({cyl_placed[0]:.2f},{cyl_placed[1]:.2f})"
        draw_object_marker(sim_ann, world2pix_sim, cyl_placed, COL_CYL, "circle", lbl)
    if tgt_placed:
        lbl = f"tgt ({tgt_placed[0]:.2f},{tgt_placed[1]:.2f})"
        draw_object_marker(sim_ann, world2pix_sim, tgt_placed, COL_TGT, "square", lbl)

    add_legend(sim_ann)
    add_legend(real_ann)

    # ── 6. Compose side-by-side ───────────────────────────────────────────────
    sim_title  = f"SIM (ep{ep} initial) — TILT_X={args.tilt_x:.0f}°, TILT_Y={args.tilt_y:.0f}°"
    real_title = f"REAL (ep{ep} initial)"

    left  = make_panel(sim_ann,  sim_title)
    right = make_panel(real_ann, real_title)
    combined = np.hstack([left, right])

    cv2.imwrite(out_path, combined)
    print(f"[compare] saved → {out_path}")


if __name__ == "__main__":
    main()