#!/usr/bin/env python3
"""
dual_panel_overlay_v4.py  —  REAL + SIM side-by-side with world grid overlay.

v4 changes vs v3:
  - Add --calibrate-sim mode: click 4 mat corners in the SIM image window to
    recompute _H_SIM and print the new matrix (paste into this file).
  - After every run, write detected positions to a JSON sidecar file
    (overlay_ep{N}_positions.json or dual_panel_positions.json) so that
    replay_interactive_v10.py --use-real-positions can pick them up.
  - detect_objects() also searches right half for cylinder if left-half mask
    is empty (handles episodes where cylinder is near centre).

Usage (direct images):
    python dual_panel_overlay_v4.py real.png sim.png [output.png]

Usage (episode mode — finds frames automatically):
    python dual_panel_overlay_v4.py --episode 7
    python dual_panel_overlay_v4.py --episode 7 --dataset ~/.cache/huggingface/lerobot/Screener2/V4_pnp_44_20260720_141907

Calibration mode (run whenever SIM grid stops matching the mat):
    python dual_panel_overlay_v4.py --episode 7 --calibrate-sim
    A window opens showing the SIM image. Click the four mat corners in this
    order: TL, TR, BL, BR.  The script prints the new _H_SIM matrix.
    Paste it into the _H_SIM constant below and save.

Episode mode looks for:
  REAL : {dataset}/videos/observation.images.overhead/chunk-000/episode_{N:06d}.mp4
         (extracted via ffmpeg — handles AV1 encoding)
  SIM  : {sim-dir}/sim_overhead_ep{N}_frame1.png or frame5.png
         (saved by replay_interactive.py)
"""

import sys
import argparse
import glob
import json
import os
import numpy as np
import cv2

# ── Homography matrices — single source of truth is homography_pix2world.py ──
# To recalibrate _H_SIM: python dual_panel_overlay.py --episode N --calibrate-sim
# Paste the printed matrix into homography_pix2world.py → _H_SIM (and update
# replay_bug.py → _H_SIM to match).
from homography_pix2world import (
    _H_REAL, _H_REAL_INV,
    _H_SIM,  _H_SIM_INV,
)

# Calibration corners  (world X,Y)  →  real_pix
# Matches homography_pix2world.py 2026-09-23 mat-corner calibration.
# Order: TL, TR, BL, BR  (matches the click order in --calibrate-sim)
CAL_WORLD  = [(-0.330, 0.005), ( 0.400, 0.005), (-0.310, 0.380), ( 0.375, 0.400)]
CAL_LABELS = ["TL", "TR", "BL", "BR"]
CAL_REAL   = [( 13.4,  67.0), (631.5,  65.8), (  8.5, 404.9), (632.7, 400.0)]

def _world_to_pix(wx, wy, H_inv):
    pt  = np.array([[[float(wx), float(wy)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H_inv.astype(np.float32))
    return float(out[0,0,0]), float(out[0,0,1])

CAL_SIM = [_world_to_pix(wx, wy, _H_SIM_INV) for wx, wy in CAL_WORLD]

# ── Helpers ──────────────────────────────────────────────────────────────────
def p2w(col, row, H):
    pt  = np.array([[[float(col), float(row)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H.astype(np.float32))
    return float(out[0,0,0]), float(out[0,0,1])

def w2p(wx, wy, H_inv):
    return _world_to_pix(wx, wy, H_inv)

def in_frame(u, v, W, H, pad=40):
    return -pad < u < W+pad and -pad < v < H+pad

# ── Drawing ──────────────────────────────────────────────────────────────────
GRID_COLOR  = (45, 140, 45)
BOUND_COLOR = (0, 200, 255)
AXIS_COLOR  = (20, 20, 20)
TICK_COLOR  = (160, 210, 160)
CYL_COLOR   = (30, 210, 30)
TGT_COLOR   = (0, 200, 255)

def draw_grid(img, H_inv, gx, gy, W, H):
    for wy in gy:
        row_pts = []
        for wx in gx:
            u, v = w2p(wx, wy, H_inv)
            if in_frame(u, v, W, H):
                row_pts.append((int(round(u)), int(round(v))))
        for a, b in zip(row_pts, row_pts[1:]):
            cv2.line(img, a, b, GRID_COLOR, 1, cv2.LINE_AA)
    for wx in gx:
        col_pts = []
        for wy in gy:
            u, v = w2p(wx, wy, H_inv)
            if in_frame(u, v, W, H):
                col_pts.append((int(round(u)), int(round(v))))
        for a, b in zip(col_pts, col_pts[1:]):
            cv2.line(img, a, b, GRID_COLOR, 1, cv2.LINE_AA)
    wy_top = gy[0]
    for wx in gx:
        u, v = w2p(wx, wy_top, H_inv)
        if 5 < u < W-5 and 5 < v < H-5:
            cv2.circle(img, (int(round(u)), int(round(v))), 2, GRID_COLOR, -1, cv2.LINE_AA)
            if abs(round(wx*100) % 10) < 0.5:
                cv2.putText(img, f"{wx:.2f}", (int(u)-10, int(v)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.24, TICK_COLOR, 1, cv2.LINE_AA)
    wx_left = gx[0]
    for wy in gy:
        u, v = w2p(wx_left, wy, H_inv)
        if 5 < u < W-5 and 5 < v < H-5:
            cv2.circle(img, (int(round(u)), int(round(v))), 2, GRID_COLOR, -1, cv2.LINE_AA)
            if abs(round(wy*100) % 10) < 0.5:
                cv2.putText(img, f"Y={wy:.2f}", (int(u)+3, int(v)+4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.22, TICK_COLOR, 1, cv2.LINE_AA)

def draw_boundary(img, cal_pix, W, H):
    order = [0, 1, 3, 2]
    pts = np.array([[int(round(cal_pix[i][0])), int(round(cal_pix[i][1]))] for i in order], np.int32)
    cv2.polylines(img, [pts], True, BOUND_COLOR, 2, cv2.LINE_AA)
    offsets = [(-42,-6),(2,-6),(-42,10),(2,10)]
    for i, ((wx, wy), (u, v)) in enumerate(zip(CAL_WORLD, cal_pix)):
        u, v = int(round(u)), int(round(v))
        cv2.circle(img, (u,v), 5, BOUND_COLOR, -1, cv2.LINE_AA)
        ox, oy = offsets[i]
        lx, ly = max(2, min(u+ox, W-60)), max(8, min(v+oy, H-4))
        cv2.putText(img, f"({wx:.2f},{wy:.3f})", (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, BOUND_COLOR, 1, cv2.LINE_AA)

def draw_x_axis(img, H_inv, gy, W, H):
    pts = []
    for wy in np.linspace(gy[0]-0.02, gy[-1]+0.02, 80):
        u, v = w2p(0.0, wy, H_inv)
        if in_frame(u, v, W, H, pad=5):
            pts.append((int(round(u)), int(round(v))))
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, AXIS_COLOR, 1, cv2.LINE_AA)

def draw_marker(img, u, v, shape, color, label, W, H, line_color=(0,230,200)):
    u, v = int(round(u)), int(round(v))
    if shape == "circle":
        cv2.circle(img, (u,v), 12, color, -1, cv2.LINE_AA)
        cv2.circle(img, (u,v), 12, (255,255,255), 2, cv2.LINE_AA)
    else:
        sz = 10
        pts = np.array([[u,v-sz],[u+sz,v],[u,v+sz],[u-sz,v]], np.int32)
        cv2.fillPoly(img, [pts], color)
        cv2.polylines(img, [pts], True, (255,255,255), 2, cv2.LINE_AA)
    lx = u + 18 if u < W - 130 else u - 125
    ly = v + 4
    cv2.line(img, (u,v), (lx, v), line_color, 1, cv2.LINE_AA)
    cv2.putText(img, label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 220), 1, cv2.LINE_AA)

# ── Object detection on real frame ───────────────────────────────────────────
_DET_IMG_CX = 320

def detect_objects(img_bgr):
    """Detect green cylinder and yellow/orange cup in a REAL overhead image.

    Returns (cyl_pix, tgt_pix) where each is (col, row) or None.
    """
    img = img_bgr.copy()
    img[415:, 455:] = 0          # mask out robot arm area (bottom-right)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Target = yellow/orange cup
    m_tgt = cv2.inRange(hsv, np.array([15,50,50],np.uint8), np.array([38,255,255],np.uint8))
    # Cylinder = green
    m_cyl = cv2.inRange(hsv, np.array([35,40,40],np.uint8), np.array([85,255,255],np.uint8))

    # Spatial priors: cylinder usually left half, target usually right half.
    # But fall back to full-frame search if the prior half is empty.
    m_cyl_left  = m_cyl.copy(); m_cyl_left[:, _DET_IMG_CX:] = 0
    m_tgt_right = m_tgt.copy(); m_tgt_right[:, :_DET_IMG_CX] = 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))

    def best_mask(mask_primary, mask_fallback):
        m = cv2.morphologyEx(cv2.morphologyEx(mask_primary, cv2.MORPH_OPEN,k), cv2.MORPH_DILATE,k)
        cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts and max(cv2.contourArea(c) for c in cnts) >= 80:
            return m
        # Fall back to full-frame mask
        return cv2.morphologyEx(cv2.morphologyEx(mask_fallback, cv2.MORPH_OPEN,k), cv2.MORPH_DILATE,k)

    m_tgt = best_mask(m_tgt_right, m_tgt)
    m_cyl = best_mask(m_cyl_left,  m_cyl)

    def centroid(mask):
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 80: return None
        M = cv2.moments(c)
        if M["m00"]==0: return None
        return int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

    return centroid(m_cyl), centroid(m_tgt)

# ── Nearest X gridline ───────────────────────────────────────────────────────
def draw_nearest_x_gridline(img, wx_target, H_inv, gy, color, W, H):
    wx = round(wx_target / 0.05) * 0.05
    pts = []
    for wy in np.linspace(gy[0]-0.02, gy[-1]+0.02, 60):
        u, v = w2p(wx, wy, H_inv)
        if in_frame(u, v, W, H, pad=5):
            pts.append((int(round(u)), int(round(v))))
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, color, 1, cv2.LINE_AA)

# ── Interactive SIM calibration ───────────────────────────────────────────────
def calibrate_sim(sim_bgr):
    """
    Interactive calibration: user clicks the 4 mat corners (TL, TR, BL, BR)
    in the SIM image.  Computes and prints a new _H_SIM matrix.

    Corner order must match CAL_WORLD:
      TL = (-0.41,  0.00)   top-left  of mat
      TR = ( 0.34,  0.00)   top-right
      BL = (-0.385, 0.405)  bottom-left
      BR = ( 0.315, 0.380)  bottom-right

    The printed matrix can be pasted straight into the _H_SIM constant above.
    """
    W, H = 640, 480
    img = cv2.resize(sim_bgr, (W, H))
    clicks = []
    instructions = [
        "Click TL corner  (-0.330,  0.005)",
        "Click TR corner  ( 0.400,  0.005)",
        "Click BL corner  (-0.310,  0.380)",
        "Click BR corner  ( 0.375,  0.400)",
    ]
    corner_colors = [(0,0,255),(0,128,255),(255,0,0),(128,0,255)]

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            print(f"  [{len(clicks)}/4] {instructions[len(clicks)-1].split()[1]} → pixel ({x}, {y})")
            cv2.circle(disp, (x, y), 6, corner_colors[len(clicks)-1], -1)
            cv2.putText(disp, CAL_LABELS[len(clicks)-1], (x+8, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, corner_colors[len(clicks)-1], 2)
            cv2.imshow(WIN, disp)

    WIN = "SIM Calibration: click TL TR BL BR then press any key"
    disp = img.copy()
    cv2.putText(disp, instructions[0], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.namedWindow(WIN)
    cv2.imshow(WIN, disp)          # must call imshow BEFORE setMouseCallback so Qt realises the window
    cv2.waitKey(1)                 # pump the event loop so the window handle is valid
    cv2.setMouseCallback(WIN, mouse_cb)

    print("\n=== SIM CALIBRATION MODE ===")
    print("Window will open.  Click these 4 corners in order:")
    for i, (lbl, (wx,wy)) in enumerate(zip(CAL_LABELS, CAL_WORLD)):
        print(f"  {i+1}. {lbl}  world ({wx:+.3f}, {wy:.3f})  ← {instructions[i]}")
    print("After 4 clicks, press any key to compute.\n")

    while True:
        if len(clicks) < 4:
            cv2.putText(disp, instructions[len(clicks)], (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow(WIN, disp)
        key = cv2.waitKey(50)
        if key != -1 and len(clicks) == 4:
            break
    cv2.destroyAllWindows()

    if len(clicks) < 4:
        print("ERROR: fewer than 4 corners clicked — aborting calibration.")
        return None

    # _H_SIM convention: pixel → world  (same as _H_REAL, used by p2w() and _world_to_pix()).
    # So src = clicked pixels, dst = known world coords.
    # inv(_H_SIM) = _H_SIM_INV maps world → pixel (used by w2p / _world_to_pix).
    src = np.array([[u,  v ] for u,  v  in clicks   ], dtype=np.float64)  # pixel
    dst = np.array([[wx, wy] for wx, wy in CAL_WORLD], dtype=np.float64)  # world
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    if H is None:
        print("ERROR: findHomography failed — are the 4 points collinear?")
        return None

    print("\n=== NEW _H_SIM (paste into homography_pix2world.py AND replay_bug.py) ===")

    print("_H_SIM = np.array([")
    for row in H:
        formatted = ", ".join(f"{v:+.8e}" for v in row)
        print(f"    [{formatted}],")
    print("], dtype=np.float64)")

    # Reprojection error: H maps pixel→world, so inv(H) maps world→pixel.
    # _world_to_pix(wx, wy, inv(H)) projects a known world corner back to pixel
    # and compares with the clicked pixel.
    H_inv = np.linalg.inv(H)
    errs = []
    for (wx, wy), (pu, pv) in zip(CAL_WORLD, clicks):
        eu, ev = _world_to_pix(wx, wy, H_inv)
        errs.append(np.hypot(eu - pu, ev - pv))
    print(f"\nReprojection errors (px): {[f'{e:.2f}' for e in errs]}")
    print(f"Mean error: {np.mean(errs):.2f} px")
    return H

# ── Episode-mode helpers ──────────────────────────────────────────────────────
def _find_real_video(dataset_root, episode, cam):
    ep6 = f"{episode:06d}"
    for pat in [
        f"videos/{cam}/chunk-000/episode_{ep6}.mp4",
        f"videos/{cam}/chunk-{episode // 500:03d}/episode_{ep6}.mp4",
    ]:
        p = os.path.join(dataset_root, pat)
        if os.path.exists(p):
            return p, 0.0
    chunk_file = os.path.join(dataset_root, f"videos/{cam}/chunk-000/file-000.mp4")
    if os.path.exists(chunk_file):
        try:
            import pandas as _pd
            parq = os.path.join(dataset_root, "meta/episodes/chunk-000/file-000.parquet")
            ep_meta = _pd.read_parquet(parq)
            row = ep_meta[ep_meta["episode_index"] == episode]
            if not row.empty:
                col = f"videos/{cam}/from_timestamp"
                if col in row.columns:
                    return chunk_file, float(row[col].iloc[0])
        except Exception as e:
            print(f"  Warning (parquet lookup): {e}")
        return chunk_file, 0.0
    return None, 0.0


def _extract_frame(video_path, offset_s=0.0):
    import subprocess
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        raise RuntimeError(f"ffprobe failed on {video_path}: {probe.stderr.strip()}")
    w, h = map(int, probe.stdout.strip().split(","))
    cmd = ["ffmpeg", "-ss", str(offset_s), "-i", video_path,
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True)
    if len(r.stdout) < w * h * 3:
        raise RuntimeError(f"ffmpeg returned too few bytes for {video_path}")
    return np.frombuffer(r.stdout, dtype=np.uint8).reshape(h, w, 3).copy()


def _find_sim_frame(sim_dir, episode):
    # Primary: sim_overhead_ep{N}_frame5.png written by replay_interactive.py
    # Fallback: debug_sim_ep{N}_marker.png written by replay_bug.py
    patterns = [
        f"{sim_dir}/sim_overhead_ep{episode}_frame5.png",
        f"{sim_dir}/sim_overhead_ep{episode:03d}_frame5.png",
        f"{sim_dir}/sim_overhead_ep{episode}_frame1.png",
        f"{sim_dir}/sim_overhead_ep{episode:03d}_frame1.png",
        f"{sim_dir}/debug_sim_ep{episode:03d}_marker.png",
        f"{sim_dir}/debug_sim_ep{episode}_marker.png",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def _resolve_images(args):
    if args.episode is not None:
        ep  = args.episode
        ds  = os.path.expanduser(args.dataset)
        sd  = os.path.expanduser(args.sim_dir)
        cam = args.cam
        print(f"[episode mode] ep={ep}  dataset={ds}  sim-dir={sd}  cam={cam}")
        vid, offset_s = _find_real_video(ds, ep, cam)
        if vid:
            print(f"  REAL video : {vid}  (offset {offset_s:.3f}s)")
            real_bgr = _extract_frame(vid, offset_s)
        else:
            raise FileNotFoundError(
                f"Cannot find real camera video for episode {ep} in {ds} (cam={cam}).\n"
                f"  Try --dataset or --cam to point at the right location."
            )
        sim_png = _find_sim_frame(sd, ep)
        if sim_png:
            print(f"  SIM image  : {sim_png}")
            sim_bgr = cv2.imread(sim_png)
        else:
            raise FileNotFoundError(
                f"Cannot find SIM marker image for episode {ep} in {sd}.\n"
                f"  Expected: {sd}/debug_sim_ep{ep:03d}_marker.png\n"
                f"  Run replay_bug.py on this episode first."
            )
        out_path = args.output or os.path.join(sd, f"overlay_ep{ep:03d}.png")
        label    = f"Episode {ep}"
    else:
        if len(args.positional) < 2:
            print(__doc__)
            sys.exit(1)
        real_bgr = cv2.imread(args.positional[0])
        sim_bgr  = cv2.imread(args.positional[1])
        if real_bgr is None:
            raise FileNotFoundError(f"Cannot read {args.positional[0]}")
        if sim_bgr is None:
            raise FileNotFoundError(f"Cannot read {args.positional[1]}")
        out_path = (args.positional[2] if len(args.positional) > 2
                    else (args.output or os.path.join(
                        os.path.expanduser("~/so101-sim-project"), "dual_panel_overlay.png")))
        label    = None
    return real_bgr, sim_bgr, out_path, label


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="REAL + SIM side-by-side with world-grid overlay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("positional", nargs="*",
                        help="real.png sim.png [output.png]  (direct-image mode)")
    parser.add_argument("--episode", "-e", type=int, default=None)
    parser.add_argument("--frame",   "-f", type=int, default=0,
                        help="Frame index label (display only)")
    parser.add_argument("--dataset",
                        default="~/.cache/huggingface/lerobot/Screener2/V4_pnp_44_20260720_141907")
    parser.add_argument("--sim-dir", default="~/so101-sim-project")
    parser.add_argument("--cam",     default="observation.images.overhead")
    parser.add_argument("--output",  "-o", default=None)
    parser.add_argument("--calibrate-sim", action="store_true",
                        help="Interactive mode: click 4 mat corners in SIM image "
                             "to recompute _H_SIM.  Prints new matrix then exits.")

    args = parser.parse_args()
    real_bgr, sim_bgr, out_path, episode_label = _resolve_images(args)

    # ── Calibration mode — compute new _H_SIM and exit ────────────────────────
    if args.calibrate_sim:
        W, H = 640, 480
        calibrate_sim(cv2.resize(sim_bgr, (W, H)))
        print("\nPaste the matrix above into homography_pix2world.py → _H_SIM constant.")
        print("Also update replay_bug.py → _H_SIM to match.")
        print("Then re-run without --calibrate-sim to verify grid alignment.")
        return

    W, H = 640, 480
    real_img = cv2.resize(real_bgr, (W, H))
    sim_img  = cv2.resize(sim_bgr,  (W, H))

    gx = np.arange(-0.35, 0.405, 0.05)   # mat X: -0.330 to +0.400
    gy = np.arange( 0.00, 0.410, 0.05)   # mat Y:  0.005 to +0.400

    real_out = real_img.copy()
    sim_out  = sim_img.copy()

    draw_grid(real_out, _H_REAL_INV, gx, gy, W, H)
    draw_grid(sim_out,  _H_SIM_INV,  gx, gy, W, H)

    draw_boundary(real_out, CAL_REAL, W, H)
    draw_boundary(sim_out,  CAL_SIM,  W, H)

    draw_x_axis(real_out, _H_REAL_INV, gy, W, H)
    draw_x_axis(sim_out,  _H_SIM_INV,  gy, W, H)

    # ── Detect objects in REAL image ──────────────────────────────────────────
    cyl_pix, tgt_pix = detect_objects(real_img)
    cyl_world = p2w(*cyl_pix, _H_REAL) if cyl_pix else None
    tgt_world = p2w(*tgt_pix, _H_REAL) if tgt_pix else None

    print(f"Cylinder pixel: {cyl_pix}  → world: {cyl_world}")
    print(f"Target   pixel: {tgt_pix}  → world: {tgt_world}")

    # ── Draw X gridlines for each detected object ─────────────────────────────
    if cyl_world:
        draw_nearest_x_gridline(real_out, cyl_world[0], _H_REAL_INV, gy, (200,100,30), W, H)
        draw_nearest_x_gridline(sim_out,  cyl_world[0], _H_SIM_INV,  gy, (200,100,30), W, H)
    if tgt_world:
        draw_nearest_x_gridline(real_out, tgt_world[0], _H_REAL_INV, gy, (30,160,220), W, H)
        draw_nearest_x_gridline(sim_out,  tgt_world[0], _H_SIM_INV,  gy, (30,160,220), W, H)

    # ── Draw markers at identical world coords on both panels ─────────────────
    for world_pos, shape, color, prefix in [
        (cyl_world, "circle",  CYL_COLOR, "CYL"),
        (tgt_world, "diamond", TGT_COLOR, "TGT"),
    ]:
        if world_pos is None:
            continue
        wx, wy = world_pos
        label = f"{prefix}  ({wx:+.3f}, {wy:.3f})"
        for img, H_inv in [(real_out, _H_REAL_INV), (sim_out, _H_SIM_INV)]:
            u, v = w2p(wx, wy, H_inv)
            if in_frame(u, v, W, H, pad=20):
                draw_marker(img, u, v, shape, color, label, W, H)

    cv2.putText(real_out,
                "REAL camera  —  boundary from calibration corners | grid every 0.05 m",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220,220,220), 1, cv2.LINE_AA)
    cv2.putText(sim_out,
                "SIM camera  —  same world coords, same grid | CYL & TGT at identical (X,Y)",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220,220,220), 1, cv2.LINE_AA)

    GAP     = 12
    HDR_H   = 60
    LEG_H   = 42
    total_h = HDR_H + H + LEG_H
    total_w = W + GAP + W + 20
    canvas  = np.full((total_h, total_w, 3), 18, dtype=np.uint8)

    x0_real = 10
    x0_sim  = 10 + W + GAP
    y0      = HDR_H

    canvas[y0:y0+H, x0_real:x0_real+W] = real_out
    canvas[y0:y0+H, x0_sim :x0_sim +W] = sim_out
    canvas[y0:y0+H, x0_real+W:x0_real+W+GAP] = 40

    cx  = total_w // 2
    hdr = ("Step 2 -> 4  |  Same world coordinates placed on both grids"
           + (f"  |  {episode_label}" if episode_label else ""))
    cv2.putText(canvas, hdr,
                (cx-285, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220,220,220), 1, cv2.LINE_AA)
    cv2.putText(canvas,
                "Boundary lines connect the four calibration corner pixels exactly  *  Grid 0.05 m spacing",
                (cx-330, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170,170,170), 1, cv2.LINE_AA)

    ly = y0 + H + 22
    items = [
        (BOUND_COLOR,  "Mat boundary (calibration corners)"),
        (AXIS_COLOR,   "X = 0 axis"),
        ((30,160,220), "Nearest X gridline to TGT"),
        ((200,100,30), "Nearest X gridline to CYL"),
        (TGT_COLOR,    "Target (same world coord both panels)"),
        (CYL_COLOR,    "Cylinder (same world coord both panels)"),
    ]
    lx = 14
    for color, text in items:
        cv2.circle(canvas, (lx+5, ly), 5, color, -1)
        cv2.putText(canvas, text, (lx+13, ly+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.29, (160,160,160), 1, cv2.LINE_AA)
        lx += len(text)*6 + 26

    cv2.imwrite(out_path, canvas)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()