#!/usr/bin/env python3
"""
dual_panel_overlay.py  —  REAL + SIM side-by-side with world grid overlay.

Usage (direct images):
    python dual_panel_overlay.py real.png sim.png [output.png]

Usage (episode mode — finds frames automatically):
    python dual_panel_overlay.py --episode 7
    python dual_panel_overlay.py --episode 7 --dataset ~/.cache/huggingface/lerobot/Screener2/V4_pnp_44_20260720_141907
    python dual_panel_overlay.py --episode 7 --sim-dir ~/so101-sim-project
    python dual_panel_overlay.py --episode 7 --cam observation.images.overhead

Episode mode looks for:
  REAL : {dataset}/videos/observation.images.overhead/chunk-000/episode_{N:06d}.mp4
         (extracted via ffmpeg — handles AV1 encoding)
  SIM  : {sim-dir}/sim_overhead_ep{N}_frame1.png
         (saved by replay_interactive.py)
"""

import sys
import argparse
import glob
import os
import numpy as np
import cv2

# ── Homography matrices ───────────────────────────────────────────────────────
# REAL: 8-point calibration 2026-09-16
_H_REAL = np.array([
    [ 1.1827585786e-03, -7.8665392145e-07, -4.3602353952e-01],
    [-4.6641202763e-05,  1.2275123725e-03, -6.1113389251e-02],
    [-1.4354815502e-04,  2.4936532876e-04,  1.0000000000e+00],
], dtype=np.float64)

# SIM: auto-calibrated 2026-09-17 from yellow debug-sphere positions
# (38 blobs detected, 20 matched, 20 RANSAC inliers, mean reproj error 1.02 px)
_H_SIM = np.array([
    [+1.17813735e-03, -8.13776072e-07, -4.48785352e-01],
    [-1.78080398e-06, +1.18938379e-03, -4.96008348e-02],
    [-8.14074686e-06, +1.83409923e-05, +1.00000000e+00],
], dtype=np.float64)

_H_REAL_INV = np.linalg.inv(_H_REAL)
_H_SIM_INV  = np.linalg.inv(_H_SIM)

# Calibration corners  (world X,Y)  →  real_pix
# SIM pixels derived from _H_SIM_INV so the boundary stays consistent
# with the grid (not hardcoded to old manual measurements).
CAL_WORLD  = [(-0.410, 0.000), ( 0.340, 0.000), (-0.385, 0.405), ( 0.315, 0.380)]
CAL_LABELS = ["TL", "TR", "BL", "BR"]
CAL_REAL   = [( 13.0,  67.0), (632.0,  66.0), (  9.0, 405.0), (633.0, 400.0)]

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
    img = img_bgr.copy()
    img[415:, 455:] = 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m_tgt = cv2.inRange(hsv, np.array([15,50,50],np.uint8), np.array([38,255,255],np.uint8))
    m_cyl = cv2.inRange(hsv, np.array([35,40,40],np.uint8), np.array([85,255,255],np.uint8))
    m_cyl[:, _DET_IMG_CX:] = 0
    m_tgt[:, :_DET_IMG_CX] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    m_tgt = cv2.morphologyEx(cv2.morphologyEx(m_tgt, cv2.MORPH_OPEN,k), cv2.MORPH_DILATE,k)
    m_cyl = cv2.morphologyEx(cv2.morphologyEx(m_cyl, cv2.MORPH_OPEN,k), cv2.MORPH_DILATE,k)
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
    patterns = [
        f"{sim_dir}/sim_overhead_ep{episode}_frame5.png",
        f"{sim_dir}/sim_overhead_ep{episode}_frame1.png",
        f"{sim_dir}/sim_overhead_ep{episode:06d}_frame5.png",
        f"{sim_dir}/sim_overhead_ep{episode:06d}_frame1.png",
        f"{sim_dir}/sim_overhead_ep{episode}_frame0.png",
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
                f"Cannot find sim overhead frame for episode {ep} in {sd}.\n"
                f"  Expected: {sd}/sim_overhead_ep{ep}_frame1.png\n"
                f"  Run replay_interactive.py on this episode first."
            )
        out_path = args.output or os.path.join(sd, f"overlay_ep{ep:03d}.png")
        label    = f"Episode {ep} — frame {args.frame}"
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
        label = None
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

    args = parser.parse_args()
    real_bgr, sim_bgr, out_path, episode_label = _resolve_images(args)

    W, H = 640, 480
    real_img = cv2.resize(real_bgr, (W, H))
    sim_img  = cv2.resize(sim_bgr,  (W, H))

    gx = np.arange(-0.40, 0.355, 0.05)
    gy = np.arange( 0.00, 0.415, 0.05)

    real_out = real_img.copy()
    sim_out  = sim_img.copy()

    draw_grid(real_out, _H_REAL_INV, gx, gy, W, H)
    draw_grid(sim_out,  _H_SIM_INV,  gx, gy, W, H)

    draw_boundary(real_out, CAL_REAL, W, H)
    draw_boundary(sim_out,  CAL_SIM,  W, H)

    draw_x_axis(real_out, _H_REAL_INV, gy, W, H)
    draw_x_axis(sim_out,  _H_SIM_INV,  gy, W, H)

    cyl_pix, tgt_pix = detect_objects(real_img)
    cyl_world = p2w(*cyl_pix, _H_REAL) if cyl_pix else None
    tgt_world = p2w(*tgt_pix, _H_REAL) if tgt_pix else None

    print(f"Cylinder pixel: {cyl_pix}  → world: {cyl_world}")
    print(f"Target   pixel: {tgt_pix}  → world: {tgt_world}")

    if cyl_world:
        draw_nearest_x_gridline(real_out, cyl_world[0], _H_REAL_INV, gy, (200,100,30), W, H)
        draw_nearest_x_gridline(sim_out,  cyl_world[0], _H_SIM_INV,  gy, (200,100,30), W, H)
    if tgt_world:
        draw_nearest_x_gridline(real_out, tgt_world[0], _H_REAL_INV, gy, (30,160,220), W, H)
        draw_nearest_x_gridline(sim_out,  tgt_world[0], _H_SIM_INV,  gy, (30,160,220), W, H)

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