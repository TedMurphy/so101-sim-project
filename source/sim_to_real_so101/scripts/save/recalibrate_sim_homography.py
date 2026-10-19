#!/usr/bin/env python3
"""
recalibrate_sim_homography.py  (v3)
=====================================
Takes a dual_panel_overlay output image that already has yellow debug
spheres in the SIM panel (from replay_interactive_v5), detects those
sphere centroids, matches them to their known world-grid positions, and
recomputes a corrected _H_SIM homography.

v3 adds iterative refinement: after a first-pass match (guided by the old,
wrong H), the newly computed H is used to rematch all detected blobs.
This recovers points that were >70px away from old-H predictions but are
close to the true positions, lifting the inlier count from ~9 to ~35.

Usage:
    python recalibrate_sim_homography.py <overlay_image.png>
    python recalibrate_sim_homography.py <overlay_image.png> --show
    python recalibrate_sim_homography.py <overlay_image.png> --debug   # save extra debug image
"""

import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

# ── Known sphere grid (same as replay_interactive_v5) ────────────────────────
_GX = np.arange(-0.40, 0.36, 0.10)   # 8 world-X values
_GY = np.arange( 0.00, 0.42, 0.10)   # 5 world-Y values
GRID_WORLD = [(float(wx), float(wy)) for wx in _GX for wy in _GY]  # 40 pts

# ── Existing SIM calibration (used only as initial matching guide) ────────────
_CALIB_WORLD = np.array([
    [-0.410,  0.000],
    [ 0.340,  0.000],
    [-0.385,  0.405],
    [ 0.315,  0.380],
], dtype=np.float64)
_CALIB_PIX = np.array([
    [  5,  61],
    [636,  62],
    [ 24, 384],
    [612, 406],
], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
def _build_H(src, dst):
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H


def _project(H, pts_xy):
    """(N,2) world/pixel → (N,2) via 3×3 homography."""
    n = len(pts_xy)
    p = np.hstack([pts_xy, np.ones((n, 1))]).T
    q = (H @ p).T
    return q[:, :2] / q[:, 2:3]


# ─────────────────────────────────────────────────────────────────────────────
def detect_yellow_spheres(bgr, border_margin=25,
                           min_area=12, max_area=2000,
                           min_circularity=0.35):
    """
    Return (accepted, rejected) each as list of (cx, cy, area).

    Key filters
    -----------
    border_margin   : ignore blobs whose centroid is within this many pixels
                      of any image edge  (removes yellow frame + corner labels)
    min_circularity : 4π·area/perimeter²  ≈ 1 for a perfect circle;
                      text, lines, and the rectangular border frame score <0.2
    max_area        : drop very large blobs (e.g. merged border segments)
    """
    h, w = bgr.shape[:2]
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Yellow: two passes (tight then relaxed) — use whichever gives more hits
    masks = []
    for lo, hi in [
        ([18, 140, 140], [38, 255, 255]),   # tight
        ([12, 80,  80],  [45, 255, 255]),   # relaxed
    ]:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        masks.append(m)

    # Pick the mask with more contours
    cnts_list = []
    for m in masks:
        c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_list.append(c)
    cnts = max(cnts_list, key=len)

    accepted, rejected = [], []
    for c in cnts:
        a = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        circ  = (4 * np.pi * a / perim**2) if perim > 0 else 0
        M     = cv2.moments(c)
        if M['m00'] < 1:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        reason = None
        if a < min_area:           reason = f"area {a:.0f}<{min_area}"
        elif a > max_area:         reason = f"area {a:.0f}>{max_area}"
        elif circ < min_circularity: reason = f"circ {circ:.2f}<{min_circularity}"
        elif cx < border_margin:   reason = f"left-edge x={cx:.0f}"
        elif cx > w-border_margin: reason = f"right-edge x={cx:.0f}"
        elif cy < border_margin:   reason = f"top-edge y={cy:.0f}"
        elif cy > h-border_margin: reason = f"bottom-edge y={cy:.0f}"

        if reason:
            rejected.append((cx, cy, a, reason))
        else:
            accepted.append((cx, cy, a))

    return accepted, rejected


# ─────────────────────────────────────────────────────────────────────────────
def match_spheres(detected, grid_world, H_w2p, max_dist_px=70, min_matches=6):
    """
    Match detected (cx,cy,area) blobs to (world_x,world_y) grid positions.
    Uses existing homography as guide; falls back to grid-structure sort.
    """
    if len(detected) < min_matches:
        return []

    world_arr = np.array(grid_world, dtype=np.float64)
    proj      = _project(H_w2p, world_arr)
    det_xy    = np.array([[b[0], b[1]] for b in detected])

    matches   = []
    used_grid = set()

    # Largest blobs first (more confident)
    order = sorted(range(len(detected)), key=lambda i: -detected[i][2])
    for di in order:
        cx, cy = det_xy[di]
        dists  = np.sqrt(((proj - np.array([cx, cy]))**2).sum(axis=1))
        best_i = int(np.argmin(dists))
        if dists[best_i] < max_dist_px and best_i not in used_grid:
            matches.append((cx, cy, grid_world[best_i][0], grid_world[best_i][1]))
            used_grid.add(best_i)

    if len(matches) >= min_matches:
        return matches

    # ── Fallback: grid-structure sort ────────────────────────────────────────
    print(f"[WARN] Only {len(matches)} guided matches. Trying structure sort …")
    pts = det_xy.copy()

    # Pixel-x ascending ↔ world_X ascending (from SIM calibration orientation)
    sort_x = np.argsort(pts[:, 0])
    pts_sx = pts[sort_x]
    n_cols, n_rows = len(_GX), len(_GY)
    chunk  = max(1, len(pts_sx) // n_cols)
    matches2 = []
    for ci, wx in enumerate(_GX):
        col = pts_sx[ci*chunk:(ci+1)*chunk]
        col = col[np.argsort(col[:, 1])]
        for ri, wy in enumerate(_GY):
            if ri >= len(col):
                break
            matches2.append((col[ri, 0], col[ri, 1], float(wx), float(wy)))

    print(f"[INFO] Structure sort: {len(matches2)} tentative matches.")
    return matches2


# ─────────────────────────────────────────────────────────────────────────────
def compute_homographies(matches):
    pix   = np.array([[m[0], m[1]] for m in matches], dtype=np.float64)
    world = np.array([[m[2], m[3]] for m in matches], dtype=np.float64)
    H_p2w, mask = cv2.findHomography(pix, world, cv2.RANSAC, 0.012)
    H_w2p, _    = cv2.findHomography(world, pix, cv2.RANSAC, 2.0)
    n_in = int(mask.sum()) if mask is not None else 0
    print(f"[INFO] RANSAC: {len(matches)} matches → {n_in} inliers "
          f"({100*n_in/len(matches):.0f}%)")
    return H_p2w, H_w2p, mask.ravel().astype(bool) if mask is not None else np.ones(len(matches), bool)


# ─────────────────────────────────────────────────────────────────────────────
def print_calibration(H_p2w, H_w2p, matches, inliers):
    m_in = [m for m, ok in zip(matches, inliers) if ok]
    if m_in:
        pix_in   = np.array([[m[0], m[1]] for m in m_in])
        world_in = np.array([[m[2], m[3]] for m in m_in])
        rp       = _project(H_w2p, world_in)
        errs     = np.sqrt(((rp - pix_in)**2).sum(axis=1))
        print(f"[INFO] Reprojection error (inliers): "
              f"mean={errs.mean():.2f}px  max={errs.max():.2f}px")

    sep = "=" * 62
    print(f"\n{sep}")
    print("PASTE INTO dual_panel_overlay.py")
    print(sep)
    print("_H_SIM = np.array([  # SIM pixel → world  (auto-calibrated)")
    for r in H_p2w:
        print(f"    [{r[0]:+.8e}, {r[1]:+.8e}, {r[2]:+.8e}],")
    print("])")
    print()
    print("_H_SIM_INV = np.array([  # world → SIM pixel  (auto-calibrated)")
    for r in H_w2p:
        print(f"    [{r[0]:+.8e}, {r[1]:+.8e}, {r[2]:+.8e}],")
    print("])")

    print(f"\n{sep}")
    print("PASTE INTO homography_pix2world.py")
    print(sep)
    print("_H_SIM = np.array([  # SIM pixel → world  (auto-calibrated)")
    for r in H_p2w:
        print(f"    [{r[0]:+.8e}, {r[1]:+.8e}, {r[2]:+.8e}],")
    print("])")

    if m_in:
        dist_from_center = []
        pts = np.array([[m[0], m[1]] for m in m_in])
        ctr = pts.mean(axis=0)
        for i, m in enumerate(m_in):
            d = np.sqrt((m[0]-ctr[0])**2 + (m[1]-ctr[1])**2)
            dist_from_center.append((d, m))
        top4 = sorted(dist_from_center, reverse=True)[:4]
        print(f"\n{sep}")
        print("4 corner inlier correspondences (for manual verification):")
        print(sep)
        for _, m in top4:
            print(f"  world({m[2]:+.3f}, {m[3]:.3f})  →  SIM px ({m[0]:.0f}, {m[1]:.0f})")


# ─────────────────────────────────────────────────────────────────────────────
def save_debug_image(sim_panel, accepted, rejected, out_path):
    """Image showing every blob: green=accepted, red=rejected (with label)."""
    out = sim_panel.copy()
    for cx, cy, a, reason in rejected:
        cv2.circle(out, (int(cx), int(cy)), 5, (0, 0, 220), 1)
        cv2.putText(out, reason[:12], (int(cx)+4, int(cy)-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 200), 1)
    for cx, cy, a in accepted:
        cv2.circle(out, (int(cx), int(cy)), 6, (0, 220, 0), 2)
    cv2.putText(out, f"green={len(accepted)} accepted  red={len(rejected)} rejected",
                (4, out.shape[0]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (200, 200, 200), 1)
    cv2.imwrite(str(out_path), out)
    print(f"[INFO] Debug blob image → {out_path}")


def save_annotated(sim_panel, matches, inliers, H_old_w2p, H_new_w2p, out_path):
    out = sim_panel.copy()
    world_arr = np.array([[m[2], m[3]] for m in matches])
    proj_old  = _project(H_old_w2p, world_arr)
    proj_new  = _project(H_new_w2p, world_arr)

    for i, m in enumerate(matches):
        px, py = int(m[0]),            int(m[1])
        ox, oy = int(proj_old[i, 0]),  int(proj_old[i, 1])
        nx, ny = int(proj_new[i, 0]),  int(proj_new[i, 1])
        ok     = bool(inliers[i])

        cv2.drawMarker(out, (ox, oy), (255, 80, 0), cv2.MARKER_CROSS, 9, 1)
        cv2.circle(out, (nx, ny), 3, (255, 220, 0), 1)
        color = (0, 255, 0) if ok else (0, 220, 220)
        cv2.circle(out, (px, py), 5, color, -1)
        cv2.arrowedLine(out, (ox, oy), (px, py), (0, 80, 255), 1, tipLength=0.3)

    cv2.putText(out, "Blue+=oldH  Green.=inlier  Cyan.=outlier  Yellow=newH",
                (4, out.shape[0]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                (200, 200, 200), 1)
    cv2.imwrite(str(out_path), out)
    print(f"[INFO] Annotated image → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--show",  action="store_true")
    ap.add_argument("--debug", action="store_true",
                    help="Also save debug image showing all blob detections")
    ap.add_argument("--border",  type=int,   default=25,
                    help="Border exclusion margin in pixels (default 25)")
    ap.add_argument("--min-circ", type=float, default=0.35,
                    help="Min circularity 0-1 (default 0.35); raise to reject more non-circles")
    ap.add_argument("--max-dist", type=float, default=70,
                    help="Max matching distance px (default 70)")
    ap.add_argument("--min-area", type=float, default=12)
    ap.add_argument("--max-area", type=float, default=2000)
    args = ap.parse_args()

    img_path = Path(args.image)
    img = cv2.imread(str(img_path))
    if img is None:
        sys.exit(f"Cannot read: {img_path}")

    h, w = img.shape[:2]
    print(f"[INFO] Full image: {w}×{h}")
    sim = img[:, w//2:].copy()
    sh, sw = sim.shape[:2]
    print(f"[INFO] SIM panel: {sw}×{sh}")

    # ── Detect ────────────────────────────────────────────────────────────────
    accepted, rejected = detect_yellow_spheres(
        sim,
        border_margin=args.border,
        min_area=args.min_area,
        max_area=args.max_area,
        min_circularity=args.min_circ,
    )
    print(f"[INFO] Blobs accepted: {len(accepted)}  rejected: {len(rejected)}")
    for r in rejected:
        print(f"       rejected  cx={r[0]:.0f} cy={r[1]:.0f} "
              f"area={r[2]:.0f}  reason: {r[3]}")

    if args.debug:
        dbg_path = img_path.parent / f"debug_blobs_{img_path.name}"
        save_debug_image(sim, accepted, rejected, dbg_path)

    if len(accepted) < 4:
        print("[ERROR] Fewer than 4 accepted blobs.")
        print("  Try:  --min-circ 0.2  --border 15  --min-area 8")
        sys.exit(1)

    # ── Match + iterative refinement ──────────────────────────────────────────
    H_old_w2p = _build_H(_CALIB_WORLD, _CALIB_PIX)

    best_H_p2w, best_H_w2p, best_matches, best_inliers = None, None, [], []
    guide_H = H_old_w2p
    dist = args.max_dist

    for iteration in range(3):
        matches = match_spheres(accepted, GRID_WORLD, guide_H,
                                max_dist_px=dist)
        print(f"[INFO] Iter {iteration}: {len(matches)} matches (guide px-dist {dist:.0f})")
        if len(matches) < 4:
            break
        H_p2w, H_w2p, inliers = compute_homographies(matches)
        n_in = int(inliers.sum())
        if n_in > int(best_inliers.sum()) if len(best_inliers) else -1:
            best_H_p2w, best_H_w2p = H_p2w, H_w2p
            best_matches, best_inliers = matches, inliers
        if H_w2p is None:
            break
        guide_H = H_w2p   # use refined H for next round
        dist    = 12       # tight threshold now that H is good

    if best_H_p2w is None:
        sys.exit("[ERROR] Could not compute homography.")

    H_p2w, H_w2p, matches, inliers = best_H_p2w, best_H_w2p, best_matches, best_inliers
    print(f"[INFO] Best result: {len(matches)} matches, {int(inliers.sum())} inliers.")

    # ── Output ────────────────────────────────────────────────────────────────
    print_calibration(H_p2w, H_w2p, matches, inliers)

    out_path = img_path.parent / f"recal_{img_path.name}"
    save_annotated(sim, matches, inliers, H_old_w2p, H_w2p, out_path)

    if args.show:
        cv2.imshow("Recalibration", cv2.imread(str(out_path)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()