# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
replay_diag_cup.py  —  H_REAL ↔ Isaac alignment diagnostic for the TARGET (cup).

Same structure as replay_diag_v2.py, but the TARGET is swept across the grid
and the cylinder is parked at a fixed reference position.

Detection strategy: FRAME DIFFERENCING.
  For each grid position:
    a. Place cup at (ix, iy), step 3×, capture frame_ON
    b. Move cup off-mat to (2.0, 2.0, 0.060), step 1×, capture frame_OFF
    c. diff = |frame_ON − frame_OFF|; find largest changed blob on the mat
    d. Move cup back to (ix, iy) for next iteration baseline consistency
  This is colour-agnostic and immune to fixed false positives.

For every H_REAL grid point:
  1. Converts H_REAL → Isaac:  ix = -hx,  iy = hy
  2. Writes target (cup) via write_root_state_to_sim + env.step()
  3. Cylinder parked at fixed H_REAL reference
  4. Frame-diffs ON vs OFF to find the cup blob
  5. Annotates each frame:
       orange  ×  = commanded marker  (world2pix_sim applied to H_REAL coords)
       orange  ○  = detected target blob pixel (from diff)
       yellow  →  = error vector
  6. Saves annotated PNG per position
  7. Saves summary figure + CSV

Run with:
  ./isaaclab.sh -p ~/so101-sim-project/source/sim_to_real_so101/scripts/replay_diag_cup.py \
                --enable_cameras --livestream 2
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cup/target H_REAL ↔ Isaac alignment diagnostic.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([
    '--livestream', '2',
    '--kit_args',
    '--ext-folder /home/ted/isaac-sim/extsDeprecated --enable isaacsim.core.utils',
])
args_cli.enable_cameras = True
args_cli.device = 'cpu'

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ────────────────────────────────────────────────────────
import os, csv
import torch
import numpy as np
import cv2
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import isaaclab_tasks                           # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks                  # noqa: F401

# ── Output directory ───────────────────────────────────────────────────────────
OUT_DIR = os.path.expanduser("~/so101-sim-project/diag_cup_frames")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Homography matrices ────────────────────────────────────────────────────────
_H_SIM = np.array([
    [ 1.2180386022e-03,  6.3913248179e-06, -4.1877764282e-01],
    [-2.5882012703e-05,  1.2027600102e-03, -6.5363631468e-02],
    [ 9.2160833913e-05, -1.4465568194e-05,  1.0000000000e+00],
], dtype=np.float64)
_H_SIM_INV = np.linalg.inv(_H_SIM)


def pix2world_sim(u, v):
    p = _H_SIM @ np.array([u, v, 1.0])
    return float(p[0]/p[2]), float(p[1]/p[2])


def world2pix_sim(wx, wy):
    p = _H_SIM_INV @ np.array([wx, wy, 1.0])
    return float(p[0]/p[2]), float(p[1]/p[2])


# ── Object placement helpers ──────────────────────────────────────────────────

def write_object(env, obj_name, isaac_x, isaac_y, isaac_z):
    device = env.unwrapped.device
    obj    = env.unwrapped.scene[obj_name]
    state  = obj.data.root_state_w.clone()
    state[:, :3]  = torch.tensor([[isaac_x, isaac_y, isaac_z]],
                                 dtype=torch.float32, device=device)
    state[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    state[:, 7:]  = 0.0
    obj.write_root_state_to_sim(state)


def hold_step(env):
    robot  = env.unwrapped.scene["robot"]
    action = robot.data.joint_pos[0].clone().unsqueeze(0)
    return env.step(action)


def get_rgb(env):
    rgb   = env.unwrapped.scene["camera_overhead"].data.output["rgb"]
    frame = rgb[0].cpu().numpy()
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame.astype(np.uint8)


# ── Frame-difference cup detector ─────────────────────────────────────────────
# Strategy: snap frame WITH cup, move cup off-mat, snap frame WITHOUT cup, diff.
# The largest changed region IS the cup.  Colour-agnostic.

# Off-mat parking position (outside camera view, Isaac coords)
_OFF_MAT_ISAAC = (3.0, 3.0, 0.060)


def _largest_blob_in_mask(mask, min_area=80):
    """Return (area, cx, cy) for the largest contour in mask, or None."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx_c = M['m10'] / M['m00']
        cy_c = M['m01'] / M['m00']
        if best is None or area > best[0]:
            best = (area, cx_c, cy_c)
    return best


def detect_cup_by_diff(frame_on, frame_off,
                       diff_thresh=25, min_area=80,
                       exclude_green=True):
    """Detect the cup by frame-differencing frame_on (cup present) vs frame_off (cup absent).

    Args:
        frame_on   : (H,W,3) uint8 RGB — cup on mat
        frame_off  : (H,W,3) uint8 RGB — cup parked off-mat
        diff_thresh: pixel-channel absolute difference threshold
        min_area   : minimum blob area in pixels
        exclude_green: mask out green pixels (cylinder) before diffing

    Returns (cx, cy) of the largest changed region, or None.
    Returned pixel coords are in the frame_on image space.
    """
    H, W = frame_on.shape[:2]

    # Absolute per-channel diff, max across channels
    diff = cv2.absdiff(frame_on, frame_off)   # (H,W,3) uint8
    diff_max = diff.max(axis=2)               # (H,W) uint8

    # Threshold to binary mask
    _, mask = cv2.threshold(diff_max, diff_thresh, 255, cv2.THRESH_BINARY)

    # Exclude arm region (top-centre)
    mask[:int(H * 0.45), int(W * 0.20):int(W * 0.80)] = 0

    # Optionally exclude green pixels (cylinder blob) from diff mask
    if exclude_green:
        hsv = cv2.cvtColor(frame_on, cv2.COLOR_RGB2HSV)
        green_mask = cv2.inRange(hsv, np.array([40, 60, 60]), np.array([100, 255, 255]))
        mask[green_mask > 0] = 0

    blob = _largest_blob_in_mask(mask.astype(np.uint8), min_area=min_area)
    if blob is None:
        return None, mask
    return (blob[1], blob[2]), mask   # (cx, cy), diff_mask


def detect_cylinder_blob(rgb_frame):
    H, W = rgb_frame.shape[:2]
    hsv  = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([40, 60, 60]), np.array([100, 255, 255]))
    mask[:int(H*0.45), int(W*0.20):int(W*0.80)] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    b = _largest_blob_in_mask(mask, min_area=30)
    return (b[1], b[2]) if b else None


# ── Per-frame annotation ───────────────────────────────────────────────────────

def annotate_frame(frame, hx, hy, cup_blob, cyl_hx_ref, cyl_hy_ref, cyl_blob,
                   diff_mask=None):
    """Annotate a SIM frame for the cup/target diagnostic.

    hx, hy       : H_REAL commanded position for target
    cup_blob     : (cx, cy) from frame-diff detection, or None
    cyl_hx_ref   : fixed cylinder H_REAL ref
    cyl_hy_ref   : fixed cylinder H_REAL ref
    cyl_blob     : (cx, cy) cylinder blob or None (for reference)
    diff_mask    : optional (H,W) uint8 diff mask to overlay (tinted cyan)
    """
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    sz  = 14

    # ── Overlay diff mask (tinted cyan) ───────────────────────────────────────
    if diff_mask is not None:
        tint = np.zeros_like(img)
        tint[diff_mask > 0] = (200, 200, 0)   # cyan in BGR
        img = cv2.addWeighted(img, 1.0, tint, 0.35, 0)

    # ── Target commanded marker (orange ×) ────────────────────────────────────
    cmd_u, cmd_v = world2pix_sim(hx, hy)
    ci, cj = int(round(cmd_u)), int(round(cmd_v))
    cv2.line(img, (ci-sz, cj-sz), (ci+sz, cj+sz), (0, 128, 255), 2)
    cv2.line(img, (ci+sz, cj-sz), (ci-sz, cj+sz), (0, 128, 255), 2)
    cv2.putText(img, f"CUP CMD ({hx:+.3f},{hy:+.3f})",
                (ci+sz+4, cj-2), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (0, 200, 255), 1, cv2.LINE_AA)

    # ── Target actual blob from diff (orange ○) ───────────────────────────────
    err_x_cm = err_y_cm = None
    if cup_blob is not None:
        tpx, tpy = cup_blob
        ti, tj = int(round(tpx)), int(round(tpy))
        cv2.circle(img, (ti, tj), 13, (0, 165, 255), 2)
        cv2.circle(img, (ti, tj),  3, (0, 165, 255), -1)
        cv2.arrowedLine(img, (ci, cj), (ti, tj), (0, 220, 255), 2, tipLength=0.3)
        bwx, bwy = pix2world_sim(tpx, tpy)
        err_x_cm = (bwx - hx) * 100
        err_y_cm = (bwy - hy) * 100
        err_m    = np.sqrt(err_x_cm**2 + err_y_cm**2)
        cv2.putText(img,
                    f"BLOB ({bwx:+.3f},{bwy:+.3f}) err={err_m:.1f}cm [diff]",
                    (ti+14, tj+4), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (0, 200, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "CUP: no diff blob", (ci+sz+4, cj+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1, cv2.LINE_AA)

    # ── Cylinder reference marker (blue ×) ────────────────────────────────────
    rcu, rcv = world2pix_sim(cyl_hx_ref, cyl_hy_ref)
    ri, rj = int(round(rcu)), int(round(rcv))
    cv2.line(img, (ri-sz, rj-sz), (ri+sz, rj+sz), (255, 80, 0), 2)
    cv2.line(img, (ri+sz, rj-sz), (ri-sz, rj+sz), (255, 80, 0), 2)
    cv2.putText(img, f"CYL ref ({cyl_hx_ref:+.3f},{cyl_hy_ref:+.3f})",
                (ri+sz+4, rj-2), cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                (255, 150, 50), 1, cv2.LINE_AA)
    if cyl_blob is not None:
        cv2.circle(img, (int(round(cyl_blob[0])), int(round(cyl_blob[1]))),
                   10, (0, 220, 0), 2)

    # ── Legend bar ─────────────────────────────────────────────────────────────
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (640, 32), (20, 20, 40), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    leg = (f"CUP_CMD:({hx:+.3f},{hy:+.3f})  err:({err_x_cm:+.1f},{err_y_cm:+.1f})cm"
           if err_x_cm is not None else
           f"CUP_CMD:({hx:+.3f},{hy:+.3f})  err:no-blob")
    cv2.putText(img, leg, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (220, 220, 255), 1, cv2.LINE_AA)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Main ───────────────────────────────────────────────────────────────────────

# Fixed cylinder reference (episode 7 cylinder H_REAL position)
CYL_HX_REF = -0.314
CYL_HY_REF = +0.346


def main():
    env_cfg = parse_env_cfg(
        "Lerobot-So101-Teleop-Cylinder-To-Target",
        device='cpu', num_envs=1,
    )
    env = gym.make("Lerobot-So101-Teleop-Cylinder-To-Target", cfg=env_cfg)
    env.reset()
    for _ in range(15):
        simulation_app.update()

    print("[CUP_DIAG] Environment ready.")
    print(f"[CUP_DIAG] Cylinder parked at H_REAL ({CYL_HX_REF:+.3f}, {CYL_HY_REF:+.3f})")
    print()

    # Park cylinder at fixed reference (Isaac: ix = -CYL_HX_REF, iy = CYL_HY_REF)
    write_object(env, "cylinder", -CYL_HX_REF, CYL_HY_REF, 0.050)
    # Will flush on first env.step below

    # ── Grid in H_REAL world coords ────────────────────────────────────────────
    grid_hx = np.arange(-0.36, 0.30, 0.08)
    grid_hy = np.arange(0.04,  0.42, 0.08)
    test_pts = [(float(hx), float(hy)) for hx in grid_hx for hy in grid_hy]

    print(f"[CUP_DIAG] {len(test_pts)} positions  "
          f"({len(grid_hx)} cols × {len(grid_hy)} rows)")
    print()

    results = []

    for idx, (hx, hy) in enumerate(test_pts):
        ix, iy = -hx, hy

        # ── Step A: place cup at target position, settle, capture frame_on ───
        write_object(env, "target", ix, iy, 0.060)
        hold_step(env)   # flush write to PhysX
        hold_step(env)   # settle
        hold_step(env)   # settle
        simulation_app.update()

        # Read back scene state (Isaac write is always exact)
        tgt_state = env.unwrapped.scene["target"].data.root_state_w[0]
        scene_ix  = float(tgt_state[0])
        scene_iy  = float(tgt_state[1])
        scene_iz  = float(tgt_state[2])
        scene_hx  = -scene_ix
        scene_hy  =  scene_iy

        frame_on  = get_rgb(env)
        cyl_blob  = detect_cylinder_blob(frame_on)

        # ── Step B: park cup off-mat, capture frame_off ───────────────────────
        write_object(env, "target", *_OFF_MAT_ISAAC)
        hold_step(env)   # flush
        hold_step(env)   # settle
        simulation_app.update()
        frame_off = get_rgb(env)

        # ── Step C: diff → find cup blob ──────────────────────────────────────
        cup_blob, diff_mask = detect_cup_by_diff(frame_on, frame_off)

        # Commanded marker pixel
        cmd_px, cmd_py = world2pix_sim(hx, hy)

        # Blob → H_SIM world
        if cup_blob is not None:
            tpx, tpy         = cup_blob
            blob_hx, blob_hy = pix2world_sim(tpx, tpy)
            err_blob_x = (blob_hx - hx) * 100
            err_blob_y = (blob_hy - hy) * 100
            tgt_detect = 'diff'
        else:
            tpx = tpy = blob_hx = blob_hy = float('nan')
            err_blob_x = err_blob_y = float('nan')
            tgt_detect = 'miss'

        err_scene_x = (scene_hx - hx) * 100
        err_scene_y = (scene_hy - hy) * 100

        row = dict(
            hx=round(hx,4), hy=round(hy,4),
            ix=round(ix,4), iy=round(iy,4),
            cmd_px=round(cmd_px,1), cmd_py=round(cmd_py,1),
            scene_hx=round(scene_hx,4), scene_hy=round(scene_hy,4), scene_iz=round(scene_iz,4),
            tgt_detect=tgt_detect,
            tpx=round(tpx,1) if tgt_detect!='miss' else float('nan'),
            tpy=round(tpy,1) if tgt_detect!='miss' else float('nan'),
            blob_hx=round(blob_hx,4) if tgt_detect!='miss' else float('nan'),
            blob_hy=round(blob_hy,4) if tgt_detect!='miss' else float('nan'),
            err_scene_x=round(err_scene_x,2),
            err_scene_y=round(err_scene_y,2),
            err_blob_x=round(err_blob_x,2) if tgt_detect!='miss' else float('nan'),
            err_blob_y=round(err_blob_y,2) if tgt_detect!='miss' else float('nan'),
        )
        results.append(row)

        # Annotate and save per-frame PNG (frame_on as background)
        ann = annotate_frame(frame_on, hx, hy, cup_blob,
                             CYL_HX_REF, CYL_HY_REF, cyl_blob,
                             diff_mask=diff_mask)
        frame_path = os.path.join(OUT_DIR, f"frame_{idx:03d}_{hx:+.3f}_{hy:+.3f}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))

        err_str = (f"({err_blob_x:+.1f},{err_blob_y:+.1f})cm"
                   if tgt_detect != 'miss' else "no-diff-blob")
        print(f"[CUP_DIAG] {idx+1:3d}/{len(test_pts)}  "
              f"H_REAL ({hx:+.3f},{hy:+.3f}) → "
              f"Isaac ({ix:+.3f},{iy:+.3f})  "
              f"detect={tgt_detect}  "
              f"blob_err={err_str}  "
              f"scene_err=({err_scene_x:+.1f},{err_scene_y:+.1f})cm")

    print()
    print("[CUP_DIAG] Sweep complete. Building plots...")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.expanduser("~/so101-sim-project/cup_diag.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"[CUP_DIAG] CSV → {csv_path}")

    # ── Summary figure ─────────────────────────────────────────────────────────
    env.reset()
    for _ in range(8):
        hold_step(env)
    bg = get_rgb(env)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor('#0d0d1a')

    # ── Left panel: SIM camera ─────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.imshow(bg)
    ax0.set_title(
        'SIM camera — CUP/TARGET diagnostic\n'
        'orange × = H_SIM prediction of H_REAL coord  (where it SHOULD appear)\n'
        'orange ○ = actual target blob  (where it ACTUALLY appears)\n'
        'green line = right-2/5 search ROI boundary',
        color='white', fontsize=10,
    )
    ax0.axis('off')

    for r in results:
        hx, hy = r['hx'], r['hy']
        cu, cv = world2pix_sim(hx, hy)
        ax0.plot(cu, cv, 'x', color='orange', ms=8, mew=2, zorder=5)
        if r['tgt_detect'] != 'miss' and not np.isnan(r['tpx']):
            ax0.plot(r['tpx'], r['tpy'], 'o', color='tomato', ms=5,
                     mec='white', mew=0.8, zorder=6)
            ax0.annotate('', xy=(r['tpx'], r['tpy']), xytext=(cu, cv),
                         arrowprops=dict(arrowstyle='->', color='yellow', lw=1.0))
            err = np.sqrt(r['err_blob_x']**2 + r['err_blob_y']**2) if (
                not np.isnan(r['err_blob_x'])) else 0.0
            if err > 3.0:
                ax0.text((cu+r['tpx'])/2 + 3, (cv+r['tpy'])/2,
                         f"{err:.0f}cm", fontsize=6, color='yellow',
                         path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

    # Cylinder reference
    ccu, ccv = world2pix_sim(CYL_HX_REF, CYL_HY_REF)
    ax0.plot(ccu, ccv, 'x', color='deepskyblue', ms=10, mew=2.5, zorder=8)
    ax0.text(ccu+10, ccv-5, f"CYL ref ({CYL_HX_REF:.3f},{CYL_HY_REF:.3f})",
             color='deepskyblue', fontsize=7, zorder=9)

    ax0.plot([], [], 'x',  color='orange',      ms=8, mew=2,      label='CUP commanded (H_SIM proj.)')
    ax0.plot([], [], 'o',  color='tomato',      ms=5, mec='white', label='CUP blob detected')
    ax0.plot([], [], '-',  color='yellow',      lw=1,              label='Error vector')
    ax0.plot([], [], 'x',  color='deepskyblue', ms=8, mew=2,      label='CYL parked (reference)')
    ax0.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white',
               labelcolor='white', fontsize=7)

    # ── Right panel: world-coord scatter ──────────────────────────────────────
    ax1 = axes[1]
    ax1.set_facecolor('#111122')
    ax1.set_aspect('equal')

    for r in results:
        hx, hy = r['hx'], r['hy']
        ax1.plot(hx, hy, 'x', color='orange', ms=9, mew=2.0, zorder=5)
        if r['tgt_detect'] != 'miss' and not np.isnan(r['blob_hx']):
            ax1.plot(r['blob_hx'], r['blob_hy'], 'o',
                     color='tomato', ms=6, mec='white', mew=0.8, zorder=6)
            dx, dy = r['blob_hx'] - hx, r['blob_hy'] - hy
            if abs(dx) > 0.005 or abs(dy) > 0.005:
                ax1.annotate('', xy=(r['blob_hx'], r['blob_hy']), xytext=(hx, hy),
                             arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))
            err = np.sqrt(dx**2 + dy**2) * 100
            if err > 2.0:
                ax1.text((hx+r['blob_hx'])/2 + 0.004, (hy+r['blob_hy'])/2,
                         f'{err:.0f}cm', fontsize=6, color='yellow',
                         path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # Cylinder parked reference
    ax1.plot(CYL_HX_REF, CYL_HY_REF, 'D', color='deepskyblue', ms=10,
             mec='white', mew=1, zorder=8, label=f'CYL parked ({CYL_HX_REF:.3f},{CYL_HY_REF:.3f})')

    # Episode 7 target reference
    ax1.plot(0.254, 0.054, '*', color='orange', ms=14,
             mec='white', mew=1, zorder=8, label='Ep7 TGT H_REAL (+0.254, +0.054)')

    bx = [-0.41, 0.34, 0.34, -0.41, -0.41]
    by = [ 0.00, 0.00, 0.41,  0.41,  0.00]
    ax1.plot(bx, by, '-', color='cyan', lw=1.5, alpha=0.6, label='Mat boundary')
    ax1.axvline(0, color='white', lw=0.8, alpha=0.4, ls='--')
    ax1.axhline(0, color='white', lw=0.8, alpha=0.4, ls='--')

    ax1.plot([], [], 'x', color='orange', ms=9, mew=2,       label='CUP H_REAL commanded')
    ax1.plot([], [], 'o', color='tomato', ms=6, mec='white', label='CUP blob → H_SIM world')
    ax1.plot([], [], '-', color='yellow', lw=1.5,             label='H_REAL↔H_SIM offset')

    ax1.set_xlabel('World X (m)  [H_REAL / H_SIM, +right in image]', color='white')
    ax1.set_ylabel('World Y (m)  [H_REAL / H_SIM, +down in image]',  color='white')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444')

    # Error stats
    blob_errs = [(r['err_blob_x'], r['err_blob_y']) for r in results
                 if r['tgt_detect'] != 'miss' and not np.isnan(r['err_blob_x'])]
    if blob_errs:
        ex  = [e[0] for e in blob_errs]
        ey  = [e[1] for e in blob_errs]
        mag = [np.sqrt(e[0]**2 + e[1]**2) for e in blob_errs]
        stats_txt = (
            f"Blob vs H_REAL commanded (cm) — n={len(blob_errs)}\n"
            f"  X: mean={np.mean(ex):+.1f}  std={np.std(ex):.1f}  "
            f"min={min(ex):+.1f}  max={max(ex):+.1f}\n"
            f"  Y: mean={np.mean(ey):+.1f}  std={np.std(ey):.1f}  "
            f"min={min(ey):+.1f}  max={max(ey):+.1f}\n"
            f"  |err|: mean={np.mean(mag):.1f}  max={max(mag):.1f}"
        )
        ax1.text(0.01, 0.01, stats_txt, transform=ax1.transAxes,
                 fontsize=7, color='white', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                           edgecolor='#555', alpha=0.85))

    from collections import Counter
    det_counts = Counter(r['tgt_detect'] for r in results)
    det_txt = "Detection: " + ", ".join(f"{k}={v}" for k,v in det_counts.items())
    ax1.text(0.01, 0.99, det_txt, transform=ax1.transAxes,
             fontsize=7, color='#aaa', va='top')

    ax1.legend(facecolor='#1a1a3a', edgecolor='white', labelcolor='white', fontsize=7)
    ax1.set_title(
        'World coords  ×=H_REAL commanded  ○=blob→H_SIM\n'
        'Systematic drift = Isaac world ≠ H_REAL world (for target/cup)',
        color='white', fontsize=10,
    )

    plt.suptitle(
        'CUP / TARGET  H_REAL ↔ Isaac alignment diagnostic\n'
        '× = H_REAL coord projected into SIM camera via H_SIM  (where it SHOULD appear)\n'
        '○ = HSV blob detection  (where it ACTUALLY appears)\n'
        'Compare with cylinder diagnostic to confirm offset is shared',
        color='white', fontsize=11, fontweight='bold', y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_png = os.path.expanduser("~/so101-sim-project/cup_diag.png")
    plt.savefig(out_png, dpi=140, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"[CUP_DIAG] Summary PNG → {out_png}")

    # ── Print summary ──────────────────────────────────────────────────────────
    if blob_errs:
        print()
        print("=== Cup blob-vs-commanded error summary (cm) ===")
        print(f"  n detected : {len(blob_errs)}/{len(test_pts)}")
        print(f"  X error : mean={np.mean(ex):+.2f}  std={np.std(ex):.2f}")
        print(f"  Y error : mean={np.mean(ey):+.2f}  std={np.std(ey):.2f}")
        print(f"  |error| : mean={np.mean(mag):.2f}  max={max(mag):.2f}")
        print()
        print("  Detection breakdown:", dict(det_counts))
        print()
        print("  If cup offset ≈ cylinder offset → shared Isaac world ↔ H_REAL offset")
        print(f"  Estimated correction: dX={-np.mean(ex)/100:+.4f}m  dY={-np.mean(ey)/100:+.4f}m")
    else:
        print("[CUP_DIAG] No blobs detected by frame-diff.")
        print("  → The diff threshold may be too high, or the cup barely changes pixels.")
        print("  → Check per-frame PNGs in ~/so101-sim-project/diag_cup_frames/ —")
        print("    the cyan tint shows what changed between ON and OFF frames.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()