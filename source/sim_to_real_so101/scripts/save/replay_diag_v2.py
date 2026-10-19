# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
replay_diag_v2.py  —  H_REAL ↔ Isaac coordinate alignment diagnostic.

For every point on a grid expressed in H_REAL world coords:
  1. Converts H_REAL → Isaac:  ix = -hx,  iy = hy
  2. Writes cylinder there via write_root_state_to_sim + env.step()
  3. Also places target at a fixed H_REAL reference position
  4. Frame-diffs ON vs OFF to detect cylinder blob (colour-agnostic)
  5. Annotates each frame:
        blue  ×  = commanded marker  (world2pix_sim applied to H_REAL coords)
        green ○  = detected cylinder blob pixel (from diff)
        orange × = fixed target commanded marker
        orange ○ = target blob (HSV)
        yellow → = error vector
  6. Saves annotated PNG per grid position
  7. Saves a summary figure (SIM camera composite + world-coord scatter)
  8. Saves a CSV with all columns

Detection strategy: FRAME DIFFERENCING (same as replay_diag_cup.py).
  For each grid position:
    a. Place cylinder at (ix, iy), step 3×, capture frame_ON
    b. Move cylinder off-mat to (3.0, 3.0, 0.050), step 1×, capture frame_OFF
    c. diff = |frame_ON − frame_OFF|; find largest changed blob on the mat
    d. Colour-agnostic — immune to HSV tuning errors.

Run with:
  ./isaaclab.sh -p ~/so101-sim-project/source/sim_to_real_so101/scripts/replay_diag_v2.py \
                --enable_cameras --livestream 2
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="H_REAL ↔ Isaac alignment diagnostic.")
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
OUT_DIR = os.path.expanduser("~/so101-sim-project/diag_v2_frames")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Homography matrices (8-point calibration 2026-09-16) ─────────────────────
_H_REAL = np.array([
    [ 1.1827585786e-03, -7.8665392145e-07, -4.3602353952e-01],
    [-4.6641202763e-05,  1.2275123725e-03, -6.1113389251e-02],
    [-1.4354815502e-04,  2.4936532876e-04,  1.0000000000e+00],
], dtype=np.float64)

_H_SIM = np.array([
    [ 1.2180386022e-03,  6.3913248179e-06, -4.1877764282e-01],
    [-2.5882012703e-05,  1.2027600102e-03, -6.5363631468e-02],
    [ 9.2160833913e-05, -1.4465568194e-05,  1.0000000000e+00],
], dtype=np.float64)

_H_SIM_INV  = np.linalg.inv(_H_SIM)
_H_REAL_INV = np.linalg.inv(_H_REAL)


def pix2world_sim(u, v):
    """SIM pixel → H_SIM world (x, y) metres."""
    p = _H_SIM @ np.array([u, v, 1.0])
    return float(p[0]/p[2]), float(p[1]/p[2])


def world2pix_sim(wx, wy):
    """H_SIM world (x, y) → SIM pixel (u, v)."""
    p = _H_SIM_INV @ np.array([wx, wy, 1.0])
    return float(p[0]/p[2]), float(p[1]/p[2])


def pix2world_real(u, v):
    """REAL pixel → H_REAL world (x, y) metres."""
    p = _H_REAL @ np.array([u, v, 1.0])
    return float(p[0]/p[2]), float(p[1]/p[2])


def world2pix_real(wx, wy):
    """H_REAL world (x, y) → REAL pixel (u, v)."""
    p = _H_REAL_INV @ np.array([wx, wy, 1.0])
    return float(p[0]/p[2]), float(p[1]/p[2])


# ── Object placement helpers ──────────────────────────────────────────────────

def write_object(env, obj_name, isaac_x, isaac_y, isaac_z):
    """Write an object to Isaac (X,Y,Z) via write_root_state_to_sim."""
    device = env.unwrapped.device
    obj    = env.unwrapped.scene[obj_name]
    state  = obj.data.root_state_w.clone()
    state[:, :3]  = torch.tensor([[isaac_x, isaac_y, isaac_z]],
                                 dtype=torch.float32, device=device)
    state[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    state[:, 7:]  = 0.0
    obj.write_root_state_to_sim(state)


def hold_step(env):
    """env.step() holding current joint positions — flushes write_root_state_to_sim."""
    robot  = env.unwrapped.scene["robot"]
    action = robot.data.joint_pos[0].clone().unsqueeze(0)
    return env.step(action)


def get_rgb(env):
    """Return (H, W, 3) uint8 from the SIM overhead camera."""
    rgb   = env.unwrapped.scene["camera_overhead"].data.output["rgb"]
    frame = rgb[0].cpu().numpy()
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame.astype(np.uint8)


# ── Off-mat parking position ──────────────────────────────────────────────────
_OFF_MAT_ISAAC = (3.0, 3.0, 0.050)


# ── Frame-difference cylinder detector ───────────────────────────────────────

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


def detect_cylinder_by_diff(frame_on, frame_off, diff_thresh=25, min_area=80):
    """Detect cylinder by frame-differencing ON (cylinder present) vs OFF (cylinder parked).

    Args:
        frame_on   : (H,W,3) uint8 RGB — cylinder on mat
        frame_off  : (H,W,3) uint8 RGB — cylinder parked off-mat
        diff_thresh: pixel-channel absolute difference threshold
        min_area   : minimum blob area in pixels

    Returns ((cx, cy), diff_mask) — (cx,cy) is None if no blob found.
    """
    H, W = frame_on.shape[:2]

    diff     = cv2.absdiff(frame_on, frame_off)
    diff_max = diff.max(axis=2)

    _, mask = cv2.threshold(diff_max, diff_thresh, 255, cv2.THRESH_BINARY)

    # Exclude arm region (top-centre)
    mask[:int(H * 0.45), int(W * 0.20):int(W * 0.80)] = 0

    blob = _largest_blob_in_mask(mask.astype(np.uint8), min_area=min_area)
    if blob is None:
        return None, mask
    return (blob[1], blob[2]), mask


def detect_target_blob(rgb_frame):
    """Detect orange/yellow target blob via HSV (target stays put, no diff needed)."""
    H, W = rgb_frame.shape[:2]
    hsv  = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
    # Try orange first
    mask = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([25, 255, 255]))
    if mask.sum() == 0:
        mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < 30:
            continue
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx_c = M['m10'] / M['m00']
        cy_c = M['m01'] / M['m00']
        if best is None or area > best[0]:
            best = (area, cx_c, cy_c)
    return (best[1], best[2]) if best else None


# ── Per-frame annotation ───────────────────────────────────────────────────────

def annotate_frame(frame, hx, hy, cyl_blob, tgt_blob, tgt_hx_ref, tgt_hy_ref,
                   diff_mask=None):
    """Draw commanded markers, detected blobs, and error text onto a copy of frame."""
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    sz  = 14

    # ── Overlay diff mask (tinted green) ─────────────────────────────────────
    if diff_mask is not None:
        tint = np.zeros_like(img)
        tint[diff_mask > 0] = (0, 200, 0)
        img = cv2.addWeighted(img, 1.0, tint, 0.30, 0)

    # ── Cylinder commanded marker (blue ×) ───────────────────────────────────
    cmd_u, cmd_v = world2pix_sim(hx, hy)
    ci, cv_int = int(round(cmd_u)), int(round(cmd_v))
    cv2.line(img, (ci-sz, cv_int-sz), (ci+sz, cv_int+sz), (255, 80, 0), 2)
    cv2.line(img, (ci+sz, cv_int-sz), (ci-sz, cv_int+sz), (255, 80, 0), 2)
    cv2.putText(img,
                f"CMD ({hx:+.3f},{hy:+.3f})",
                (ci+sz+4, cv_int-2), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (255, 200, 100), 1, cv2.LINE_AA)

    # ── Cylinder actual blob from diff (green ○) ─────────────────────────────
    cyl_err_x_cm, cyl_err_y_cm = None, None
    if cyl_blob is not None:
        bpx, bpy = cyl_blob
        bi, bj = int(round(bpx)), int(round(bpy))
        cv2.circle(img, (bi, bj), 12, (0, 220, 0), 2)
        cv2.circle(img, (bi, bj),  3, (0, 220, 0), -1)
        cv2.arrowedLine(img, (ci, cv_int), (bi, bj), (0, 220, 255), 2, tipLength=0.3)
        blob_hx, blob_hy = pix2world_sim(bpx, bpy)
        cyl_err_x_cm = (blob_hx - hx) * 100
        cyl_err_y_cm = (blob_hy - hy) * 100
        err_m = np.sqrt(cyl_err_x_cm**2 + cyl_err_y_cm**2)
        cv2.putText(img,
                    f"BLOB ({blob_hx:+.3f},{blob_hy:+.3f}) err={err_m:.1f}cm [diff]",
                    (bi+14, bj+4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (100, 255, 100), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "CYL: no diff blob", (ci+sz+4, cv_int+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 100), 1, cv2.LINE_AA)

    # ── Target commanded marker (orange ×) ───────────────────────────────────
    tgt_cmd_u, tgt_cmd_v = world2pix_sim(tgt_hx_ref, tgt_hy_ref)
    ti, tj = int(round(tgt_cmd_u)), int(round(tgt_cmd_v))
    cv2.line(img, (ti-sz, tj-sz), (ti+sz, tj+sz), (0, 128, 255), 2)
    cv2.line(img, (ti+sz, tj-sz), (ti-sz, tj+sz), (0, 128, 255), 2)
    cv2.putText(img,
                f"TGT CMD ({tgt_hx_ref:+.3f},{tgt_hy_ref:+.3f})",
                (ti+sz+4, tj-2), cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                (0, 180, 255), 1, cv2.LINE_AA)

    # ── Target actual blob (orange ○, HSV) ───────────────────────────────────
    if tgt_blob is not None:
        tpx, tpy = tgt_blob
        tbi, tbj = int(round(tpx)), int(round(tpy))
        cv2.circle(img, (tbi, tbj), 12, (0, 165, 255), 2)
        cv2.circle(img, (tbi, tbj),  3, (0, 165, 255), -1)
        cv2.arrowedLine(img, (ti, tj), (tbi, tbj), (0, 100, 255), 1, tipLength=0.3)
        tgt_blob_hx, tgt_blob_hy = pix2world_sim(tpx, tpy)
        tgt_err_m = np.sqrt((tgt_blob_hx-tgt_hx_ref)**2 + (tgt_blob_hy-tgt_hy_ref)**2) * 100
        cv2.putText(img,
                    f"TGT BLOB err={tgt_err_m:.1f}cm",
                    (tbi+14, tbj+4), cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                    (0, 165, 255), 1, cv2.LINE_AA)

    # ── Legend bar ────────────────────────────────────────────────────────────
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (640, 32), (20, 20, 40), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    leg_txt = (f"CYL_CMD:{hx:+.3f},{hy:+.3f}  "
               f"CYL_err:({cyl_err_x_cm:+.1f},{cyl_err_y_cm:+.1f})cm [diff]"
               if cyl_err_x_cm is not None else
               f"CYL_CMD:{hx:+.3f},{hy:+.3f}  CYL_err:no-diff-blob")
    cv2.putText(img, leg_txt, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 255), 1, cv2.LINE_AA)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Main ───────────────────────────────────────────────────────────────────────

# Fixed target reference position in H_REAL world coords (episode 7 target)
TGT_HX_REF = +0.254
TGT_HY_REF = +0.054


def main():
    env_cfg = parse_env_cfg(
        "Lerobot-So101-Teleop-Cylinder-To-Target",
        device='cpu', num_envs=1,
    )
    env = gym.make("Lerobot-So101-Teleop-Cylinder-To-Target", cfg=env_cfg)
    env.reset()

    for _ in range(15):
        simulation_app.update()

    print("[DIAG_V2] Environment ready.")
    print(f"[DIAG_V2] Target fixed at H_REAL ({TGT_HX_REF:+.3f}, {TGT_HY_REF:+.3f})")
    print("[DIAG_V2] Cylinder detection: FRAME DIFFERENCING (colour-agnostic)")
    print()

    # Pre-place target at its fixed reference; flush on first step below
    tgt_ix = -TGT_HX_REF
    tgt_iy =  TGT_HY_REF
    write_object(env, "target", tgt_ix, tgt_iy, 0.060)

    # ── Grid in H_REAL world coords ────────────────────────────────────────────
    grid_hx = np.arange(-0.36, 0.30, 0.08)
    grid_hy = np.arange(0.04,  0.42, 0.08)
    test_pts = [(float(hx), float(hy)) for hx in grid_hx for hy in grid_hy]

    print(f"[DIAG_V2] {len(test_pts)} positions  "
          f"({len(grid_hx)} cols × {len(grid_hy)} rows)")
    print()

    results = []

    for idx, (hx, hy) in enumerate(test_pts):
        ix, iy = -hx, hy

        # ── Step A: place cylinder, settle, capture frame_on ─────────────────
        write_object(env, "cylinder", ix, iy, 0.050)
        hold_step(env)   # flush write to PhysX
        hold_step(env)   # settle
        hold_step(env)   # settle
        simulation_app.update()

        # Read back scene state
        cyl_state = env.unwrapped.scene["cylinder"].data.root_state_w[0]
        scene_ix  = float(cyl_state[0])
        scene_iy  = float(cyl_state[1])
        scene_iz  = float(cyl_state[2])
        scene_hx  = -scene_ix
        scene_hy  =  scene_iy

        frame_on  = get_rgb(env)
        tgt_blob  = detect_target_blob(frame_on)

        # ── Step B: park cylinder off-mat, capture frame_off ─────────────────
        write_object(env, "cylinder", *_OFF_MAT_ISAAC)
        hold_step(env)   # flush
        hold_step(env)   # settle
        simulation_app.update()
        frame_off = get_rgb(env)

        # ── Step C: diff → find cylinder blob ────────────────────────────────
        cyl_blob, diff_mask = detect_cylinder_by_diff(frame_on, frame_off)

        # Commanded marker pixel
        cmd_px, cmd_py = world2pix_sim(hx, hy)

        # Blob → H_SIM world
        if cyl_blob is not None:
            bpx, bpy         = cyl_blob
            blob_hx, blob_hy = pix2world_sim(bpx, bpy)
            cyl_detect       = 'diff'
        else:
            bpx, bpy         = float('nan'), float('nan')
            blob_hx, blob_hy = float('nan'), float('nan')
            cyl_detect       = 'miss'

        if tgt_blob is not None:
            tpx, tpy                 = tgt_blob
            tgt_blob_hx, tgt_blob_hy = pix2world_sim(tpx, tpy)
            tgt_detect               = 'ok'
        else:
            tpx, tpy                 = float('nan'), float('nan')
            tgt_blob_hx, tgt_blob_hy = float('nan'), float('nan')
            tgt_detect               = 'miss'

        err_scene_x = (scene_hx - hx) * 100
        err_scene_y = (scene_hy - hy) * 100
        err_blob_x  = (blob_hx - hx) * 100 if cyl_detect == 'diff' else float('nan')
        err_blob_y  = (blob_hy - hy) * 100 if cyl_detect == 'diff' else float('nan')

        row = dict(
            hx=round(hx,4), hy=round(hy,4),
            ix=round(ix,4), iy=round(iy,4),
            cmd_px=round(cmd_px,1), cmd_py=round(cmd_py,1),
            scene_hx=round(scene_hx,4), scene_hy=round(scene_hy,4), scene_iz=round(scene_iz,4),
            cyl_detect=cyl_detect,
            bpx=round(bpx,1) if cyl_detect=='diff' else float('nan'),
            bpy=round(bpy,1) if cyl_detect=='diff' else float('nan'),
            blob_hx=round(blob_hx,4) if cyl_detect=='diff' else float('nan'),
            blob_hy=round(blob_hy,4) if cyl_detect=='diff' else float('nan'),
            err_scene_x=round(err_scene_x,2),
            err_scene_y=round(err_scene_y,2),
            err_blob_x=round(err_blob_x,2) if cyl_detect=='diff' else float('nan'),
            err_blob_y=round(err_blob_y,2) if cyl_detect=='diff' else float('nan'),
            tgt_detect=tgt_detect,
            tpx=round(tpx,1) if tgt_detect=='ok' else float('nan'),
            tpy=round(tpy,1) if tgt_detect=='ok' else float('nan'),
            tgt_blob_hx=round(tgt_blob_hx,4) if tgt_detect=='ok' else float('nan'),
            tgt_blob_hy=round(tgt_blob_hy,4) if tgt_detect=='ok' else float('nan'),
        )
        results.append(row)

        # Annotate and save per-frame PNG (frame_on as background)
        ann = annotate_frame(frame_on, hx, hy, cyl_blob, tgt_blob,
                             TGT_HX_REF, TGT_HY_REF, diff_mask=diff_mask)
        frame_path = os.path.join(OUT_DIR, f"frame_{idx:03d}_{hx:+.3f}_{hy:+.3f}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))

        err_str = (f"({err_blob_x:+.1f},{err_blob_y:+.1f})cm"
                   if cyl_detect == 'diff' else "no-diff-blob")
        print(f"[DIAG_V2] {idx+1:3d}/{len(test_pts)}  "
              f"H_REAL ({hx:+.3f},{hy:+.3f}) → "
              f"Isaac ({ix:+.3f},{iy:+.3f})  "
              f"detect={cyl_detect}  "
              f"blob_err={err_str}  "
              f"scene_err=({err_scene_x:+.1f},{err_scene_y:+.1f})cm")

    print()
    print("[DIAG_V2] Sweep complete. Building summary plots...")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.expanduser("~/so101-sim-project/placement_diag_v2.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"[DIAG_V2] CSV → {csv_path}")

    # ── Summary figure ─────────────────────────────────────────────────────────
    env.reset()
    for _ in range(8):
        hold_step(env)
    bg = get_rgb(env)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor('#0d0d1a')

    # ── Left panel: SIM camera ────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.imshow(bg)
    ax0.set_title(
        'SIM camera — CYLINDER diagnostic\n'
        'blue × = H_SIM prediction of H_REAL coord  (where it SHOULD appear)\n'
        'green ○ = actual cylinder blob  (frame-diff, colour-agnostic)\n'
        'orange × = fixed target reference',
        color='white', fontsize=10,
    )
    ax0.axis('off')

    for r in results:
        hx, hy = r['hx'], r['hy']
        cu, cv = world2pix_sim(hx, hy)
        ax0.plot(cu, cv, 'x', color='deepskyblue', ms=8, mew=2, zorder=5)
        if r['cyl_detect'] == 'diff' and not np.isnan(r['bpx']):
            ax0.plot(r['bpx'], r['bpy'], 'o', color='lime', ms=5,
                     mec='white', mew=0.8, zorder=6)
            ax0.annotate('', xy=(r['bpx'], r['bpy']), xytext=(cu, cv),
                         arrowprops=dict(arrowstyle='->', color='yellow', lw=1.0))
            err = np.sqrt(r['err_blob_x']**2 + r['err_blob_y']**2) if (
                not np.isnan(r['err_blob_x'])) else 0.0
            if err > 3.0:
                ax0.text((cu+r['bpx'])/2 + 3, (cv+r['bpy'])/2,
                         f"{err:.0f}cm", fontsize=6, color='yellow',
                         path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

    tgu, tgv = world2pix_sim(TGT_HX_REF, TGT_HY_REF)
    ax0.plot(tgu, tgv, 'x', color='orange', ms=10, mew=2.5, zorder=8)
    ax0.text(tgu+10, tgv-5, f"TGT ({TGT_HX_REF:.3f},{TGT_HY_REF:.3f})",
             color='orange', fontsize=7, zorder=9)

    ax0.plot([], [], 'x', color='deepskyblue', ms=8, mew=2, label='CYL commanded (H_SIM proj.)')
    ax0.plot([], [], 'o', color='lime', ms=5, mec='white', label='CYL blob (diff)')
    ax0.plot([], [], '-', color='yellow', lw=1, label='Error vector')
    ax0.plot([], [], 'x', color='orange', ms=8, mew=2, label='TGT ref')
    ax0.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white',
               labelcolor='white', fontsize=7)

    # ── Right panel: world-coord scatter ──────────────────────────────────────
    ax1 = axes[1]
    ax1.set_facecolor('#111122')
    ax1.set_aspect('equal')

    for r in results:
        hx, hy = r['hx'], r['hy']
        ax1.plot(hx, hy, 'x', color='deepskyblue', ms=9, mew=2.0, zorder=5)
        if r['cyl_detect'] == 'diff' and not np.isnan(r['blob_hx']):
            ax1.plot(r['blob_hx'], r['blob_hy'], 'o',
                     color='lime', ms=6, mec='white', mew=0.8, zorder=6)
            dx, dy = r['blob_hx'] - hx, r['blob_hy'] - hy
            if abs(dx) > 0.005 or abs(dy) > 0.005:
                ax1.annotate('', xy=(r['blob_hx'], r['blob_hy']), xytext=(hx, hy),
                             arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))
            err = np.sqrt(dx**2 + dy**2) * 100
            if err > 2.0:
                ax1.text((hx+r['blob_hx'])/2 + 0.004, (hy+r['blob_hy'])/2,
                         f'{err:.0f}cm', fontsize=6, color='yellow',
                         path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    ax1.plot(TGT_HX_REF, TGT_HY_REF, '*', color='orange', ms=14,
             mec='white', mew=1, zorder=8, label=f'TGT cmd ({TGT_HX_REF:.3f},{TGT_HY_REF:.3f})')
    ax1.plot(-0.314, 0.346, 'D', color='magenta', ms=10, mec='white', mew=1,
             zorder=8, label='Ep7 CYL H_REAL (-0.314, 0.346)')

    bx = [-0.41, 0.34, 0.34, -0.41, -0.41]
    by = [ 0.00, 0.00, 0.41,  0.41,  0.00]
    ax1.plot(bx, by, '-', color='cyan', lw=1.5, alpha=0.6, label='Mat boundary')
    ax1.axvline(0, color='white', lw=0.8, alpha=0.4, ls='--')
    ax1.axhline(0, color='white', lw=0.8, alpha=0.4, ls='--')

    ax1.plot([], [], 'x', color='deepskyblue', ms=9, mew=2, label='CYL H_REAL commanded')
    ax1.plot([], [], 'o', color='lime', ms=6, mec='white', label='CYL blob → H_SIM world')
    ax1.plot([], [], '-', color='yellow', lw=1.5, label='H_REAL↔H_SIM offset')

    ax1.set_xlabel('World X (m)  [H_REAL / H_SIM, +right in image]', color='white')
    ax1.set_ylabel('World Y (m)  [H_REAL / H_SIM, +down in image]',  color='white')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444')

    blob_errs = [(r['err_blob_x'], r['err_blob_y']) for r in results
                 if r['cyl_detect'] == 'diff' and not np.isnan(r['err_blob_x'])]
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
    det_counts = Counter(r['cyl_detect'] for r in results)
    det_txt = "Detection: " + ", ".join(f"{k}={v}" for k, v in det_counts.items())
    ax1.text(0.01, 0.99, det_txt, transform=ax1.transAxes,
             fontsize=7, color='#aaa', va='top')

    ax1.legend(facecolor='#1a1a3a', edgecolor='white', labelcolor='white', fontsize=7)
    ax1.set_title(
        'World coords  ×=H_REAL commanded  ○=blob→H_SIM\n'
        'Systematic drift = Isaac world ≠ H_REAL world (for cylinder)',
        color='white', fontsize=10,
    )

    plt.suptitle(
        'CYLINDER  H_REAL ↔ Isaac alignment diagnostic  [frame-diff detection]\n'
        '× = H_REAL coord projected into SIM camera via H_SIM  (where it SHOULD appear)\n'
        '○ = frame-diff blob  (where it ACTUALLY appears)\n'
        'Systematic gap = Isaac world ≠ H_REAL world',
        color='white', fontsize=11, fontweight='bold', y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_png = os.path.expanduser("~/so101-sim-project/placement_diag_v2.png")
    plt.savefig(out_png, dpi=140, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"[DIAG_V2] Summary PNG → {out_png}")

    if blob_errs:
        print()
        print("=== Cylinder blob-vs-commanded error summary (cm) ===")
        print(f"  n detected : {len(blob_errs)}/{len(test_pts)}")
        print(f"  X error : mean={np.mean(ex):+.2f}  std={np.std(ex):.2f}  "
              f"min={min(ex):+.2f}  max={max(ex):+.2f}")
        print(f"  Y error : mean={np.mean(ey):+.2f}  std={np.std(ey):.2f}  "
              f"min={min(ey):+.2f}  max={max(ey):+.2f}")
        print(f"  |error| : mean={np.mean(mag):.2f}  max={max(mag):.2f}")
        print()
        print("  Detection breakdown:", dict(det_counts))
        print()
        print("  If mean X error ≈ const and mean Y error ≈ const:")
        print("  → fixed offset between H_REAL world and Isaac world")
        print(f"  Estimated correction: dX={-np.mean(ex)/100:+.4f}m  dY={-np.mean(ey)/100:+.4f}m")
    else:
        print("[DIAG_V2] No blobs detected by frame-diff.")
        print("  → Check per-frame PNGs — green tint shows what changed between ON and OFF.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()