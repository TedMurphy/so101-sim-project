# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
replay_diag.py  —  Cylinder placement calibration diagnostic.

Sweeps a regular grid of target positions across the mat, writes the
cylinder there via write_root_state_to_sim + env.step(), then detects
the actual cylinder blob in the overhead SIM camera via HSV.

Outputs:
  ~/so101-sim-project/cylinder_placement_diag.png
  ~/so101-sim-project/cylinder_placement_diag.csv

Run with:
  ./isaaclab.sh -p ~/so101-sim-project/source/sim_to_real_so101/scripts/replay_diag.py \
                --enable_cameras --livestream 2
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cylinder placement calibration diagnostic.")
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

# ── H_SIM homography (SIM overhead pixel → H_SIM world, metres) ───────────────
_H_SIM = np.array([
    [ 1.26659037e-03, -1.15176118e-06, -4.16389850e-01],
    [-2.38682034e-06,  1.25586847e-03, -7.64706966e-02],
    [ 2.29476315e-04, -8.30288972e-06,  1.00000000e+00],
], dtype=np.float64)
_H_SIM_INV = np.linalg.inv(_H_SIM)


def pix2world(u, v):
    """SIM overhead pixel → H_SIM world (x, y) metres."""
    p = _H_SIM @ np.array([u, v, 1.0])
    return float(p[0]/p[2]), float(p[1]/p[2])


def world2pix(wx, wy):
    """H_SIM world (x, y) metres → SIM overhead pixel (u, v)."""
    p = _H_SIM_INV @ np.array([wx, wy, 1.0])
    return float(p[0]/p[2]), float(p[1]/p[2])


# ── Helpers ────────────────────────────────────────────────────────────────────

def place_cylinder(env, isaac_x, isaac_y, isaac_z=0.050):
    """Write cylinder to Isaac (X, Y, Z) via write_root_state_to_sim.
    Call env.step() afterwards to flush to PhysX."""
    device = env.unwrapped.device
    obj    = env.unwrapped.scene["cylinder"]
    state  = obj.data.root_state_w.clone()
    state[:, :3]  = torch.tensor([[isaac_x, isaac_y, isaac_z]],
                                 dtype=torch.float32, device=device)
    state[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    state[:, 7:]  = 0.0
    obj.write_root_state_to_sim(state)


def hold_step(env):
    """env.step() holding current joint positions — flushes any pending
    write_root_state_to_sim and advances physics one tick."""
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


def detect_cylinder_blob(rgb_frame):
    """HSV-detect the largest green blob (cylinder) in the SIM frame.

    Returns (cx_pixel, cy_pixel) or None if nothing found.
    The arm occupies centre-top of the 640×480 image; we mask it out.
    """
    H, W = rgb_frame.shape[:2]
    hsv  = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)

    # Green HSV range (cylinder colour in SIM)
    mask = cv2.inRange(hsv, np.array([40, 60, 60]), np.array([100, 255, 255]))

    # Mask arm region (centre-top)
    mask[:int(H*0.45), int(W*0.20):int(W*0.80)] = 0

    # Morphological clean-up
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
    if best is None:
        return None
    return best[1], best[2]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Boot environment ───────────────────────────────────────────────────────
    env_cfg = parse_env_cfg(
        "Lerobot-So101-Teleop-Cylinder-To-Target",
        device='cpu', num_envs=1,
    )
    env = gym.make("Lerobot-So101-Teleop-Cylinder-To-Target", cfg=env_cfg)
    env.reset()

    # Pump to initialise rendering
    for _ in range(15):
        simulation_app.update()

    print("[DIAG] Environment ready. Starting placement sweep...")
    print()

    # ── Grid of test positions (in H_SIM world coords) ─────────────────────────
    # Mat spans roughly: X ∈ [-0.41, +0.34],  Y ∈ [0.00, +0.41]
    # We test within a safe inner margin.
    # Convention: Isaac_X = -H_SIM_X,  Isaac_Y = H_SIM_Y
    grid_hx = np.arange(-0.38, 0.32, 0.08)   # H_SIM X (7–9 columns)
    grid_hy = np.arange(0.04,  0.42, 0.08)   # H_SIM Y (5 rows)

    test_pts = []   # (hsim_x_target, hsim_y_target)
    for hx in grid_hx:
        for hy in grid_hy:
            test_pts.append((float(hx), float(hy)))

    print(f"[DIAG] {len(test_pts)} test positions  "
          f"({len(grid_hx)} cols × {len(grid_hy)} rows)")
    print()

    results = []  # dicts

    for idx, (tx, ty) in enumerate(test_pts):
        # Isaac write coords
        ix, iy = -tx, ty

        # Write cylinder, flush, settle (3 steps)
        place_cylinder(env, ix, iy)
        hold_step(env)          # ← flushes write_root_state_to_sim to PhysX
        hold_step(env)          # settle
        hold_step(env)          # settle
        simulation_app.update()

        # Read actual Isaac position from the scene (post-step)
        cyl_state = env.unwrapped.scene["cylinder"].data.root_state_w[0]
        actual_ix = float(cyl_state[0])
        actual_iy = float(cyl_state[1])
        actual_iz = float(cyl_state[2])
        # Convert Isaac → H_SIM world
        actual_hx = -actual_ix
        actual_hy =  actual_iy

        # Capture camera frame and detect blob
        frame = get_rgb(env)
        blob  = detect_cylinder_blob(frame)

        if blob is not None:
            bpx, bpy  = blob
            cam_hx, cam_hy = pix2world(bpx, bpy)
            detection = "ok"
        else:
            cam_hx, cam_hy = actual_hx, actual_hy   # fall back to scene state
            detection = "blob_miss"

        err_state_x = (actual_hx - tx) * 100   # cm
        err_state_y = (actual_hy - ty) * 100
        err_cam_x   = (cam_hx   - tx) * 100
        err_cam_y   = (cam_hy   - ty) * 100

        row = dict(
            tgt_hx=round(tx, 4), tgt_hy=round(ty, 4),
            act_hx=round(actual_hx, 4), act_hy=round(actual_hy, 4),
            act_iz=round(actual_iz, 4),
            cam_hx=round(cam_hx, 4), cam_hy=round(cam_hy, 4),
            err_state_x=round(err_state_x, 2), err_state_y=round(err_state_y, 2),
            err_cam_x=round(err_cam_x, 2),   err_cam_y=round(err_cam_y, 2),
            detection=detection,
        )
        results.append(row)

        print(f"[DIAG] {idx+1:3d}/{len(test_pts)}  "
              f"tgt H_SIM ({tx:+.3f},{ty:+.3f})  "
              f"→ Isaac write ({ix:+.3f},{iy:+.3f})  "
              f"→ scene actual H_SIM ({actual_hx:+.3f},{actual_hy:+.3f})  "
              f"err=({err_state_x:+.1f},{err_state_y:+.1f})cm  "
              f"detect={detection}")

    print()
    print("[DIAG] Sweep complete. Building plots...")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    import csv, os
    csv_path = os.path.expanduser("~/so101-sim-project/cylinder_placement_diag.csv")
    keys = list(results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"[DIAG] CSV → {csv_path}")

    # ── Figure 1: overlay on SIM image ─────────────────────────────────────────
    # Grab one final clean frame for background
    env.reset()
    for _ in range(5):
        hold_step(env)
    bg_frame = get_rgb(env)

    # ── Figure 2: world-coord scatter with error vectors ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('#0d0d1a')

    # ── Left panel: SIM camera background with pixel-space arrows ─────────────
    ax0 = axes[0]
    ax0.imshow(bg_frame)
    ax0.set_title('SIM camera: attempted (×) vs actual scene state (○)',
                  color='white', fontsize=11)
    ax0.axis('off')

    for r in results:
        # Attempted pixel
        au, av = world2pix(r['tgt_hx'], r['tgt_hy'])
        # Actual (from scene state)
        bu, bv = world2pix(r['act_hx'], r['act_hy'])
        ax0.plot(au, av, 'x', color='deepskyblue', ms=9, mew=2, zorder=5)
        ax0.plot(bu, bv, 'o', color='lime',        ms=6, mec='white', mew=1, zorder=6)
        if abs(au-bu) > 2 or abs(av-bv) > 2:
            ax0.annotate('', xy=(bu, bv), xytext=(au, av),
                         arrowprops=dict(arrowstyle='->', color='yellow', lw=1.3))
        err = np.sqrt((r['act_hx']-r['tgt_hx'])**2 + (r['act_hy']-r['tgt_hy'])**2) * 100
        if err > 1.0:
            ax0.text((au+bu)/2 + 4, (av+bv)/2,
                     f'{err:.1f}', fontsize=6, color='yellow',
                     path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

    # ── Right panel: H_SIM world coords ───────────────────────────────────────
    ax1 = axes[1]
    ax1.set_facecolor('#111122')
    ax1.set_aspect('equal')

    for r in results:
        tx, ty = r['tgt_hx'], r['tgt_hy']
        ax_a, ay_a = r['act_hx'], r['act_hy']
        ax1.plot(tx,  ty,  'x', color='deepskyblue', ms=10, mew=2.5, zorder=5)
        ax1.plot(ax_a, ay_a, 'o', color='lime', ms=7, mec='white', mew=1, zorder=6)
        dx, dy = ax_a - tx, ay_a - ty
        if abs(dx) > 0.002 or abs(dy) > 0.002:
            ax1.annotate('', xy=(ax_a, ay_a), xytext=(tx, ty),
                         arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))
        err = np.sqrt(dx**2 + dy**2) * 100
        if err > 0.5:
            ax1.text((tx+ax_a)/2 + 0.005, (ty+ay_a)/2,
                     f'{err:.1f}cm', fontsize=7, color='yellow',
                     path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    # Mat boundary
    bx = [-0.41, 0.34, 0.34, -0.41, -0.41]
    by = [ 0.00, 0.00, 0.41,  0.41,  0.00]
    ax1.plot(bx, by, '-', color='cyan', lw=1.5, alpha=0.6)
    ax1.axvline(0, color='white', lw=0.8, alpha=0.4, ls='--')

    # Mark the two episode-7 objects for reference
    ax1.plot(-0.314, 0.346, '*', color='lime',   ms=14, mec='white', mew=1,
             zorder=8, label='Ep7 CYL target (-0.314, 0.346)')
    ax1.plot( 0.254, 0.054, '*', color='orange', ms=14, mec='white', mew=1,
             zorder=8, label='Ep7 TGT target (+0.254, 0.054)')

    ax1.set_xlabel('H_SIM world X (m)  [left = +x, robot right = -x]', color='white')
    ax1.set_ylabel('H_SIM world Y (m)  [up in image = lower y]',        color='white')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444')
    leg = ax1.legend(facecolor='#1a1a3a', edgecolor='white',
                     labelcolor='white', fontsize=8)
    ax1.set_title('H_SIM world coords: × = write target, ○ = scene actual\nYellow arrow = error',
                  color='white', fontsize=11)

    # Legend entries
    ax1.plot([], [], 'x',  color='deepskyblue', ms=10, mew=2, label='Write target')
    ax1.plot([], [], 'o',  color='lime',        ms=7,  mec='white', label='Scene actual')
    ax1.plot([], [], '-',  color='yellow',      lw=1.5, label='Error vector')
    ax1.legend(facecolor='#1a1a3a', edgecolor='white', labelcolor='white', fontsize=8)

    plt.suptitle(
        'Cylinder placement diagnostic — write_root_state_to_sim vs scene actual\n'
        'Same (X,Y) used for Isaac write and H_SIM query  |  Isaac_X = -H_SIM_X',
        color='white', fontsize=12, fontweight='bold', y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = os.path.expanduser("~/so101-sim-project/cylinder_placement_diag.png")
    plt.savefig(out_png, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"[DIAG] PNG  → {out_png}")

    # ── Summary stats ──────────────────────────────────────────────────────────
    errs_x = [r['err_state_x'] for r in results]
    errs_y = [r['err_state_y'] for r in results]
    mags   = [np.sqrt(r['err_state_x']**2 + r['err_state_y']**2) for r in results]
    print()
    print("=== Placement error summary (scene state, cm) ===")
    print(f"  X error:  mean={np.mean(errs_x):+.2f}  std={np.std(errs_x):.2f}  "
          f"min={min(errs_x):+.2f}  max={max(errs_x):+.2f}")
    print(f"  Y error:  mean={np.mean(errs_y):+.2f}  std={np.std(errs_y):.2f}  "
          f"min={min(errs_y):+.2f}  max={max(errs_y):+.2f}")
    print(f"  |error|:  mean={np.mean(mags):.2f}   std={np.std(mags):.2f}   "
          f"max={max(mags):.2f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()