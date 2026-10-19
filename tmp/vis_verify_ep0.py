"""
vis_pick_place_calibration.py
Reproduces the "Pick & Place Calibration" diagnostic figure.

Left panel:  Sim CameraOverhead — shows where objects are placed (green/yellow solid),
             where camera detected them in real video (blue/magenta dashed), and
             where the EEF jaw actually goes at pick/place (red/orange cross).
Right panel: Real HuggingFace video frame with detected cylinder (cyan) and cup (pink).

Run from ~/IsaacLab:
  ./isaaclab.sh -p ~/so101-sim-project/tmp/vis_pick_place_calibration.py \
      --headless --enable_cameras [--episode 7]
Output: ~/so101-sim-project/tmp/pick_place_cal_epN.png
"""

import argparse, os, subprocess, math, warnings
import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")

# ── CLI + Isaac Lab bootstrap ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--episode",        type=int, default=0)
parser.add_argument("--headless",       action="store_true")
parser.add_argument("--enable_cameras", action="store_true")
args_cli, _ = parser.parse_known_args()
EP = args_cli.episode

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": args_cli.headless,
                             "enable_cameras": args_cli.enable_cameras})
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from sim_to_real_so101.tasks.cylinder_to_target_env_cfg import CylinderToTargetEnvCfg
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R_scipy

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA    = os.path.expanduser("~/.cache/huggingface/lerobot/Screener2/V4_pnp_44_20260720_141907")
PARQUET = DATA + "/data/chunk-000/file-000.parquet"
EP_META = DATA + "/meta/episodes/chunk-000/file-000.parquet"
VIDEO   = DATA + "/videos/observation.images.overhead/chunk-000/file-000.mp4"
URDF    = "/home/ted/SO101/Misc/so101_new_calib.urdf"
OUT     = os.path.expanduser(f"~/so101-sim-project/tmp/pick_place_cal_ep{EP}.png")

# ── Video camera calibration ──────────────────────────────────────────────────
IMG_CX, IMG_CY   = 320, 240
SCALE_X, SCALE_Y = 0.000816, 0.001081
OX, OY           = 0.090, 0.008
BASE_XY          = np.array([-0.03, 0.0])
LO_CYL = np.array([35, 40,  40], np.uint8); HI_CYL = np.array([85, 255, 255], np.uint8)
LO_TGT = np.array([10, 50,  50], np.uint8); HI_TGT = np.array([45, 255, 255], np.uint8)
OPEN_THR = 18.0

def pix2world(col, row):
    cx = OX + (row - IMG_CY) * SCALE_X
    cy = OY + (col - IMG_CX) * SCALE_Y
    return float(-cy - 0.017), float(cx + 0.107)

def world2vidpix(wx, wy):
    cy = -wx - 0.017
    cx = wy - 0.107
    col = int(round(IMG_CX + (cy - OY) / SCALE_Y))
    row = int(round(IMG_CY + (cx - OX) / SCALE_X))
    return col, row   # (x, y) for matplotlib / cv2

def _ok(wx, wy):
    if 0.22 <= wx <= 0.38 and 0.17 <= wy <= 0.38: return False
    if math.hypot(wx - BASE_XY[0], wy - BASE_XY[1]) < 0.12: return False
    return True

def _find_blob(frame, side, lo, hi):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    h, w = frame.shape[:2]
    if side == "left":  mask[:, w//2:] = 0
    if side == "right": mask[:, :w//2] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 50: return None
    M = cv2.moments(c)
    if M["m00"] == 0: return None
    col_c = int(M["m10"]/M["m00"])
    row_c = int(M["m01"]/M["m00"])
    return pix2world(col_c, row_c), (col_c, row_c)

def find_pick_place(g):
    fo = pi = pl = None
    for i in range(1, len(g)):
        if g[i] > OPEN_THR and g[i-1] <= OPEN_THR: fo = i; break
    if fo:
        for i in range(fo+1, len(g)):
            if g[i] < OPEN_THR and g[i-1] >= OPEN_THR: pi = i; break
    if pi:
        for i in range(pi+1, len(g)):
            if g[i] > OPEN_THR and g[i-1] <= OPEN_THR: pl = i; break
    return pi, pl

# ── FK ─────────────────────────────────────────────────────────────────────────
chain = Chain.from_urdf_file(URDF)

def fk_world(sv):
    angles = [0] + [math.radians(d) for d in sv[:5]] + [0]
    T = chain.forward_kinematics(angles)
    lx, ly = T[0, 3], T[1, 3]
    return float(-ly + BASE_XY[0]), float(lx + BASE_XY[1])

# ── Load data ─────────────────────────────────────────────────────────────────
df_all  = pd.read_parquet(PARQUET)
ep_meta = pd.read_parquet(EP_META)
df      = df_all[df_all["episode_index"] == EP].reset_index(drop=True)
states  = np.array(df["observation.state"].tolist())
g_vals  = states[:, 5]
pi, pl  = find_pick_place(g_vals)
meta    = ep_meta[ep_meta["episode_index"] == EP]
voff    = float(meta["videos/observation.images.overhead/from_timestamp"].iloc[0])
print(f"[EP{EP}] pick={pi}, place={pl}, frames={len(df)}")

# ── Detect from real video ────────────────────────────────────────────────────
first_frame = None
cyl_w_all, cyl_p_all, tgt_w_all, tgt_p_all = [], [], [], []
for dt in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    r = subprocess.run(
        ["ffmpeg", "-ss", str(voff+dt), "-i", VIDEO,
         "-frames:v","1","-f","rawvideo","-pix_fmt","bgr24","pipe:1"],
        capture_output=True)
    if len(r.stdout) != 640*480*3: continue
    f = np.frombuffer(r.stdout, np.uint8).reshape(480,640,3).copy()
    if first_frame is None: first_frame = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    f[415:,455:]=0; f[435:,:110]=0
    c = _find_blob(f,"left", LO_CYL,HI_CYL)
    t = _find_blob(f,"right",LO_TGT,HI_TGT)
    if c and _ok(*c[0]): cyl_w_all.append(c[0]); cyl_p_all.append(c[1])
    if t and _ok(*t[0]): tgt_w_all.append(t[0]); tgt_p_all.append(t[1])

assert cyl_w_all and tgt_w_all, "Detection failed"
cyl_wx = float(np.median([p[0] for p in cyl_w_all]))
cyl_wy = float(np.median([p[1] for p in cyl_w_all]))
tgt_wx = float(np.median([p[0] for p in tgt_w_all]))
tgt_wy = float(np.median([p[1] for p in tgt_w_all]))
cyl_vpx = (int(np.median([p[0] for p in cyl_p_all])), int(np.median([p[1] for p in cyl_p_all])))
tgt_vpx = (int(np.median([p[0] for p in tgt_p_all])), int(np.median([p[1] for p in tgt_p_all])))
print(f"[DET] cyl=({cyl_wx:.3f},{cyl_wy:.3f})  tgt=({tgt_wx:.3f},{tgt_wy:.3f})")

# ── FK EEF at pick and place ──────────────────────────────────────────────────
fk_pick  = fk_world(states[pi])
fk_place = fk_world(states[pl])
fk_pick_vpx  = world2vidpix(*fk_pick)
fk_place_vpx = world2vidpix(*fk_place)
gap_pick_cm  = math.hypot(fk_pick[0]-cyl_wx,  fk_pick[1]-cyl_wy)  * 100
gap_place_cm = math.hypot(fk_place[0]-tgt_wx, fk_place[1]-tgt_wy) * 100
print(f"[FK] pick=({fk_pick[0]:.3f},{fk_pick[1]:.3f})  gap→cyl={gap_pick_cm:.1f}cm")
print(f"[FK] place=({fk_place[0]:.3f},{fk_place[1]:.3f}) gap→cup={gap_place_cm:.1f}cm")

# ── Sim: place objects and capture overhead image ─────────────────────────────
def set_positions(env, cyl_xyz, tgt_xyz):
    dev = env.device
    for name, xyz in [("cylinder", cyl_xyz), ("target", tgt_xyz)]:
        obj = env.scene[name]
        st  = obj.data.root_state_w.clone()
        st[:, :3]  = torch.tensor([xyz], dtype=torch.float32, device=dev)
        st[:, 3:7] = torch.tensor([[1.,0.,0.,0.]], device=dev)
        st[:, 7:]  = 0.
        obj.write_root_state_to_sim(st)

env_cfg = CylinderToTargetEnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.sim.device     = "cpu"
env = ManagerBasedRLEnv(cfg=env_cfg)
env.reset()
set_positions(env, (cyl_wx, cyl_wy, 0.065), (tgt_wx, tgt_wy, 0.060))
zero = torch.zeros(1, env.action_space.shape[-1])
for _ in range(30):
    env.step(zero); simulation_app.update()

obs, *_ = env.step(zero); simulation_app.update()

# Get sim overhead RGB
sim_rgb = obs["visual"]["rgb_overhead"][0].cpu().numpy()
if sim_rgb.dtype != np.uint8:
    sim_rgb = (sim_rgb*255).astype(np.uint8) if sim_rgb.max()<=1.0 else sim_rgb.astype(np.uint8)
# Already RGB (Isaac Lab outputs RGB)
sim_img = sim_rgb.copy()
H_sim, W_sim = sim_img.shape[:2]
print(f"[SIM] image shape: {sim_img.shape}")

# Get sim camera intrinsics and pose for projection
cam = env.scene["camera_overhead"]
K_sim = cam.data.intrinsic_matrices[0].cpu().numpy()     # (3,3)
cam_pos_w   = cam.data.pos_w[0].cpu().numpy()            # (3,)
# quat_w_ros is [x,y,z,w] in Isaac Lab (ROS convention)
cam_quat_xyzw = cam.data.quat_w_ros[0].cpu().numpy()    # (4,) [x,y,z,w]
print(f"[CAM] pos={cam_pos_w.round(3)}")
print(f"[CAM] K=\n{K_sim.round(1)}")

def world_to_sim_px(world_xyz, z_obj=0.05):
    """Project world XYZ to sim image pixel (col, row) using camera intrinsics."""
    p = np.array(world_xyz, dtype=float)
    if len(p) == 2:
        p = np.array([p[0], p[1], z_obj])
    # World → camera frame (transpose = inverse rotation)
    Rmat = R_scipy.from_quat(cam_quat_xyzw).as_matrix()  # world_R_cam
    p_cam = Rmat.T @ (p - cam_pos_w)
    # OpenGL → OpenCV (flip Y and Z)
    p_cv  = np.array([p_cam[0], -p_cam[1], -p_cam[2]])
    if p_cv[2] <= 0.001: return None
    proj = K_sim @ (p_cv / p_cv[2])
    return int(round(proj[0])), int(round(proj[1]))

# Test projection with known sim object positions
cyl_sim_pos = env.scene["cylinder"].data.root_state_w[0,:3].cpu().numpy()
tgt_sim_pos = env.scene["target"].data.root_state_w[0,:3].cpu().numpy()
print(f"[SIM] cyl settled={cyl_sim_pos.round(3)}")
print(f"[SIM] tgt settled={tgt_sim_pos.round(3)}")

cyl_sim_px  = world_to_sim_px(cyl_sim_pos)
tgt_sim_px  = world_to_sim_px(tgt_sim_pos)
cyl_raw_px  = world_to_sim_px([cyl_wx, cyl_wy, 0.065])
tgt_raw_px  = world_to_sim_px([tgt_wx, tgt_wy, 0.060])
eef_pick_px = world_to_sim_px([fk_pick[0], fk_pick[1], 0.08])
eef_plac_px = world_to_sim_px([fk_place[0], fk_place[1], 0.08])
print(f"[PROJ] cyl_sim_px={cyl_sim_px}  cyl_raw_px={cyl_raw_px}")
print(f"[PROJ] tgt_sim_px={tgt_sim_px}  tgt_raw_px={tgt_raw_px}")
print(f"[PROJ] eef_pick={eef_pick_px}  eef_place={eef_plac_px}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor('#1a1a2e')
fig.suptitle(f"Episode {EP} — Pick & Place Calibration",
             fontsize=16, fontweight='bold', color='white', y=0.98)
fig.text(0.5, 0.94,
         "Blue/Magenta = real video RAW  |  Green/Yellow = sim placement  |  Red/Orange = EEF jaw",
         ha='center', fontsize=11, color='white')

for ax in axes:
    ax.set_facecolor('#0d0d1a')
    ax.tick_params(colors='gray')
    for spine in ax.spines.values(): spine.set_color('gray')

# ── LEFT: Sim overhead ────────────────────────────────────────────────────────
axL = axes[0]
axL.imshow(sim_img, origin='upper')
axL.set_title(f"Sim Overhead (ep{EP}, frame 5)", color='white', fontsize=12)
axL.set_xlim(0, W_sim); axL.set_ylim(H_sim, 0)

def draw_circle_sim(ax, px, color, ls, radius=22, lw=2.5):
    if px is None: return
    c = plt.Circle(px, radius, color=color, fill=False, linestyle=ls, linewidth=lw)
    ax.add_patch(c)

def draw_cross_sim(ax, px, color, size=30, lw=3):
    if px is None: return
    ax.plot([px[0]-size, px[0]+size], [px[1], px[1]], color=color, lw=lw)
    ax.plot([px[0], px[0]], [px[1]-size, px[1]+size], color=color, lw=lw)
    # Dashed outer ring
    c = plt.Circle(px, size*0.8, color=color, fill=False, linestyle='--', linewidth=1.5, alpha=0.6)
    ax.add_patch(c)

# RAW cylinder from real video (blue dashed)
draw_circle_sim(axL, cyl_raw_px,  '#4488ff', '--')
# Sim placed cylinder (green solid)
draw_circle_sim(axL, cyl_sim_px,  '#00cc44', '-')
# RAW cup from real video (magenta dashed)
draw_circle_sim(axL, tgt_raw_px,  '#ff44ff', '--')
# Sim placed cup (yellow solid)
draw_circle_sim(axL, tgt_sim_px,  '#ffdd00', '-')
# EEF at pick (red cross)
draw_cross_sim(axL, eef_pick_px,  '#ff3333')
# EEF at place (orange cross)
draw_cross_sim(axL, eef_plac_px,  '#ff8800')

# Labels near circles
def label_sim(ax, px, text, color, dy=-28):
    if px is None: return
    ax.text(px[0], px[1]+dy, text, color=color, fontsize=7.5,
            ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d0d1a', alpha=0.7, edgecolor='none'))

# Compute DX between raw and sim
cyl_dx = (cyl_sim_pos[0]-cyl_wx) if cyl_sim_px else 0
tgt_dx = (tgt_sim_pos[0]-tgt_wx) if tgt_sim_px else 0

label_sim(axL, cyl_raw_px,  "RAW (real video)", '#4488ff', dy=30)
label_sim(axL, cyl_sim_px,  f"Sim placed\n(DX={cyl_dx:+.2f})", '#00cc44', dy=-32)
label_sim(axL, tgt_raw_px,  "RAW (real video)", '#ff44ff', dy=-32)
label_sim(axL, tgt_sim_px,  f"Sim placed\n(DX={tgt_dx:+.2f})", '#ffdd00', dy=32)
if eef_pick_px:
    axL.text(eef_pick_px[0]+15, eef_pick_px[1]-15, "EEF at pick",
             color='#ff3333', fontsize=8, fontweight='bold')
if eef_plac_px:
    axL.text(eef_plac_px[0]+15, eef_plac_px[1]+20, "EEF at place",
             color='#ff8800', fontsize=8, fontweight='bold')

# Legend box (top-right)
legend_lines = [
    Line2D([0],[0],color='#4488ff', ls='--', lw=2, label=f"RAW cylinder  (real video)  X={cyl_wx:.3f} Y={cyl_wy:.3f}"),
    Line2D([0],[0],color='#00cc44', ls='-',  lw=2, label=f"Sim cylinder  (DX={cyl_dx:+.2f})   X={cyl_sim_pos[0]:.3f} Y={cyl_sim_pos[1]:.3f}"),
    Line2D([0],[0],color='#ff3333', ls='-',  lw=2, marker='x', label=f"EEF jaw at pick      X={fk_pick[0]:.3f} Y={fk_pick[1]:.3f}"),
    Line2D([0],[0],color='#ff44ff', ls='--', lw=2, label=f"RAW bucket   (real video)  X={tgt_wx:.3f} Y={tgt_wy:.3f}"),
    Line2D([0],[0],color='#ffdd00', ls='-',  lw=2, label=f"Sim bucket   (DX={tgt_dx:+.2f})   X={tgt_sim_pos[0]:.3f} Y={tgt_sim_pos[1]:.3f}"),
    Line2D([0],[0],color='#ff8800', ls='-',  lw=2, marker='x', label=f"EEF jaw at place    X={fk_place[0]:.3f} Y={fk_place[1]:.3f}"),
]
leg = axL.legend(handles=legend_lines, loc='upper right', fontsize=7.5,
                 facecolor='#1a1a2e', edgecolor='gray', labelcolor='white',
                 framealpha=0.9)

# Error box (bottom-left)
gap_cyl_sim  = math.hypot(fk_pick[0]-cyl_sim_pos[0],  fk_pick[1]-cyl_sim_pos[1])*100
gap_cyl_raw  = gap_pick_cm
gap_tgt_sim  = math.hypot(fk_place[0]-tgt_sim_pos[0], fk_place[1]-tgt_sim_pos[1])*100
gap_tgt_raw  = gap_place_cm
ok_sym = lambda d: "✓" if d<5 else ("≈ OK" if d<8 else "✗")
err_text = (
    f"PICK:  EEF → sim cyl   ΔX={abs(fk_pick[0]-cyl_sim_pos[0])*100:.1f}cm  ΔY={abs(fk_pick[1]-cyl_sim_pos[1])*100:.1f}cm  {ok_sym(gap_cyl_sim)}\n"
    f"       EEF → RAW cyl   Δ={gap_cyl_raw:.1f}cm\n"
    f"PLACE: EEF → sim bucket ΔX={abs(fk_place[0]-tgt_sim_pos[0])*100:.1f}cm ΔY={abs(fk_place[1]-tgt_sim_pos[1])*100:.1f}cm  {ok_sym(gap_tgt_sim)}\n"
    f"       EEF → RAW bucket Δ={gap_tgt_raw:.1f}cm"
)
axL.text(5, H_sim-5, err_text, fontsize=8.5, color='white', va='bottom',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#111122', edgecolor='gray', alpha=0.92))

# ── RIGHT: Real video frame ───────────────────────────────────────────────────
axR = axes[1]
axR.imshow(first_frame, origin='upper')
axR.set_title(f"Real HuggingFace Video (ep{EP}, frame 0)", color='white', fontsize=12)

def draw_circle_vid(ax, px, color, radius=25, lw=3):
    c = plt.Circle(px, radius, color=color, fill=False, linewidth=lw)
    ax.add_patch(c)

draw_circle_vid(axR, cyl_vpx, '#00ccff', radius=25)   # cyan = cylinder
draw_circle_vid(axR, tgt_vpx, '#ff44ff', radius=25)   # magenta = cup

axR.text(cyl_vpx[0]+30, cyl_vpx[1],
         f"Green cylinder\npx {cyl_vpx}\n→ world ({cyl_wx:.3f}, {cyl_wy:+.3f})",
         color='#00ccff', fontsize=9, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d0d1a', alpha=0.8, edgecolor='none'))
axR.text(tgt_vpx[0]-170, tgt_vpx[1]+30,
         f"Cup/bucket\npx {tgt_vpx}\n→ world ({tgt_wx:.3f}, {tgt_wy:+.3f})",
         color='#ff44ff', fontsize=9, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d0d1a', alpha=0.8, edgecolor='none'))
axR.set_xlim(0, 640); axR.set_ylim(480, 0)

plt.tight_layout(rect=[0,0,1,0.93])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\n[SAVED] {OUT}")
print(f"\nSummary:")
print(f"  Cylinder: camera=({cyl_wx:.3f},{cyl_wy:.3f})  sim_placed=({cyl_sim_pos[0]:.3f},{cyl_sim_pos[1]:.3f})")
print(f"  FK pick:  ({fk_pick[0]:.3f},{fk_pick[1]:.3f})  gap_to_sim_cyl={gap_cyl_sim:.1f}cm  gap_to_raw={gap_cyl_raw:.1f}cm")
print(f"  Cup:      camera=({tgt_wx:.3f},{tgt_wy:.3f})  sim_placed=({tgt_sim_pos[0]:.3f},{tgt_sim_pos[1]:.3f})")
print(f"  FK place: ({fk_place[0]:.3f},{fk_place[1]:.3f})  gap_to_sim_cup={gap_tgt_sim:.1f}cm   gap_to_raw={gap_tgt_raw:.1f}cm")

env.close()
simulation_app.close()
