"""
Run from ~/IsaacLab:
  ./isaaclab.sh -p /tmp/vis_ep0_simcam.py --headless --enable_cameras
Saves /tmp/sim_overhead_ep0.png — the sim camera view after placing objects
at the detection-derived world positions for episode 0.
"""
import argparse, sys, os, subprocess, math
import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--enable_cameras", action="store_true")
args_cli, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from sim_to_real_so101.tasks.cylinder_to_target_env_cfg import CylinderToTargetEnvCfg
import warnings; warnings.filterwarnings("ignore")

# ── Detection constants (from replay_interactive.py) ─────────────────────────
DATA    = os.path.expanduser("~/.cache/huggingface/lerobot/Screener2/V4_pnp_44_20260720_141907")
VIDEO   = DATA + "/videos/observation.images.overhead/chunk-000/file-000.mp4"
EP_META = DATA + "/meta/episodes/chunk-000/file-000.parquet"
PARQUET = DATA + "/data/chunk-000/file-000.parquet"

LO_CYL = np.array([35,40,40], np.uint8);  HI_CYL = np.array([85,255,255], np.uint8)
LO_TGT = np.array([10,50,50], np.uint8);  HI_TGT = np.array([45,255,255], np.uint8)

def pix2world(col, row):
    wx = -0.025 - (col - 320) * 0.001081
    wy =  0.197 + (row - 240) * 0.000816
    return wx, wy

def detect_ep0():
    import cv2
    ep_meta = pd.read_parquet(EP_META)
    voff = float(ep_meta[ep_meta["episode_index"]==0]["videos/observation.images.overhead/from_timestamp"].iloc[0])
    cp, tp = [], []
    for dt in [0.5, 1.0, 1.5, 2.0]:
        r = subprocess.run(["ffmpeg","-ss",str(voff+dt),"-i",VIDEO,
            "-frames:v","1","-f","rawvideo","-pix_fmt","bgr24","pipe:1"], capture_output=True)
        if len(r.stdout)!=640*480*3: continue
        f = np.frombuffer(r.stdout,np.uint8).reshape(480,640,3).copy()
        f[415:,455:]=0;  f[435:,:110]=0
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        for lo,hi,side,lst in [(LO_CYL,HI_CYL,"left",cp),(LO_TGT,HI_TGT,"right",tp)]:
            mask = cv2.inRange(hsv,lo,hi)
            if side=="left": mask[:,320:]=0
            else:            mask[:,:320]=0
            k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
            cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            c=max(cnts,key=cv2.contourArea)
            if cv2.contourArea(c)<50: continue
            M=cv2.moments(c)
            cx,cy=int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"])
            lst.append(pix2world(cx,cy))
    if not cp or not tp: return None
    cyl = (float(np.median([p[0] for p in cp])), float(np.median([p[1] for p in cp])))
    tgt = (float(np.median([p[0] for p in tp])), float(np.median([p[1] for p in tp])))
    return cyl, tgt

det = detect_ep0()
if det is None:
    print("Detection failed"); sys.exit(1)
cyl_xy, tgt_xy = det
CYL_Z = 0.035 + 0.030
TGT_Z = 0.035 + 0.025
print(f"[DET] cylinder world = {cyl_xy},  target world = {tgt_xy}")

# ── Build env ────────────────────────────────────────────────────────────────
env_cfg = CylinderToTargetEnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.sim.device = "cpu"
env = ManagerBasedRLEnv(cfg=env_cfg)
env.reset()

def set_pos(env, name, xyz):
    obj = env.scene[name]
    s = obj.data.root_state_w.clone()
    s[:,:3] = torch.tensor([xyz], dtype=torch.float32)
    s[:,3:7] = torch.tensor([[1,0,0,0]])
    s[:,7:] = 0
    obj.write_root_state_to_sim(s)

set_pos(env, "cylinder", (cyl_xy[0], cyl_xy[1], CYL_Z))
set_pos(env, "target",   (tgt_xy[0], tgt_xy[1], TGT_Z))
zero = torch.zeros(1, env.action_space.shape[-1])
for _ in range(30): env.step(zero); simulation_app.update()

# ── Capture overhead camera ──────────────────────────────────────────────────
obs, *_ = env.step(zero)
simulation_app.update()
rgb = env.scene["camera_overhead"].data.output["rgb"][0].cpu().numpy()  # (H, W, 4) RGBA
import cv2 as cv2_
img = cv2_.cvtColor(rgb[:,:,:3], cv2_.COLOR_RGB2BGR)

# Annotate with detected world positions as text
cv2_.putText(img, f"CYL world ({cyl_xy[0]:.3f},{cyl_xy[1]:.3f})", (10,20),
             cv2_.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
cv2_.putText(img, f"TGT world ({tgt_xy[0]:.3f},{tgt_xy[1]:.3f})", (10,40),
             cv2_.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,255), 1)

out = "/tmp/sim_overhead_ep0.png"
cv2_.imwrite(out, img)
print(f"Saved: {out}")
env.close(); simulation_app.close()
