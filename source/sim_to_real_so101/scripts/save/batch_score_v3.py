import argparse, sys, os, time
import numpy as np
import pandas as pd
import torch
parser = argparse.ArgumentParser()
parser.add_argument("--task",         type=str, default="Lerobot-So101-Teleop-Cylinder-To-Target")
parser.add_argument("--real_repo_id", type=str, default="Screener2/V4_pnp_44_20260720_141907")
parser.add_argument("--episodes",     type=str, default="all")
parser.add_argument("--skip",         type=str, default="34,41")
parser.add_argument("--success_r",    type=float, default=0.050)
parser.add_argument("--settle_steps", type=int, default=60)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
from isaaclab.envs import ManagerBasedRLEnv
from sim_to_real_so101.tasks.cylinder_to_target_env_cfg import CylinderToTargetEnvCfg
import subprocess, math, warnings
warnings.filterwarnings("ignore")
real_root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{args_cli.real_repo_id}")
PARQUET   = f"{real_root}/data/chunk-000/file-000.parquet"
EP_META   = f"{real_root}/meta/episodes/chunk-000/file-000.parquet"
VIDEO     = f"{real_root}/videos/observation.images.overhead/chunk-000/file-000.mp4"
df_all  = pd.read_parquet(PARQUET)
ep_meta = pd.read_parquet(EP_META)
all_eps = sorted(df_all["episode_index"].unique().tolist())
skip    = set(int(x) for x in args_cli.skip.split(",") if x.strip())
if args_cli.episodes == "all":
    run_eps = [e for e in all_eps if e not in skip]
else:
    run_eps = [int(x) for x in args_cli.episodes.split(",") if x.strip()]
print(f"[BATCH] Scoring {len(run_eps)} episodes. Success radius={args_cli.success_r*100:.0f}cm")
IMG_CX, IMG_CY   = 320, 240
from homography_pix2world import pix2world_real as pix2world
BASE_XY          = np.array([-0.03, 0.0])
LO_CYL = np.array([35,40,40],np.uint8);  HI_CYL = np.array([85,255,255],np.uint8)
LO_TGT = np.array([10,50,50],np.uint8);  HI_TGT = np.array([45,255,255],np.uint8)
def _ok(wx, wy):
    # exclusion removed: camera-frame box invalid in world frame
    if math.hypot(wx-BASE_XY[0], wy-BASE_XY[1])<0.12: return False
    return True
def _find_blob(frame, side, lo, hi):
    import cv2
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    h,w = frame.shape[:2]
    if side=="left":  mask[:,w//2:]=0
    if side=="right": mask[:,:w//2]=0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
    mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,k)
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts,key=cv2.contourArea)
    if cv2.contourArea(c)<50: return None
    M = cv2.moments(c)
    if M["m00"]==0: return None
    return pix2world(int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
def detect_objects(vid_offset):
    cp,tp=[],[]
    for dt in [0.0,0.5,1.0,1.5,2.0,2.5,3.0]:
        r=subprocess.run(["ffmpeg","-ss",str(vid_offset+dt),"-i",VIDEO,
                          "-frames:v","1","-f","rawvideo","-pix_fmt","bgr24","pipe:1"],
                         capture_output=True)
        if len(r.stdout)!=640*480*3: continue
        f=np.frombuffer(r.stdout,np.uint8).reshape(480,640,3).copy()
        f[415:,455:]=0
        f[435:,:110]=0   # mask green cylinder reflection bottom-left
        c=_find_blob(f,"left",LO_CYL,HI_CYL)
        t=_find_blob(f,"right",LO_TGT,HI_TGT)
        if c and _ok(*c): cp.append(c)
        if t and _ok(*t): tp.append(t)
    if not cp or not tp: return None
    return (float(np.median([p[0] for p in cp])),float(np.median([p[1] for p in cp])),
            float(np.median([p[0] for p in tp])),float(np.median([p[1] for p in tp])))
def set_positions(env, cyl_xyz, tgt_xyz):
    device = env.device
    for obj_name, xyz in [("cylinder", cyl_xyz), ("target", tgt_xyz)]:
        obj   = env.scene[obj_name]
        state = obj.data.root_state_w.clone()
        state[:,:3]  = torch.tensor([xyz], dtype=torch.float32, device=device)
        state[:,3:7] = torch.tensor([[1.0,0.0,0.0,0.0]], device=device)
        state[:,7:]  = 0.0
        obj.write_root_state_to_sim(state)
OPEN_THR = 18.0
def find_pick_place(g):
    first_open=None
    for i in range(1,len(g)):
        if g[i]>OPEN_THR and g[i-1]<=OPEN_THR: first_open=i; break
    pick_idx=None
    if first_open:
        for i in range(first_open+1,len(g)):
            if g[i]<OPEN_THR and g[i-1]>=OPEN_THR: pick_idx=i; break
    place_idx=None
    if pick_idx:
        for i in range(pick_idx+1,len(g)):
            if g[i]>OPEN_THR and g[i-1]<=OPEN_THR: place_idx=i; break
    return pick_idx, place_idx
env_cfg = CylinderToTargetEnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.sim.device     = "cpu"
env_cfg.viewer.eye    = (0.22, 0.001, 1.2)   # directly overhead
env_cfg.viewer.lookat = (0.22, 0.0,   0.035) # mat surface
env = ManagerBasedRLEnv(cfg=env_cfg)
env.reset()
# Set viewport camera to top-down overhead view (must use set_camera_view,
# not env_cfg.viewer — Kit's free camera overwrites USD prim values every frame)
try:
    import omni.kit.viewport.utility as _vp_util
    _vp = _vp_util.get_active_viewport()
    _vp.camera_path = "/World/envs/env_0/CameraOverhead"
    simulation_app.update()
    print("[camera] viewport → CameraOverhead")
except Exception as _e:
    print(f"[camera] viewport set failed: {_e}")
print("[BATCH] Environment ready.\n")

# ── Action interface (matches replay_interactive.py exactly) ──────────────
from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface
_cameras = {}
try:
    for _obj in env.scene.keys():
        if _obj.startswith("camera_"):
            _cameras[_obj.replace("camera_", "")] = {
                "height": env.scene[_obj].cfg.height,
                "width":  env.scene[_obj].cfg.width,
            }
except Exception as _e:
    print(f"[BATCH] camera cfg fallback: {_e}")
    _cameras = {"overhead": {"height": 480, "width": 640}}
robot_iface = LeRobotSO101Interface(
    device=env.device, port="dummy", id="replay",
    cameras=_cameras, fps=30, kind="leader",
)
print(f"[BATCH] LeRobotSO101Interface ready. cameras={list(_cameras.keys())}")
# ─────────────────────────────────────────────────────────────────────────────

HDR = (f"{'Ep':>3}  {'Cyl_x':>7} {'Cyl_y':>7}  {'Tgt_x':>7} {'Tgt_y':>7}  "
       f"{'End_cyl_x':>9} {'End_cyl_y':>9}  {'Dist_cm':>7}  {'Result':>6}")
print(HDR); print("-"*len(HDR))
results=[]
for ep_idx in run_eps:
    t0=time.time()
    df     = df_all[df_all["episode_index"]==ep_idx].reset_index(drop=True)
    states = np.array(df["observation.state"].tolist())
    actions= np.array(df["action"].tolist())
    g = states[:,5]
    pi,pl = find_pick_place(g)
    if pi is None or pl is None:
        print(f"{ep_idx:>3}  -- no gripper event"); results.append({"ep":ep_idx,"result":"no_event"}); continue
    print(f"    [pick_idx={pi} place_idx={pl}]  gripper_val_at_pick={g[pi]:.1f}")
    meta_row = ep_meta[ep_meta["episode_index"]==ep_idx]
    voff     = float(meta_row["videos/observation.images.overhead/from_timestamp"].iloc[0])
    det      = detect_objects(voff)
    if det is None:
        print(f"{ep_idx:>3}  -- detection failed"); results.append({"ep":ep_idx,"result":"det_failed"}); continue
    cyl_x,cyl_y,tgt_x,tgt_y = det
    cyl_xyz = (cyl_x, cyl_y, 0.065)
    tgt_xyz = (tgt_x, tgt_y, 0.060)
    env.reset()
    set_positions(env, cyl_xyz, tgt_xyz)
    zero = torch.zeros(1, env.action_space.shape[-1])
    for _ in range(args_cli.settle_steps):
        env.step(zero)
    set_positions(env, cyl_xyz, tgt_xyz)
    robot     = env.scene["robot"]
    cyl_obj   = env.scene["cylinder"]
    grasping  = False
    grasp_off = None
    _eef_pl = None; _d_pl_xy = float('nan')   # placement tracking
    for i in range(len(df)):
        # Use the same action mapping as replay_interactive.py
        raw_action = torch.tensor(actions[i], dtype=torch.float32)
        sim_action = robot_iface.get_mapped_actions_vectorized(raw_action)
        env.step(sim_action.unsqueeze(0))
        if i == 5:
            try:
                import numpy as _np2
                from PIL import Image as _PIL2
                _rgb = env.scene["camera_overhead"].data.output["rgb"]
                _f = _rgb[0].cpu().numpy().astype(_np2.uint8)
                _out = os.path.expanduser(
                    f"~/so101-sim-project/sim_overhead_batch_ep{ep_idx}_frame5.png")
                _PIL2.fromarray(_f[..., :3] if _f.shape[-1] == 4 else _f).save(_out)
                print(f"[IMG] Saved {_out}", flush=True)
            except Exception as _e:
                print(f"[IMG] Save failed: {_e}", flush=True)
        simulation_app.update()
        if i % 20 == 0:
            _eef = robot.data.body_state_w[0, -1, :3].cpu().numpy()
            print(f"[EEF_B] ep={ep_idx} frame={i:4d}  x={_eef[0]:.4f}  y={_eef[1]:.4f}  z={_eef[2]:.4f}", flush=True)
        if i == pi:
            _eef_pi = robot.data.body_state_w[0, -1, :3].cpu().numpy()
            _cyl_pi = cyl_obj.data.root_state_w[0, :3].cpu().numpy()
            _d_pi   = float(np.linalg.norm(_eef_pi - _cyl_pi))
            print(f"    [PICK_IDX={pi}] EEF={_eef_pi.round(3)}  cyl={_cyl_pi.round(3)}  d={_d_pi*100:.1f}cm", flush=True)
        if i == pl:
            _eef_pl  = robot.data.body_state_w[0, -1, :3].cpu().numpy()
            _tgt_pl  = env.scene["target"].data.root_state_w[0, :3].cpu().numpy()
            _d_pl_xy = float(np.linalg.norm(_eef_pl[:2] - _tgt_pl[:2]))
            _d_pl_3d = float(np.linalg.norm(_eef_pl - _tgt_pl))
            print(f"    [PLACE_IDX={pl}] EEF={_eef_pl.round(3)}  tgt={_tgt_pl.round(3)}"
                  f"  d_xy={_d_pl_xy*100:.1f}cm  d_3d={_d_pl_3d*100:.1f}cm", flush=True)
        gv     = float(states[i, 5])
        closed = (gv < OPEN_THR)
        if closed and not grasping:
            eef = robot.data.body_state_w[0, -1, :3].cpu().numpy()
            cyl = cyl_obj.data.root_state_w[0, :3].cpu().numpy()
            d   = float(np.linalg.norm(eef - cyl))
            if d < 0.05:
                grasping  = True
                grasp_off = cyl - eef
                print(f"    [grasp] frame {i:4d}  EEF={eef.round(3)}  d={d*100:.1f}cm")
        elif not closed and grasping:
            grasping = False
            print(f"    [release] frame {i:4d}")
        if grasping:
            eef = robot.data.body_state_w[0, -1, :3].cpu().numpy()
            st  = cyl_obj.data.root_state_w.clone()
            st[0, :3] = torch.tensor(eef + grasp_off, dtype=torch.float32, device=env.device)
            st[0, 7:] = 0.0
            cyl_obj.write_root_state_to_sim(st)
    cyl_final = env.scene["cylinder"].data.root_state_w[0,:3].cpu().numpy()
    tgt_final = env.scene["target"].data.root_state_w[0,:3].cpu().numpy()
    dist_xy   = float(np.linalg.norm(cyl_final[:2]-tgt_final[:2]))
    success   = dist_xy < args_cli.success_r
    result    = "PASS" if success else "fail"
    elapsed   = time.time()-t0
    # place error (EEF vs target at place frame, captured in loop above)
    place_d_xy  = _d_pl_xy
    place_eef_x = _eef_pl[0] if _eef_pl is not None else float('nan')
    place_eef_y = _eef_pl[1] if _eef_pl is not None else float('nan')
    print(f"{ep_idx:>3}  {cyl_x:>7.3f} {cyl_y:>7.3f}  {tgt_x:>7.3f} {tgt_y:>7.3f}  "
          f"{cyl_final[0]:>9.3f} {cyl_final[1]:>9.3f}  {dist_xy*100:>7.1f}  {result:>6}"
          f"  place_d={place_d_xy*100:.1f}cm  ({elapsed:.0f}s)")
    results.append({"ep":ep_idx,"cyl_x":cyl_x,"cyl_y":cyl_y,"tgt_x":tgt_x,"tgt_y":tgt_y,
                    "end_cyl_x":cyl_final[0],"end_cyl_y":cyl_final[1],"dist_cm":dist_xy*100,"result":result,
                    "place_eef_x":place_eef_x,"place_eef_y":place_eef_y,"place_d_xy_cm":place_d_xy*100})
    sys.stdout.flush()
print("-"*len(HDR))
scored = [r for r in results if r.get("result") in ("PASS","fail")]
passed = [r for r in scored if r["result"]=="PASS"]
if scored:
    print(f"\nSummary: {len(passed)}/{len(scored)} PASSED ({100*len(passed)/len(scored):.1f}%)")
    all_dists=[r["dist_cm"] for r in scored]
    print(f"Final cyl→tgt dist: mean={np.mean(all_dists):.1f}cm  std={np.std(all_dists):.1f}cm")
    place_ds = [r["place_d_xy_cm"] for r in scored if not np.isnan(r.get("place_d_xy_cm", float('nan')))]
    if place_ds:
        print(f"Place EEF→tgt dist: mean={np.mean(place_ds):.1f}cm  std={np.std(place_ds):.1f}cm  n={len(place_ds)}")
out_csv=os.path.expanduser("~/so101-sim-project/tmp/batch_score_results.csv")
pd.DataFrame(results).to_csv(out_csv,index=False)
print(f"Results: {out_csv}")
env.close()
simulation_app.close()
