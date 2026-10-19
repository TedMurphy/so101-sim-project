"""
Compare camera-detected object positions vs FK-computed EEF pick/place positions
across all episodes in V4_pnp_44_20260720_141907.
"""
import numpy as np
import pandas as pd
import subprocess
import math
import warnings
import sys
warnings.filterwarnings("ignore")
from ikpy.chain import Chain

URDF    = "/home/ted/SO101/Misc/so101_new_calib.urdf"
DATA    = "/home/ted/.cache/huggingface/lerobot/Screener2/V4_pnp_44_20260720_141907"
VIDEO   = DATA + "/videos/observation.images.overhead/chunk-000/file-000.mp4"
EP_META = DATA + "/meta/episodes/chunk-000/file-000.parquet"
PARQUET = DATA + "/data/chunk-000/file-000.parquet"
OUT_CSV = "/home/ted/so101-sim-project/tmp/ep_comparison.csv"

IMG_CX, IMG_CY   = 320, 240
SCALE_X, SCALE_Y = 0.000816, 0.001081
OX, OY           = 0.090, 0.008
BASE_XY          = np.array([-0.03, 0.0])

chain = Chain.from_urdf_file(URDF)

def fk_world(state_deg):
    angles = [0] + [math.radians(d) for d in state_deg[:5]] + [0]
    T = chain.forward_kinematics(angles)
    lx, ly = T[0,3], T[1,3]
    return float(-ly + BASE_XY[0]), float(lx + BASE_XY[1])

def pix2world(col, row):
    return OX+(row-IMG_CY)*SCALE_X, OY+(col-IMG_CX)*SCALE_Y

LO_CYL = np.array([35,40,40],np.uint8);  HI_CYL = np.array([85,255,255],np.uint8)
LO_TGT = np.array([15,50,50],np.uint8);  HI_TGT = np.array([38,255,255],np.uint8)

def _ok(wx, wy):
    if 0.22<=wx<=0.38 and 0.17<=wy<=0.38: return False
    if math.hypot(wx-BASE_XY[0], wy-BASE_XY[1])<0.12: return False
    return True

def _find_blob(frame, side, lo, hi):
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
    return pix2world(int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

def detect_ep(vid_offset):
    cp,tp=[],[]
    for dt in [0.0,0.5,1.0,1.5,2.0,2.5,3.0]:
        r=subprocess.run(["ffmpeg","-ss",str(vid_offset+dt),"-i",VIDEO,
                          "-frames:v","1","-f","rawvideo","-pix_fmt","bgr24","pipe:1"],
                         capture_output=True)
        if len(r.stdout)!=640*480*3: continue
        f=np.frombuffer(r.stdout,np.uint8).reshape(480,640,3).copy()
        f[415:,455:]=0
        c=_find_blob(f,"left",LO_CYL,HI_CYL)
        t=_find_blob(f,"right",LO_TGT,HI_TGT)
        if c and _ok(*c): cp.append(c)
        if t and _ok(*t): tp.append(t)
    if not cp or not tp: return None
    return (float(np.median([p[0] for p in cp])),float(np.median([p[1] for p in cp])),
            float(np.median([p[0] for p in tp])),float(np.median([p[1] for p in tp])))

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

df_all  = pd.read_parquet(PARQUET)
ep_meta = pd.read_parquet(EP_META)
n_eps   = df_all["episode_index"].nunique()
print(f"Dataset: {n_eps} episodes, {len(df_all)} total frames\n")

HDR = (f"{'Ep':>3}  {'FK_pk_x':>8} {'FK_pk_y':>8}  {'Det_cy_x':>8} {'Det_cy_y':>8}  {'dX_cyl':>7} {'dY_cyl':>7}  "
       f"{'FK_pl_x':>8} {'FK_pl_y':>8}  {'Det_tg_x':>8} {'Det_tg_y':>8}  {'dX_tgt':>7} {'dY_tgt':>7}")
print(HDR); print("-"*len(HDR))

rows=[]; dxc,dyc,dxt,dyt=[],[],[],[]
for ep_idx in sorted(df_all["episode_index"].unique()):
    df = df_all[df_all["episode_index"]==ep_idx].reset_index(drop=True)
    states = np.array(df["observation.state"].tolist())
    g = states[:,5]
    pi,pl = find_pick_place(g)
    if pi is None or pl is None:
        print(f"{ep_idx:>3}  -- no event (g {g.min():.1f}–{g.max():.1f})")
        rows.append({"ep":ep_idx,"note":"no_event"}); continue
    fpk = fk_world(states[pi]); fpl = fk_world(states[pl])
    voff = float(ep_meta[ep_meta["episode_index"]==ep_idx]["videos/observation.images.overhead/from_timestamp"].iloc[0])
    det = detect_ep(voff)
    if det is None:
        print(f"{ep_idx:>3}  -- det failed  FK_pick=({fpk[0]:.3f},{fpk[1]:.3f})")
        rows.append({"ep":ep_idx,"note":"det_failed","fk_pick_x":fpk[0],"fk_pick_y":fpk[1],"fk_place_x":fpl[0],"fk_place_y":fpl[1]}); continue
    dcx,dcy,dtx,dty2 = det
    dx_c=fpk[0]-dcx; dy_c=fpk[1]-dcy; dx_t=fpl[0]-dtx; dy_t=fpl[1]-dty2
    dxc.append(dx_c); dyc.append(dy_c); dxt.append(dx_t); dyt.append(dy_t)
    print(f"{ep_idx:>3}  {fpk[0]:>8.3f} {fpk[1]:>8.3f}  {dcx:>8.3f} {dcy:>8.3f}  {dx_c:>7.3f} {dy_c:>7.3f}  "
          f"{fpl[0]:>8.3f} {fpl[1]:>8.3f}  {dtx:>8.3f} {dty2:>8.3f}  {dx_t:>7.3f} {dy_t:>7.3f}")
    rows.append({"ep":ep_idx,"fk_pick_x":fpk[0],"fk_pick_y":fpk[1],"det_cyl_x":dcx,"det_cyl_y":dcy,
                 "dX_cyl":dx_c,"dY_cyl":dy_c,"fk_place_x":fpl[0],"fk_place_y":fpl[1],
                 "det_tgt_x":dtx,"det_tgt_y":dty2,"dX_tgt":dx_t,"dY_tgt":dy_t})
    sys.stdout.flush()

print("-"*len(HDR))
if dxc:
    print(f"{'MEAN':>3}  {'':>8} {'':>8}  {'':>8} {'':>8}  {np.mean(dxc):>7.3f} {np.mean(dyc):>7.3f}  "
          f"{'':>8} {'':>8}  {'':>8} {'':>8}  {np.mean(dxt):>7.3f} {np.mean(dyt):>7.3f}")
    print(f"{'STD ':>3}  {'':>8} {'':>8}  {'':>8} {'':>8}  {np.std(dxc):>7.3f} {np.std(dyc):>7.3f}  "
          f"{'':>8} {'':>8}  {'':>8} {'':>8}  {np.std(dxt):>7.3f} {np.std(dyt):>7.3f}")
    print(f"\nSuccessful: {len(dxc)}/{n_eps} episodes")

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"CSV: {OUT_CSV}")
