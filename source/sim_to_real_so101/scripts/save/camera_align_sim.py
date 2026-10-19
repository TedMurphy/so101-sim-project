# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Isaac Sim scene for TiledCamera alignment.

Run with:
  python3 ~/isaac-sim/python.sh \
      ~/so101-sim-project/source/sim_to_real_so101/scripts/camera_align_sim.py

Instructions:
  1. Scene opens: white mat, green cylinder (left/+Y), yellow cup (right/-Y).
  2. Orbit the viewport so it matches the real EMEET S600 overhead view:
       - Robot gripper at top-center
       - Green cylinder lower-left
       - Yellow cup lower-right
  3. Press C in the viewport (or terminal) to print Isaac Lab config values.
  4. Ctrl-C to exit.
"""

import signal
import sys
from isaacsim import SimulationApp

app = SimulationApp({"headless": False, "renderer": "RaytracedLighting"})

import numpy as np
import omni.usd
import omni.appwindow
import carb.input
from pxr import UsdGeom, UsdLux, Gf, Vt

stage = omni.usd.get_context().get_stage()


# ── helpers ────────────────────────────────────────────────────────────────────

def make_cube(path, pos, scale, color=(0.8, 0.8, 0.8)):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddScaleOp().Set(Gf.Vec3f(*scale))
    cube.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))


def make_cylinder(path, pos, radius, height, color=(0.8, 0.8, 0.8)):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.GetRadiusAttr().Set(radius)
    cyl.GetHeightAttr().Set(height)
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    cyl.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))


def quat_to_wxyz(q: Gf.Quatd):
    """Return (w, x, y, z) from a Gf.Quatd."""
    im = q.GetImaginary()
    return (q.GetReal(), im[0], im[1], im[2])


def read_and_print_camera():
    """Read viewport camera transform and print Isaac Lab config snippet."""
    try:
        from omni.kit.viewport.utility import get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        vp = get_active_viewport()
        if vp is None:
            print("[!] No active viewport found.")
            return

        cam_path = vp.camera_path
        cs = ViewportCameraState(cam_path, vp)

        # Position (world space = local space when robot is at origin)
        p = cs.position_world          # Gf.Vec3d
        pos = (round(p[0], 4), round(p[1], 4), round(p[2], 4))

        # Orientation: read from USD prim's xformOp:orient
        cam_prim = stage.GetPrimAtPath(str(cam_path))
        orient_attr = cam_prim.GetAttribute("xformOp:orient")
        if orient_attr and orient_attr.IsValid():
            q = orient_attr.Get()          # Gf.Quatd or Gf.Quatf
            if hasattr(q, 'GetReal'):
                w, x, y, z = quat_to_wxyz(Gf.Quatd(q))
            else:
                w, x, y, z = 1.0, 0.0, 0.0, 0.0
        else:
            # Fall back: build from position + target
            t  = cs.target_world
            fwd = Gf.Vec3d(t[0]-p[0], t[1]-p[1], t[2]-p[2]).GetNormalized()
            # simple downward-looking approximation
            w, x, y, z = 0.0, 0.0, 0.0, 1.0

        print()
        print("=" * 60)
        print("  Copy-paste into cylinder_to_target_env_cfg.py:")
        print("=" * 60)
        print(f"  offset=TiledCameraCfg.OffsetCfg(")
        print(f"      pos={pos},")
        print(f"      rot=np.array([{w:.6f}, {x:.6f}, {y:.6f}, {z:.6f}]),")
        print(f"      convention=\"opengl\",")
        print(f"  ),")
        print()
        print(f"  Camera world position : {pos}")
        print(f"  Camera quaternion WXYZ: ({w:.6f}, {x:.6f}, {y:.6f}, {z:.6f})")
        print("=" * 60)
        print()

    except Exception as e:
        print(f"[!] read_and_print_camera failed: {e}")
        import traceback; traceback.print_exc()


# ── Build scene ────────────────────────────────────────────────────────────────

# Coordinate frame matches Isaac Lab env cfg:
#   Robot base at (0, 0, 0)
#   Mat center at (0.22, 0.0, 0.032)  -- 32mm thick white foam mat
#   Cylinder at   (0.22, +0.10, ...)  -- green, left in real image
#   Target at     (0.22, -0.10, ...)  -- yellow, right in real image

MAT_POS   = (0.22, 0.0,   0.016)   # half-height above ground
MAT_SCALE = (0.25, 0.20,  0.016)   # 50 cm × 40 cm × 3.2 cm

# White mat
make_cube("/World/Mat", MAT_POS, MAT_SCALE, color=(0.95, 0.95, 0.95))

# Blue tape corner markers
for (sx, sy), label in [((1,1),"NE"),((1,-1),"SE"),((-1,1),"NW"),((-1,-1),"SW")]:
    make_cube(f"/World/Tape{label}",
              (MAT_POS[0] + sx*0.115, MAT_POS[1] + sy*0.085, MAT_POS[2]+0.018),
              (0.012, 0.012, 0.002),
              color=(0.05, 0.15, 0.85))

# Green cylinder (left in real image = +Y in sim)
CYL_H = 0.06
make_cylinder("/World/GreenCylinder",
              (0.22, 0.10, CYL_H / 2),
              radius=0.02, height=CYL_H,
              color=(0.05, 0.75, 0.10))

# Yellow cup (right in real image = -Y in sim)
make_cylinder("/World/YellowCup",
              (0.22, -0.10, 0.025),
              radius=0.035, height=0.05,
              color=(0.95, 0.82, 0.05))

# Robot base stub at origin
make_cube("/World/RobotBase", (0.0, 0.0, 0.04), (0.06, 0.06, 0.04), color=(0.25, 0.25, 0.25))
# Arm approximation
make_cube("/World/RobotArm",  (0.08, 0.0, 0.18), (0.018, 0.018, 0.18), color=(0.35, 0.35, 0.35))

# Ground plane
ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
ground.GetPointsAttr().Set(Vt.Vec3fArray([
    Gf.Vec3f(-3,-3,0), Gf.Vec3f(3,-3,0), Gf.Vec3f(3,3,0), Gf.Vec3f(-3,3,0)]))
ground.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
ground.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0,1,2,3]))
ground.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.45, 0.42, 0.38)]))

# Lighting
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.GetIntensityAttr().Set(800.0)

# ── Input: press C to print config ────────────────────────────────────────────
_input     = carb.input.acquire_input_interface()
_app_win   = omni.appwindow.get_default_app_window()
_keyboard  = _app_win.get_keyboard()
_print_req = False

def _on_key(event, *_):
    global _print_req
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input == carb.input.KeyboardInput.C:
            _print_req = True

_sub = _input.subscribe_to_keyboard_events(_keyboard, _on_key)

# ── Pump a few frames then print instructions ──────────────────────────────────
for _ in range(5):
    app.update()

print()
print("=" * 60)
print("  CAMERA ALIGNMENT SCENE — Isaac Sim")
print("=" * 60)
print("  White mat   : X=0.22, Y= 0.00  (center)")
print("  Green cyl   : X=0.22, Y=+0.10  → LEFT  in real image")
print("  Yellow cup  : X=0.22, Y=-0.10  → RIGHT in real image")
print("  Robot stub  : at origin (0,0,0) → TOP-CENTER in real image")
print()
print("  1. Orbit viewport overhead until the scene matches the real")
print("     EMEET S600 view (gripper top, cylinder lower-left,")
print("     yellow cup lower-right, mat filling most of frame).")
print("  2. Press  C  (in the viewport or terminal) to print the")
print("     Isaac Lab config snippet.")
print("  3. Ctrl-C to exit.")
print("=" * 60)
print()

# ── Main loop ─────────────────────────────────────────────────────────────────
while app.is_running():
    if _print_req:
        _print_req = False
        read_and_print_camera()
    app.update()

_input.unsubscribe_to_keyboard_events(_keyboard, _sub)
app.close()
