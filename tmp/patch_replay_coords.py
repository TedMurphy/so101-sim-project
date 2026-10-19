#!/usr/bin/env python3
fpath = "/home/ted/so101-sim-project/source/sim_to_real_so101/scripts/replay_interactive.py"
with open(fpath, "r") as f:
    src = f.read()
errors = []

TABLE_START = 'EPISODE_POSITIONS = {'
TABLE_END   = '\ndef set_viewport_camera'
if TABLE_START in src and TABLE_END in src:
    i = src.index(TABLE_START)
    j = src.index(TABLE_END)
    src = src[:i] + "# EPISODE_POSITIONS table removed — positions detected live from video.\n" + src[j:]
    print("OK: removed EPISODE_POSITIONS table.")
else:
    errors.append("SKIP: EPISODE_POSITIONS table not found.")

OLD_CONSTS = ("_DET_IMG_CX  = 320        # image centre col  (pixels)\n"
    "_DET_IMG_CY  = 240        # image centre row  (pixels)\n"
    "_DET_SCALE_X = 0.000816   # m/pixel  (image row  \u2192 world X)\n"
    "_DET_SCALE_Y = 0.001081   # m/pixel  (image col  \u2192 world Y)\n"
    "_DET_OX      = 0.090      # world X at image centre\n"
    "_DET_OY      = 0.008      # world Y at image centre")
NEW_CONSTS = ("_DET_IMG_CX  = 320   # image centre col (pixels)\n"
    "_DET_IMG_CY  = 240   # image centre row (pixels)\n"
    "# Camera -> Isaac Lab world frame calibration.\n"
    "# Physical: camera 57.5 cm above table; robot base world(-0.03,0,0.005) Rz+90deg.\n"
    "# Rows -> world Y (arm reach, 0=robot base, +Y into mat).\n"
    "# Cols -> world X (lateral, negated; centre col ~ world X -0.025).\n"
    "_DET_CAM_SCALE_ROW = 0.000816   # m/pixel  (row  -> world Y)\n"
    "_DET_CAM_SCALE_COL = 0.001081   # m/pixel  (col  -> world X, negated)\n"
    "_DET_CAM_WORLD_X0  = -0.025     # Isaac Lab world X at image centre column\n"
    "_DET_CAM_WORLD_Y0  =  0.197     # Isaac Lab world Y at image centre row")
if OLD_CONSTS in src:
    src = src.replace(OLD_CONSTS, NEW_CONSTS, 1); print("OK: updated calibration constants.")
else:
    errors.append("ERROR: calibration constants block not found.")

OLD_EXCL = "_DET_EXCL   = dict(x_min=0.22, x_max=0.38, y_min=0.17, y_max=0.38)"
NEW_EXCL = ("# Exclusion zone in Isaac Lab world frame (bottom-right of real image).\n"
    "_DET_EXCL   = dict(x_min=-0.400, x_max=-0.185, y_min=0.325, y_max=0.490)")
if OLD_EXCL in src:
    src = src.replace(OLD_EXCL, NEW_EXCL, 1); print("OK: updated _DET_EXCL to world frame.")
else:
    errors.append("ERROR: _DET_EXCL line not found.")

OLD_PIX = ('def _det_pix2world(px_col, py_row):\n'
    '    """Pixel (col, row) \u2192 (world_X, world_Y)."""\n'
    '    wx = _DET_OX + (py_row - _DET_IMG_CY) * _DET_SCALE_X\n'
    '    wy = _DET_OY + (px_col - _DET_IMG_CX) * _DET_SCALE_Y\n'
    '    return wx, wy')
NEW_PIX = ('def _det_pix2world(px_col, py_row):\n'
    '    """Pixel (col, row) -> Isaac Lab world (world_x, world_y).\n'
    '    Rows -> world Y (arm reach). Cols -> world X (lateral, negated).\n'
    '    """\n'
    '    world_x = _DET_CAM_WORLD_X0 - (px_col - _DET_IMG_CX) * _DET_CAM_SCALE_COL\n'
    '    world_y = _DET_CAM_WORLD_Y0 + (py_row - _DET_IMG_CY) * _DET_CAM_SCALE_ROW\n'
    '    return world_x, world_y')
if OLD_PIX in src:
    src = src.replace(OLD_PIX, NEW_PIX, 1); print("OK: updated _det_pix2world.")
else:
    errors.append("ERROR: _det_pix2world body not found.")

OLD_SET = ('def set_object_positions(env, episode_idx, real_root=None):\n'
    '    """Override cylinder & target world positions after env.reset().\n'
    '    Runs dynamic video detection first; falls back to EPISODE_POSITIONS table."""\n'
    '    pos = None\n'
    '    if real_root is not None:\n'
    '        pos = detect_episode_objects(episode_idx, real_root)\n'
    '    if pos is None:\n'
    '        pos = EPISODE_POSITIONS.get(episode_idx)\n'
    '        if pos is not None:\n'
    '            print(f"[INFO]: Fallback: using table positions for episode {episode_idx}.")\n'
    '    if pos is None:\n'
    '        print(f"[INFO]: No positions for episode {episode_idx}, using randomised placement.")\n'
    '        return')
NEW_SET = ('def set_object_positions(env, episode_idx, real_root=None):\n'
    '    """Override cylinder & target world positions after env.reset().\n'
    '    Detects positions from real episode video (world frame).\n'
    '    Falls back to randomised placement if detection fails.\n'
    '    """\n'
    '    pos = None\n'
    '    if real_root is not None:\n'
    '        pos = detect_episode_objects(episode_idx, real_root)\n'
    '    if pos is None:\n'
    '        print(f"[INFO]: Detection failed for episode {episode_idx} -- using randomised placement.")\n'
    '        return')
if OLD_SET in src:
    src = src.replace(OLD_SET, NEW_SET, 1); print("OK: simplified set_object_positions.")
else:
    errors.append("ERROR: set_object_positions block not found.")

OLD_CAM = ('    CAM_X      = 0.21\n'
    '    CAM_Y      = 0.025\n'
    '    CAM_Z      = 0.52\n'
    '    TILT_DEG   = 5.0')
NEW_CAM = ('    # Physical camera: world(-0.025, 0.197, 0.575), Rz(180deg).\n'
    '    CAM_X      = -0.025\n'
    '    CAM_Y      =  0.197\n'
    '    CAM_Z      =  0.575\n'
    '    TILT_DEG   = 5.0')
if OLD_CAM in src:
    src = src.replace(OLD_CAM, NEW_CAM, 1); print("OK: updated viewport camera position.")
else:
    errors.append("ERROR: CAM_X block not found.")

OLD_QB = ('    # q_base: 90 deg about world +Z (image-up = world -X, image-right = world +Y)\n'
    '    _w2, _x2, _y2, _z2 = 0.7071, 0.0, 0.0, 0.7071')
NEW_QB = ('    # q_base: Rz(180deg) -- image-right=world-X, image-down=world+Y (matches real cam).\n'
    '    _w2, _x2, _y2, _z2 = 0.0, 0.0, 0.0, 1.0')
if OLD_QB in src:
    src = src.replace(OLD_QB, NEW_QB, 1); print("OK: updated viewport camera rotation to Rz(180).")
else:
    errors.append("ERROR: q_base line not found.")

if errors:
    print("\nISSUES (file NOT written):"); [print(" ", e) for e in errors]
else:
    with open(fpath, "w") as f: f.write(src)
    print("\nFile written. Verifying...")
    checks = ["_DET_CAM_WORLD_X0", "world_x = _DET_CAM_WORLD_X0",
              "Detection failed for episode", "CAM_X      = -0.025",
              "_w2, _x2, _y2, _z2 = 0.0, 0.0, 0.0, 1.0"]
    [print(f"  [{'FOUND' if c in src else 'MISSING'}] {c}") for c in checks]
