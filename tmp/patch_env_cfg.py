#!/usr/bin/env python3
fpath = "/home/ted/so101-sim-project/source/sim_to_real_so101/tasks/cylinder_to_target_env_cfg.py"
with open(fpath, "r") as f:
    src = f.read()
errors = []

OLD_H = "CAMERA_HEIGHT_ABOVE_TABLE = 0.57"
if OLD_H in src:
    src = src.replace(OLD_H, "CAMERA_HEIGHT_ABOVE_TABLE = 0.575  # physical: 57.5 cm", 1)
    print("OK: fixed camera height.")
else:
    errors.append("SKIP: camera height already updated.")

OLD_OFF = ('    offset=TiledCameraCfg.OffsetCfg(\n'
    '        pos=(0.20, 0.005, 0.602),\n'
    '        rot=np.array([0, 0, 0.7071, 0.7071]),\n'
    '        convention="opengl",\n'
    '    ),')
NEW_OFF = ('    offset=TiledCameraCfg.OffsetCfg(\n'
    '        # Physical camera world(-0.025, 0.197, 0.575), Rz(180deg).\n'
    '        # image-right=world-X, image-down=world+Y matches real camera.\n'
    '        pos=(-0.025, 0.197, 0.575),\n'
    '        rot=np.array([0, 0, 0, 1]),  # (w,x,y,z) Rz(180deg)\n'
    '        convention="opengl",\n'
    '    ),')
if OLD_OFF in src:
    src = src.replace(OLD_OFF, NEW_OFF, 1); print("OK: fixed TiledCamera position+rotation.")
else:
    errors.append("ERROR: TiledCamera OffsetCfg not found.")

OLD_Z = ('MAT_POS = (0.14, 0.025, 0.032)  # shifted +0.08: mat now spans X=0.02\u21920.42\n'
    'CYLINDER_HEIGHT = 0.06\n'
    'TARGET_HEIGHT = 0.050\n'
    'ZONE_Y_OFFSET = 0.26\n'
    'ZONE_HALF_SIZE = 0.040\n'
    'CYLINDER_ZONE_CENTER = (MAT_POS[0], MAT_POS[1] - ZONE_Y_OFFSET, MAT_POS[2] - 0.012 + CYLINDER_HEIGHT / 2)\n'
    'TARGET_ZONE_CENTER = (MAT_POS[0], MAT_POS[1] + ZONE_Y_OFFSET, MAT_POS[2] - 0.012 + TARGET_HEIGHT / 2)')
NEW_Z = ('TABLE_Z         = 0.032\n'
    'CYLINDER_HEIGHT = 0.06\n'
    'TARGET_HEIGHT   = 0.050\n'
    'ZONE_HALF_SIZE  = 0.040  # +/-4 cm randomisation for new episode generation\n'
    '# Object zone centres in Isaac Lab world frame (from FK vs detection, 10 episodes).\n'
    '# World X = lateral (robot-left +, robot-right -). World Y = arm reach (0=base, +=into mat).\n'
    'CYLINDER_ZONE_CENTER = ( 0.220,  0.210, TABLE_Z + CYLINDER_HEIGHT / 2)  # green, left image half\n'
    'TARGET_ZONE_CENTER   = (-0.260,  0.175, TABLE_Z + TARGET_HEIGHT   / 2)  # yellow, right image half')
if OLD_Z in src:
    src = src.replace(OLD_Z, NEW_Z, 1); print("OK: fixed zone centres to world frame.")
else:
    errors.append("ERROR: zone centre block not found exactly.")

if errors:
    print("\nISSUES (file NOT written):"); [print(" ", e) for e in errors]
else:
    with open(fpath, "w") as f: f.write(src)
    print("\nFile written. Verifying...")
    checks = ["CAMERA_HEIGHT_ABOVE_TABLE = 0.575", "pos=(-0.025, 0.197, 0.575)",
              "rot=np.array([0, 0, 0, 1])", "CYLINDER_ZONE_CENTER = ( 0.220",
              "TARGET_ZONE_CENTER   = (-0.260"]
    [print(f"  [{'FOUND' if c in src else 'MISSING'}] {c}") for c in checks]
