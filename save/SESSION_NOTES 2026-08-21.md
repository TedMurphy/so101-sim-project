# SO-101 Sim-to-Real Session Notes
_Last updated: 2026-08-21_

## Key Findings

### 1. The "15 cm kinematic mismatch" was a misinterpretation
- Sim EEF world-X at pick = ~20 cm — this is just the X component
- The arm is rotated ~48-50° toward -Y (shoulder_pan=50° URDF)
- Total horizontal reach from robot base = sqrt((20.2+3)² + 25.5²) ≈ **34.5 cm**
- Real arm tape measurement = **35.8 cm** (diagonal distance, not forward X distance)
- These match. The arm kinematics are correct.

### 2. Root cause of pick failure: wrong cylinder EPISODE_POSITIONS
- Original episode 7 cylinder: (+0.284, -0.279, 0.050) — 8 cm too far in world X
- Bad DX correction moved it to (+0.344, -0.259, 0.050) — even worse
- **Fixed value: (+0.233, -0.263, 0.050)** — tuned by replay observation
- Someone likely used the 35.8 cm tape measure reading as the world-X distance,
  but it's actually the diagonal distance at ~48° angle. World X ≈ 20-23 cm.

### 3. lerobot_interface.py — current correct state
File: `~/so101-sim-project/source/sim_to_real_so101/utils/lerobot_interface.py`

```python
def get_mapped_actions_vectorized(self, raw_values):
    normalized = torch.zeros_like(raw_values)
    normalized[:-1] = (raw_values[:-1] + 100) / 200.0
    normalized[-1] = raw_values[-1] / 100.0
    mapped_deg = self.joint_mins + normalized * (self.joint_maxs - self.joint_mins)
    # mapped_deg[0] = -mapped_deg[0]  # disabled: objects swapped to correct side instead
    mapped_deg[4] -= 90.0  # Wrist_Roll offset: sim jaw opens side-to-side like real robot
    return mapped_deg * torch.pi / 180
```

**DO NOT add `mapped_deg[1] = -mapped_deg[1]`** — this was tried and makes arm go behind robot.

### 4. Sweep data (diagnostic reference)
Shoulder sweep (shoulder_pan=50°, elbow=-30°, URDF degrees):
- Max EEF_X = 20.47 cm at shoulder_lift=50°

Elbow sweep (shoulder_pan=50°, shoulder_lift=59°, URDF degrees):
- Max Dist_XY = 34.13 cm at elbow=-50°
- At elbow=-30° (pick): EEF_X=20.20 cm, Dist_XY=32.53 cm, EEF_Z=10.24 cm

### 5. Robot configuration
- Base: pos=(-0.03, 0.0, 0.005), rot=(0.0, 0.0, 0.7071, 0.7071) (Rz+90°)
- At all-zero joints: EEF (jaw) at X=27.12 cm, Y=3.94 cm, Z=29.11 cm world coords
- All USD joints use physics:axis = Z
- Shoulder_pan (Rotation) axis = world -Z; positive rotates arm toward -Y
- Shoulder_lift (Pitch) axis = world +Y; positive tilts arm forward/down
- Elbow axis = world +Y

### 6. Episode 7 pick frame joint angles (dataset degrees)
Frame 156: [45.45°, 58.99°, -26.37°, -7.12°, 5.67°, 31.42°]
After get_mapped_actions_vectorized → URDF degrees:
- shoulder_pan = 50°, shoulder_lift = 59°, elbow = -30°
- wrist_flex = -6.8°, wrist_roll = -80.9°, jaw = 24.6°

### 7. Camera setup (overhead)
- Position: (0.21, 0.025, 0.52) meters
- image-right = world +Y, image-down = world +X
- H_APERTURE=36mm, V_APERTURE=22.5mm (4:3)

### 8. USD/URDF details
- File: `~/so101-sim-project/source/sim_to_real_so101/assets/usd/SO-ARM101-USD.usd`
- metersPerUnit=1.0, upAxis=Z
- URDF: so101_new_calib.urdf baked into USD

### 9. Dataset
- V4_pnp_44_20260720_141907
- Stores joint angles in degrees, approximately [-100, 100] range
- Captured with orange_robot.json calibration

### 10. Constraints
- DO NOT downgrade to Isaac Lab 2.3.0 — hard segfault on Blackwell GPU
- Use `--livestream 2` for WebRTC display
- Use `./isaaclab.sh -p` from `~/IsaacLab/`, not `python3`
- Cup positions must come from camera/video, NOT from EEF

### 11. TODO / In progress
- [ ] Apply image mask for off-mat yellow bucket (bottom-right corner of overhead camera)
  - Bucket at episode 0 appears at approx rows 375-480, cols 490-640 in overhead image
  - Need to insert `_f[375:, 490:] = 0` before `.save()` in replay_interactive.py line ~306
- [ ] Verify other episodes' cylinder positions are reasonable
- [ ] Confirm target (bucket) positions are correct for all episodes
