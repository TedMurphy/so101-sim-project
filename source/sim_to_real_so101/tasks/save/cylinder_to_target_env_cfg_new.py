# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import numpy as np
from scipy.spatial.transform import Rotation as _R

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.sensors import TiledCameraCfg
def euler_angles_to_quat(euler_angles, degrees=False, extrinsic=False):
    """Scipy drop-in for isaacsim.core.utils.rotations.euler_angles_to_quat."""
    import numpy as _np
    angles = _np.radians(euler_angles) if degrees else _np.asarray(euler_angles, dtype=float)
    order = 'xyz' if extrinsic else 'XYZ'   # uppercase = intrinsic in scipy
    q = _R.from_euler(order, angles).as_quat()   # returns [x, y, z, w]
    return _np.array([q[3], q[0], q[1], q[2]])   # convert to [w, x, y, z]
from isaaclab.envs.mdp import reset_root_state_uniform, reset_joints_by_offset, image

from sim_to_real_so101 import assets

from .so101_env_cfg import (
    LerobotSo101BaseSceneCfg,
    ObservationsCfg,
    SO101TeleopEnvCfg,
)

assets_path = os.path.dirname(os.path.abspath(assets.__file__))

CAMERA_HEIGHT_ABOVE_TABLE = 0.60     # was 0.57
CAMERA_FOV_DEG = 69.0
CAMERA_APERTURE_MM = 27.0
CAMERA_FOCAL_LENGTH_MM = CAMERA_APERTURE_MM / (2 * np.tan(np.deg2rad(CAMERA_FOV_DEG / 2)))

# ── Camera tilt controls ──────────────────────────────────────────────────────
#   Both values are INCREMENTAL degrees added on top of the baseline quaternion.
#   At 0 / 0 the image is identical to the original working camera view.
#
#   TILT_X_DEG : forward/back pitch — positive tilts camera top toward robot
#                increases the bottom-to-top size ratio; negative reverses it
#   TILT_Y_DEG : left/right roll   — positive tilts mat clockwise in image
#
#   Start at 0, 0 to confirm you get the original image, then nudge from there.
TILT_X_DEG = 0.0
TILT_Y_DEG = 0.0

# Baseline quaternion [x, y, z, w] — the original working rotation. Do not edit.
_BASE_ROT_XYZW = np.array([0.0273, 0.0693, 0.9971, 0.0156])

def _cam_rot(tilt_x_deg, tilt_y_deg):
    """Apply incremental X/Y tilt on top of the baseline quaternion."""
    r_base  = _R.from_quat(_BASE_ROT_XYZW)
    r_delta = _R.from_euler('XY', [tilt_x_deg, tilt_y_deg], degrees=True)
    r_total = r_delta * r_base   # intrinsic: delta applied in base frame
    q = r_total.as_quat()        # scipy → [x, y, z, w]
    return np.array([q[0], q[1], q[2], q[3]])
# ─────────────────────────────────────────────────────────────────────────────

TABLE_Z = 0.035  # table surface Z for object spawning

CYLINDER_HEIGHT = 0.06
TARGET_HEIGHT = 0.050

# Object zone centres in Isaac Lab world frame (FK vs detection, 10 episodes).
ZONE_HALF_SIZE  = 0.040  # ±4 cm randomisation radius

CYLINDER_ZONE_CENTER = ( 0.220,  0.210, TABLE_Z + CYLINDER_HEIGHT / 2)  # green, left half
TARGET_ZONE_CENTER   = (-0.260,  0.175, 0.200)  # yellow, right half

overhead_camera = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/CameraOverhead",
    update_period=0.0,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        projection_type="pinhole",
        f_stop=100,
        focal_length=21,          # was 18.14756 — matches real camera zoom
        horizontal_aperture=CAMERA_APERTURE_MM,
        focus_distance=CAMERA_HEIGHT_ABOVE_TABLE,
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.005, 0.320, 0.635),   # matches real camera position
        rot=_cam_rot(TILT_X_DEG, TILT_Y_DEG),
        convention="opengl",
    ),
)


@configclass
class CylinderToTargetSceneCfg(LerobotSo101BaseSceneCfg):

    ground_plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(color=(0.25, 0.25, 0.25)),
    )
    dome_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)),
    )

    mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/mat.usda",
            scale=(1.0, 1.0, 1.0),   # real mat: 0.76m X x 0.42m Y
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95), roughness=0.4),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.035, 0.21, 0.032),  # world centre: X=(-0.415+0.345)/2, Y=(0+0.42)/2
            rot=np.array([0.0, 0.0, 1.0, 0.0]),  # Rz(180 deg) XYZW: wide 76cm end near robot
        ),
    )

    camera_overhead = overhead_camera

    cylinder: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/Cylinder.usd",
            scale=(1.0, 1.0, 2.0),  # 3cm dia x 6cm tall
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(CYLINDER_ZONE_CENTER[0], CYLINDER_ZONE_CENTER[1], CYLINDER_ZONE_CENTER[2])
        ),
    )

    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/TargetCircle.usd",
            scale=(1.0, 1.0, 50.0),  # 9cm outer dia x 5cm tall
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TARGET_ZONE_CENTER),
    )


@configclass
class CylinderToTargetEventCfg:

    reset_robot_position = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
            ),
            "position_range": (0, 0),
            "velocity_range": (0, 0),
        },
    )

    reset_cylinder_position = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-ZONE_HALF_SIZE, ZONE_HALF_SIZE), "y": (-ZONE_HALF_SIZE, ZONE_HALF_SIZE), "yaw": (-np.pi, np.pi)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cylinder"),
        },
    )

    reset_target_position = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-ZONE_HALF_SIZE, ZONE_HALF_SIZE), "y": (-ZONE_HALF_SIZE, ZONE_HALF_SIZE)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("target"),
        },
    )


@configclass
class CylinderToTargetObservationsCfg(ObservationsCfg):

    @configclass
    class VisualCfg(ObsGroup):
        rgb_overhead = ObsTerm(
            func=image,
            params={"sensor_cfg": SceneEntityCfg("camera_overhead"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    visual: VisualCfg = VisualCfg()


@configclass
class CylinderToTargetEnvCfg(SO101TeleopEnvCfg):

    scene: CylinderToTargetSceneCfg = CylinderToTargetSceneCfg()
    events: CylinderToTargetEventCfg = CylinderToTargetEventCfg()
    observations: CylinderToTargetObservationsCfg = CylinderToTargetObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.render.enable_translucency = True
        self.sim.render.carb_settings = {
            "rtx.reflections.enabled": True,
            "rtx.translucency.reflectAtAllBounce": True,
            "rtx.translucency.sampleRoughness": True,
            "rtx.translucency.reflectionThroughputThreshold": 0.05,
            "rtx.translucency.maxRefractionBounces": 5,
            "rtx.raytracing.fractionalCutoutOpacity": True,
        }
        self.viewer.eye = (-0.035 - 0.35, 0.21 + 0.001, 0.032 + 0.55)
        self.viewer.lookat = (-0.035 + 0.15, 0.21, 0.032)