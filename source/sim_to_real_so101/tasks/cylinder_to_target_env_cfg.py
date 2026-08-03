# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.sensors import TiledCameraCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaaclab.envs.mdp import reset_root_state_uniform, reset_joints_by_offset, image

from sim_to_real_so101 import assets

from .so101_env_cfg import (
    LerobotSo101BaseSceneCfg,
    ObservationsCfg,
    SO101TeleopEnvCfg,
)

assets_path = os.path.dirname(os.path.abspath(assets.__file__))

CAMERA_HEIGHT_ABOVE_TABLE = 0.39
CAMERA_FOV_DEG = 73.0
CAMERA_APERTURE_MM = 20.955
CAMERA_FOCAL_LENGTH_MM = CAMERA_APERTURE_MM / (2 * np.tan(np.deg2rad(CAMERA_FOV_DEG / 2)))

MAT_POS = (0.22, 0.0, 0.032)

CYLINDER_HEIGHT = 0.06
TARGET_HEIGHT = 0.002

ZONE_Y_OFFSET = 0.175
ZONE_HALF_SIZE = 0.125

CYLINDER_ZONE_CENTER = (MAT_POS[0], MAT_POS[1] + ZONE_Y_OFFSET, MAT_POS[2] + CYLINDER_HEIGHT / 2)
TARGET_ZONE_CENTER = (MAT_POS[0], MAT_POS[1] - ZONE_Y_OFFSET, MAT_POS[2] + TARGET_HEIGHT / 2)

overhead_camera = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/CameraOverhead",
    update_period=1 / 30,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        projection_type="pinhole",
        focal_length=CAMERA_FOCAL_LENGTH_MM,
        horizontal_aperture=CAMERA_APERTURE_MM,
        focus_distance=CAMERA_HEIGHT_ABOVE_TABLE,
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(MAT_POS[0], MAT_POS[1], MAT_POS[2] + CAMERA_HEIGHT_ABOVE_TABLE),
        rot=euler_angles_to_quat(np.array([0, 0, 0]), degrees=True),
        convention="opengl",
    ),
)


@configclass
class CylinderToTargetSceneCfg(LerobotSo101BaseSceneCfg):

    dome_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0)),
    )

    mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/mat.usda",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.85), roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=MAT_POS,
            rot=euler_angles_to_quat(np.array([0, 0, 90]), degrees=True),
        ),
    )

    camera_overhead = overhead_camera

    cylinder: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/Cylinder.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CYLINDER_ZONE_CENTER),
    )

    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/TargetCircle.usd",
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
        self.viewer.eye = (MAT_POS[0] - 0.3, MAT_POS[1], MAT_POS[2] + 0.4)
        self.viewer.lookat = (MAT_POS[0], MAT_POS[1], MAT_POS[2])
