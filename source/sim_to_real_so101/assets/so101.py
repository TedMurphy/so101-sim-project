# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

here = os.path.dirname(os.path.abspath(__file__))

SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{here}/usd/full_robot_rebuild.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "Rotation": -0.2736,
            "Pitch": -0.6109,
            "Elbow": -0.0745,
            "Wrist_Pitch": 1.5148,
            "Wrist_Roll": -1.6034,
            "Jaw": -0.1465,
        },
        pos=(-0.05, 0.0, 0),
    ),
    actuators={
        "rotation": ImplicitActuatorCfg(joint_names_expr=["Rotation"], effort_limit_sim=30, stiffness=55, damping=0.7),
        "pitch": ImplicitActuatorCfg(joint_names_expr=["Pitch"], effort_limit_sim=30, stiffness=30, damping=0.8),
        "elbow": ImplicitActuatorCfg(joint_names_expr=["Elbow"], effort_limit_sim=30, stiffness=25, damping=0.7),
        "wrist_pitch": ImplicitActuatorCfg(joint_names_expr=["Wrist_Pitch"], effort_limit_sim=30, stiffness=12, damping=0.5),
        "wrist_roll": ImplicitActuatorCfg(joint_names_expr=["Wrist_Roll"], effort_limit_sim=30, stiffness=7, damping=0.5),
        "gripper": ImplicitActuatorCfg(joint_names_expr=["Jaw"], effort_limit_sim=30, stiffness=4, damping=0.3),
    },
)

S0101_NO_CAMERA_CFG = SO101_CFG.copy()
S0101_CONTACT_GRASP_CFG = SO101_CFG.copy()
S0101_CONTACT_GRASP_CFG.spawn.activate_contact_sensors = True
