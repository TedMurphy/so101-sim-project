# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom MDP functions for the SO-101 cylinder-to-target task.

Everything else (joint_pos, joint_pos_rel, image, reset_joints_by_offset,
reset_root_state_uniform, JointPositionActionCfg) is provided directly by
Isaac Lab's own isaaclab.envs.mdp module -- no need to duplicate it here.
"""
import torch
from pxr import Sdf

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

# Robot color palette
ROBOT_COLORS = {
    "orange": (0.876, 0.317, 0.132),
    "teal": (0.0, 0.8, 0.502),
    "white": (0.95, 0.95, 0.95),
    "black": (0.08, 0.08, 0.08),
}


def ee_frame_state(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the state of the end effector frame in the robot coordinate system."""
    robot = env.scene[robot_cfg.name]
    robot_root_pos, robot_root_quat = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos, ee_frame_quat = ee_frame.data.target_pos_w.torch[:, 0, :], ee_frame.data.target_quat_w.torch[:, 0, :]
    ee_frame_pos_robot, ee_frame_quat_robot = math_utils.subtract_frame_transforms(
        robot_root_pos, robot_root_quat, ee_frame_pos, ee_frame_quat
    )
    ee_frame_state = torch.cat([ee_frame_pos_robot, ee_frame_quat_robot], dim=1)
    return ee_frame_state


def randomize_robot_color(
    env,
    env_ids: torch.Tensor | None,
    color_names: list[str] = list(ROBOT_COLORS.keys()),
):
    """Randomly set robot color from predefined palette on each reset."""
    idx = torch.randint(0, len(color_names), (1,), device="cpu").item()
    selected_color = ROBOT_COLORS[color_names[idx]]

    with Sdf.ChangeBlock():
        robot = env.scene["robot"]
        material_prim_path = robot.cfg.prim_path + "/Looks/material_a_3d_printed/Shader"
        material_prim = sim_utils.find_matching_prims(material_prim_path)[0]
        material_prim.GetAttribute("inputs:diffuse_color_constant").Set(selected_color)
