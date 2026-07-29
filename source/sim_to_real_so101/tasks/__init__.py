# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package containing task implementations."""

from isaaclab_tasks.utils import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)

import gymnasium as gym

gym.register(
    id="Lerobot-So101-Teleop-Cylinder-To-Target",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cylinder_to_target_env_cfg:CylinderToTargetEnvCfg",
    },
)
