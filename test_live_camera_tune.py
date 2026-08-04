import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([
    '--livestream', '2',
    '--kit_args', '--ext-folder /home/ted/isaac-sim/extsDeprecated --enable isaacsim.core.utils --/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize=true',
])
args_cli.enable_cameras = True
args_cli.device = 'cpu'

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks

env_cfg = parse_env_cfg('Lerobot-So101-Teleop-Cylinder-To-Target', device='cpu', num_envs=None, use_fabric=True)
env = gym.make('Lerobot-So101-Teleop-Cylinder-To-Target', cfg=env_cfg)
print('SUCCESS: environment created -- connect WebRTC now and interact with the stage')
env.reset()

print("Cylinder world pos:", env.unwrapped.scene["cylinder"].data.root_pos_w.torch)
print("Robot base world pos:", env.unwrapped.scene["robot"].data.root_pos_w.torch)

from pxr import Usd, UsdGeom
stage = env.unwrapped.scene.stage
mat_prim = stage.GetPrimAtPath("/World/envs/env_0/Mat")
xform_cache = UsdGeom.XformCache()
mat_world_transform = xform_cache.GetLocalToWorldTransform(mat_prim)
print("Mat world transform:", mat_world_transform)
print("Mat world position:", mat_world_transform.ExtractTranslation())

while simulation_app.is_running():
    simulation_app.update()
