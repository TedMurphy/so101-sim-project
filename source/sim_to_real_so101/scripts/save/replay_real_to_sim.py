# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay real teleop action sequences into simulation, capturing new
simulated visual observations. Extends a real LeRobot dataset with sim
episodes that share the same recorded actions but new (simulated) camera
frames and resulting robot state -- no physical robot required."""
import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay real dataset actions into Isaac Lab sim.")
parser.add_argument("--task", type=str, default="Lerobot-So101-Teleop-Cylinder-To-Target")
parser.add_argument("--real_repo_id", type=str, default="Screener2/V4_pnp_44_20260720_141907")
parser.add_argument("--out_repo_id", type=str, default="so101_sim_extension")
parser.add_argument("--out_root", type=str, default=os.path.expanduser("~/so101_sim_extension_dataset"))
parser.add_argument("--task_name", type=str, default="pick_and_place_sim")
parser.add_argument("--start_episode", type=int, default=0)
parser.add_argument("--end_episode", type=int, default=1)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([
    '--livestream', '2',
    '--kit_args', '--ext-folder /home/ted/isaac-sim/extsDeprecated --enable isaacsim.core.utils',
])
args_cli.enable_cameras = True
args_cli.device = 'cpu'

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import pandas as pd
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks
from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface
from sim_to_real_so101.utils.lerobot_recorder import LeRobotRecorder

import omni.kit.app


def main():
    # Load real dataset action data directly from parquet (no video decoding needed)
    real_dataset_root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{args_cli.real_repo_id}")
    parquet_path = f"{real_dataset_root}/data/chunk-000/file-000.parquet"
    df = pd.read_parquet(parquet_path)
    print(f"[INFO]: Loaded real dataset: {len(df)} total frames, {df['episode_index'].nunique()} episodes")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=None, use_fabric=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[INFO]: SUCCESS: environment created")
    print("Camera position:", env.unwrapped.scene["camera_overhead"].data.pos_w.torch)
    print("Camera orientation (quat):", env.unwrapped.scene["camera_overhead"].data.quat_w_ros.torch)
    from isaaclab.utils.math import quat_apply
    cam_quat_world = env.unwrapped.scene["camera_overhead"].data.quat_w_world.torch
    forward_local = torch.tensor([[0.0, 0.0, -1.0]], device=cam_quat_world.device)
    forward_world = quat_apply(cam_quat_world, forward_local)
    print("Camera forward direction (world):", forward_world)
    print("Robot base position:", env.unwrapped.scene["robot"].data.root_pos_w.torch)
    print("Cylinder position:", env.unwrapped.scene["cylinder"].data.root_pos_w.torch)
    print("Target position:", env.unwrapped.scene["target"].data.root_pos_w.torch)

    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            camera_cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {"height": camera_cfg.height, "width": camera_cfg.width}
            print(f"[INFO]: Found Camera: {obj.replace('camera_', '')}")

    # Interface used purely for its action-mapping math -- never connects to real hardware
    robot_iface = LeRobotSO101Interface(
        device=env.unwrapped.device,
        port="dummy",
        id="replay",
        cameras=cameras,
        fps=30,
        kind="leader",
    )

    recorder = LeRobotRecorder(
        task_name=args_cli.task_name,
        repo_id=args_cli.out_repo_id,
        dataset_root=args_cli.out_root,
        fps=30,
        device=env.unwrapped.device,
        cameras=cameras,
        save_mp4=False,
        depth=False,
        instance_id_seg=False,
    )
    try:
        recorder.init_dataset()
        print("[INFO]: SUCCESS: output dataset initialized")
    except ValueError:
        print("[INFO]: Output dataset already exists -- appending new episodes")

    for ep_idx in range(args_cli.start_episode, args_cli.end_episode):
        ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        n_frames = len(ep_df)
        print(f"[INFO]: Replaying episode {ep_idx} ({n_frames} frames)...")

        env.reset()

        for _, row in ep_df.iterrows():
            real_action = torch.tensor(row["action"], dtype=torch.float32, device=env.unwrapped.device)
            sim_action = robot_iface.get_mapped_actions_vectorized(real_action)

            obs, _, _, _, _ = env.step(sim_action.unsqueeze(0))

            visual_obs = obs.get("visual", None)
            if visual_obs is None:
                print("[WARNING]: No visual observation group -- skipping recording")
                continue

            joint_pos_obs = obs["policy"]["joint_pos_obs"][0]
            real_obs, visual_buffers, depth_buffers, instance_id_seg_buffers = (
                robot_iface.sim_to_real_dataset_processor(joint_pos_obs, visual_obs)
            )
            recorder.push_frame_to_buffer(real_action, real_obs, visual_buffers, depth_buffers, instance_id_seg_buffers)

        omni.kit.app.queue_event(LeRobotRecorder.STOP_RECORDING_EVENT, payload={})
        simulation_app.update()  # let the event actually get dispatched/queued
        import time
        start_wait = time.time()
        while not recorder.episode_queue.empty() and (time.time() - start_wait) < 120:
            time.sleep(0.5)
        time.sleep(5)  # small buffer after queue drains, for save_episode()/finalize() to fully return
        print(f"[INFO]: SUCCESS: episode {ep_idx} saved")

    print("[INFO]: All requested episodes replayed and saved.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
