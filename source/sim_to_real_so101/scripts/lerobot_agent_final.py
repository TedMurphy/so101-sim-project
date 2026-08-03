import argparse
import os
import datetime

def ts():
    return datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab SO-101 Teleop agent (test v8).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--port", type=str, default=os.getenv("TELEOP_PORT", "/dev/ttyACM0"), help="Port of the robot.")
parser.add_argument("--robot_id", type=str, default=os.getenv("TELEOP_ID", "leader_arm_1"), help="ID of the robot.")
parser.add_argument("--repo_id", type=str, default=None, help="Repository ID to store the dataset.")
parser.add_argument("--repo_root", type=str, default=None, help="Repository root to store the dataset.")
parser.add_argument("--save_mp4", action="store_true", default=False, help="Save depth and RGB as mp4 videos.")
parser.add_argument("--depth", action="store_true", default=False, help="Save depth as mp4 video.")
parser.add_argument("--instance_id_seg", action="store_true", default=False, help="Save instance id segmentation as mp4 video.")
parser.add_argument("--task_name", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=101, help="Environment seed")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([
    '--task', 'Lerobot-So101-Teleop-Cylinder-To-Target',
    '--robot_id', 'SO101_leader_arm',
    '--repo_id', 'test_disposable',
    '--repo_root', '/home/ted/so101_test_dataset_DELETE_ME',
    '--task_name', 'test_task',
    '--livestream', '2',
    '--kit_args', '--ext-folder /home/ted/isaac-sim/extsDeprecated --enable isaacsim.core.utils',
])
args_cli.enable_cameras = True
args_cli.device = 'cpu'

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks
from sim_to_real_so101.utils.keyboard import KeyboardControl
from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface
from sim_to_real_so101.utils.lerobot_recorder import LeRobotRecorder


def main():
    keyboard_control = KeyboardControl()

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f'[{ts()}] SUCCESS: environment created')
    env.reset()
    print(f'[{ts()}] SUCCESS: environment reset')

    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            camera_cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {"height": camera_cfg.height, "width": camera_cfg.width}
            print(f"[INFO]: Found Camera: {obj.replace('camera_', '')}")

    robot_iface = LeRobotSO101Interface(
        device=env.unwrapped.device,
        port=args_cli.port,
        id=args_cli.robot_id,
        cameras=cameras,
        fps=30,
        kind='leader',
    )
    robot_iface.init_device()
    robot_iface.connect()
    print(f'[{ts()}] SUCCESS: robot connected')

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    recording_mode = all([args_cli.repo_id, args_cli.repo_root, args_cli.task_name])
    print(f'[{ts()}] SUCCESS: recording_mode = {recording_mode}')

    if recording_mode:
        recorder = LeRobotRecorder(
            task_name=args_cli.task_name,
            repo_id=args_cli.repo_id,
            dataset_root=args_cli.repo_root,
            fps=30,
            device=env.unwrapped.device,
            cameras=cameras,
            save_mp4=args_cli.save_mp4,
            depth=args_cli.depth,
            instance_id_seg=args_cli.instance_id_seg,
        )
        try:
            recorder.init_dataset()
            print(f'[{ts()}] SUCCESS: dataset initialized')
        except ValueError:
            print("[ERROR]: Failed to initialize dataset. folder already exists")
            env.close()
            simulation_app.close()

    keyboard_control.recording = True

    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            real_action = robot_iface.robot.get_action()
            real_action, mapped_action = robot_iface.real_to_sim_obs_processor(real_action)
            actions[:] = mapped_action

            obs, _, _, _, _ = env.step(actions)
            count += 1
            pass

            if keyboard_control.reset_world:
                keyboard_control.reset_world = False
                env.reset()
                continue

            if recording_mode and keyboard_control.recording:
                visual_obs = obs.get("visual", None)
                if visual_obs is None:
                    print("[WARNING]: No 'visual' observation group - recording requires a task with cameras")
                    keyboard_control.recording = False
                    continue
                joint_pos_obs = obs["policy"]["joint_pos_obs"][0]
                visual_obs = obs["visual"]
                real_obs, visual_buffers, depth_buffers, instance_id_seg_buffers = (
                    robot_iface.sim_to_real_dataset_processor(joint_pos_obs, visual_obs)
                )
                recorder.push_frame_to_buffer(real_action, real_obs, visual_buffers, depth_buffers, instance_id_seg_buffers)
                

            # no cap -- run until Ctrl+C

    print('SUCCESS: loop completed, exiting')

    if recording_mode:
        import omni.kit.app
        omni.kit.app.queue_event(LeRobotRecorder.STOP_RECORDING_EVENT, payload={})
        for _ in range(30):
            simulation_app.update()
        print('SUCCESS: stop-recording event dispatched and processed')

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
