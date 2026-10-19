#!/usr/bin/env python3
"""
mirror_episodes.py
------------------
Create left-right mirrored copies of existing episodes.

For each source episode:
  - Flips the overhead camera image horizontally
  - Negates Rotation (shoulder pan) and Wrist_Roll joint angles/actions
  - Saves as a new episode appended to the dataset

This converts "pick right, place left" demonstrations into
"pick left, place right" demonstrations without new teleoperation.

Usage:
    python mirror_episodes.py \
        --src_dir  ./data/real \
        --dst_dir  ./data/real_mirrored \
        --episodes all          # or: --episodes 0 1 2 3 4

Verify a few mirrored episodes visually before adding to training.
"""

import argparse
import os
import shutil
import h5py
import numpy as np


# ── Configure to match your dataset ───────────────────────────────────────────

# Key in the HDF5 that holds joint positions (observation)
OBS_STATE_KEY  = "observation.state"       # shape (T, n_joints)

# Key in the HDF5 that holds joint actions
ACTION_KEY     = "action"                  # shape (T, n_joints)

# Key in the HDF5 that holds the overhead camera image
IMAGE_KEY      = "observation.images.top"  # shape (T, H, W, 3)  uint8

# Joint order — must match your dataset's column order exactly.
# Only Rotation and Wrist_Roll are negated; everything else is unchanged.
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# Joints to negate for a left-right mirror
NEGATE_JOINTS = {"Rotation", "Wrist_Roll"}

# ─────────────────────────────────────────────────────────────────────────────


def negate_mask(joint_names, negate_set):
    """Return a sign vector: -1 for joints to negate, +1 for all others."""
    return np.array(
        [-1.0 if name in negate_set else 1.0 for name in joint_names],
        dtype=np.float32
    )


def mirror_episode(src_path: str, dst_path: str, sign: np.ndarray):
    """Read one episode HDF5, mirror it, write to dst_path."""
    with h5py.File(src_path, "r") as src:
        with h5py.File(dst_path, "w") as dst:

            def _copy_mirrored(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data = obj[:]

                    if name == OBS_STATE_KEY or name == ACTION_KEY:
                        # Negate the laterally-symmetric joints
                        data = data.astype(np.float32) * sign

                    elif name == IMAGE_KEY:
                        # Horizontal flip: reverse the width axis (axis=2 for THWC)
                        data = data[:, :, ::-1, :].copy()

                    dst.create_dataset(name, data=data,
                                       compression=obj.compression,
                                       compression_opts=obj.compression_opts
                                       if obj.compression else None)

                    # Copy dataset attributes
                    for k, v in obj.attrs.items():
                        dst[name].attrs[k] = v

                elif isinstance(obj, h5py.Group) and name != "":
                    dst.require_group(name)
                    for k, v in obj.attrs.items():
                        dst[name].attrs[k] = v

            # Copy root attributes
            for k, v in src.attrs.items():
                dst.attrs[k] = v

            src.visititems(_copy_mirrored)

    print(f"  mirrored → {os.path.basename(dst_path)}")


def find_episodes(src_dir):
    """Return sorted list of (index, path) for all episode HDF5 files."""
    episodes = []
    for fname in sorted(os.listdir(src_dir)):
        if fname.endswith(".hdf5") or fname.endswith(".h5"):
            # Parse index from filename: episode_000.hdf5, ep0.hdf5, etc.
            stem = os.path.splitext(fname)[0]
            digits = "".join(filter(str.isdigit, stem))
            idx = int(digits) if digits else len(episodes)
            episodes.append((idx, os.path.join(src_dir, fname)))
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir",  required=True, help="Source dataset directory")
    ap.add_argument("--dst_dir",  required=True, help="Output directory for mirrored episodes")
    ap.add_argument("--episodes", nargs="+", default=["all"],
                    help="Episode indices to mirror, or 'all' (default)")
    ap.add_argument("--start_idx", type=int, default=None,
                    help="Starting episode index for output filenames "
                         "(default: continue from highest existing index in dst_dir)")
    args = ap.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)

    sign = negate_mask(JOINT_NAMES, NEGATE_JOINTS)
    print(f"Joint sign vector: { {n: int(s) for n, s in zip(JOINT_NAMES, sign)} }")

    all_eps = find_episodes(args.src_dir)
    if not all_eps:
        print(f"No HDF5 files found in {args.src_dir}")
        return

    if args.episodes == ["all"]:
        selected = all_eps
    else:
        wanted = set(int(x) for x in args.episodes)
        selected = [(i, p) for i, p in all_eps if i in wanted]

    # Determine starting output index
    if args.start_idx is not None:
        out_idx = args.start_idx
    else:
        existing = find_episodes(args.dst_dir)
        out_idx = (max(i for i, _ in existing) + 1) if existing else 0

    print(f"\nMirroring {len(selected)} episode(s) → {args.dst_dir}")
    print(f"Output indices start at {out_idx}\n")

    for src_idx, src_path in selected:
        fname = os.path.basename(src_path)
        dst_fname = f"episode_{out_idx:06d}.hdf5"
        dst_path  = os.path.join(args.dst_dir, dst_fname)
        print(f"  ep {src_idx:3d}  {fname}")
        mirror_episode(src_path, dst_path, sign)
        out_idx += 1

    print(f"\nDone. {len(selected)} mirrored episode(s) written to {args.dst_dir}")
    print("\nNext steps:")
    print("  1. Spot-check a few episodes: play back the image and joint angles visually.")
    print("  2. If the mirrored arm pose looks correct, merge dst_dir into your training dataset.")
    print("  3. If Wrist_Roll doesn't need negating for your arm, remove it from NEGATE_JOINTS.")


if __name__ == "__main__":
    main()