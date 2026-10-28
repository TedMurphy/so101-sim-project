#!/usr/bin/env python3
"""
create_augmented_datasets.py
============================
Builds 3 augmented LeRobot v3.0 datasets from two source datasets:

  1. FullGrid_double_sim   : 15 SIM episodes  × 2 → 30, shuffled
  2. FullGrid_double_real  : 15 REAL episodes × 2 → 30, shuffled
  3. FullGrid_combo_real_sim : 15 SIM + 15 REAL → 30, shuffled

For each new dataset:
  - Shuffles/duplicates episodes with a fixed seed (printed to stdout)
  - Extracts per-episode video clips from the source chunk video via ffmpeg
  - Re-concatenates clips in the new order
  - Rewrites data parquets (episode_index, frame_index columns updated)
  - Rewrites meta/episodes.parquet and meta/info.json
  - Uploads the finished dataset to HuggingFace Hub

Usage:
    conda activate lerobot
    python3 create_augmented_datasets.py                  # build all three
    python3 create_augmented_datasets.py --target real    # double-real only
    python3 create_augmented_datasets.py --target sim     # double-sim only
    python3 create_augmented_datasets.py --target combo   # combined only

    --target    Which dataset to build: real | sim | combo | all (default: all)
    --dry-run   Build datasets locally, skip HF upload
    --seed N    Random seed for episode ordering (default: 42)
    --no-cache  Force re-download source datasets

Requirements: pandas, pyarrow, huggingface_hub, ffmpeg (system binary)
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download, HfApi

# ── Configuration ──────────────────────────────────────────────────────────────
HF_OWNER   = "Screener2"
SIM_REPO   = f"{HF_OWNER}/FullGrid_20261007_sim"
REAL_REPO  = f"{HF_OWNER}/FullGrid_20261007_095459"
CACHE_DIR  = Path.home() / ".cache" / "huggingface" / "lerobot"
VIDEO_CAM  = "observation.images.overhead"
VIDEO_TS_COL = f"videos/{VIDEO_CAM}/from_timestamp"

TARGETS = [
    {
        "repo_id":    f"{HF_OWNER}/FullGrid_double_sim",
        "episodes":   [("sim", i) for i in range(15)] * 2,   # 30 total, shuffled
        "task":       "Green cylinder to yellow bucket (sim)",
    },
    {
        "repo_id":    f"{HF_OWNER}/FullGrid_double_real",
        "episodes":   [("real", i) for i in range(15)] * 2,  # 30 total, shuffled
        "task":       "Green cylinder to yellow bucket",
    },
    {
        "repo_id":    f"{HF_OWNER}/FullGrid_combo_real_sim",
        "episodes":   [("sim",  i) for i in range(15)]        # 30 total, shuffled
                    + [("real", i) for i in range(15)],
        "task":       "Green cylinder to yellow bucket",
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def ensure_local(repo_id: str, force: bool = False) -> Path:
    """Download dataset from HF Hub to cache if not already present."""
    local = CACHE_DIR / repo_id.replace("/", "/")
    if local.exists() and not force:
        print(f"[CACHE] Using cached {repo_id} at {local}")
        return local
    print(f"[HF] Downloading {repo_id} …")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local),
        local_dir_use_symlinks=False,
    )
    return Path(path)


def load_source(local_path: Path):
    """
    Load meta/info.json and the episodes parquet for a source dataset.

    LeRobot v3 stores episode metadata at:
        meta/episodes/chunk-000/file-000.parquet
    (NOT meta/episodes.parquet — that path doesn't exist in these datasets.)
    """
    meta_dir = local_path / "meta"

    # ── info.json ─────────────────────────────────────────────────────────────
    info_path = meta_dir / "info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}

    # ── episodes parquet — search in priority order ───────────────────────────
    eps_candidates = [
        meta_dir / "episodes" / "chunk-000" / "file-000.parquet",  # actual v3 path
        meta_dir / "episodes.parquet",                              # flat fallback
    ] + sorted(meta_dir.glob("episodes/**/*.parquet"))              # any other layout

    eps_path = next((p for p in eps_candidates if p.exists()), None)
    if eps_path is None:
        # Print what we actually have so we can diagnose
        print(f"\n[DIAG] Could not find episodes parquet. Layout under {meta_dir}:")
        for p in sorted(meta_dir.rglob("*")):
            print(f"  {p.relative_to(meta_dir)}")
        raise FileNotFoundError(f"No episodes parquet found under {meta_dir}")

    print(f"  [META] Episodes: {eps_path.relative_to(local_path)}")
    eps_df = pd.read_parquet(str(eps_path))
    return info, eps_df


def load_all_episode_data(local_path: Path, episode_idx: int) -> pd.DataFrame:
    """
    Load frame rows for a single episode from the data parquets.

    LeRobot v3 typically stores ALL episodes in one file:
        data/chunk-000/file-000.parquet
    but may also store per-episode files.  We mirror the search order
    used in replay_episodes.py.
    """
    chunk_idx = episode_idx // 500
    candidates = [
        local_path / "data" / "chunk-000" / "file-000.parquet",
        local_path / "data" / f"chunk-{chunk_idx:03d}" / "file-000.parquet",
        local_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet",
        local_path / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(str(p))
            ep_df = df[df["episode_index"] == episode_idx].sort_values("frame_index")
            if not ep_df.empty:
                return ep_df.reset_index(drop=True)
    # Last resort: scan all parquets
    all_pq = sorted((local_path / "data").rglob("*.parquet"))
    for p in all_pq:
        df = pd.read_parquet(str(p))
        if episode_idx in df["episode_index"].values:
            return df[df["episode_index"] == episode_idx].sort_values("frame_index").reset_index(drop=True)
    raise RuntimeError(f"Episode {episode_idx} not found in any parquet under {local_path}/data/")


# data_parquet_path removed — use load_all_episode_data() instead, which
# handles the actual v3 layout (data/chunk-000/file-000.parquet with all episodes).


def video_path(local_path: Path, chunk: int = 0) -> Path:
    return local_path / "videos" / VIDEO_CAM / f"chunk-{chunk:03d}" / "file-000.mp4"


def get_video_duration(video_file: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", str(video_file)],
        capture_output=True, text=True, check=True
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def extract_clip(src_video: Path, start: float, n_frames_to_extract: int, out: Path):
    """
    Extract exactly n_frames_to_extract frames from src_video starting at `start` seconds.

    We use -frames:v (frame count) instead of -t (duration) to guarantee the
    extracted clip contains EXACTLY the requested number of frames.

    With -t n_frames/fps, Isaac Sim (exact 30 fps) hits a boundary-frame bug:
    the duration lands exactly on the next frame's PTS, causing ffmpeg to include
    one extra frame.  That shifts every downstream episode's PTS by one frame,
    so from_timestamp / to_timestamp no longer match the concatenated video —
    the player overshoots by a frame and briefly renders the next episode's first
    frame before looping.  -frames:v is immune to this because it counts, not times.

    We re-encode with libx264 (CRF 15, fast preset) rather than stream-copy.
    Stream-copy preserves the original PTS (e.g. a clip at t=60s keeps PTS≈60s),
    which causes the concat demuxer to produce a video with timestamp gaps —
    episodes 7+ appear to not play because the player jumps forward in time.
    Re-encoding resets PTS to start at 0 for every clip.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y",
         "-ss", f"{start:.6f}",              # fast seek before -i
         "-i", str(src_video),
         "-frames:v", str(n_frames_to_extract),  # exact frame count, NOT -t duration
         "-c:v", "libx264",
         "-preset", "fast",
         "-crf", "15",                        # visually near-lossless
         "-reset_timestamps", "1",            # PTS starts at 0 in output clip
         str(out)],
        check=True,
        capture_output=True,
    )


def concatenate_clips(clip_list: list[Path], out: Path):
    """Concatenate MP4 clips in order using ffmpeg concat demuxer."""
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write a concat list file
    list_file = out.parent / "_concat_list.txt"
    with open(list_file, "w") as f:
        for clip in clip_list:
            f.write(f"file '{clip.resolve()}'\n")
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "concat",
         "-safe", "0",
         "-i", str(list_file),
         "-c:v", "copy",
         str(out)],
        check=True,
        capture_output=True,
    )
    list_file.unlink()


# reindex_data_parquet removed — reindexing is now done inline in build_dataset()
# using load_all_episode_data() which handles the actual v3 single-file layout.


def normalize_array_dtypes(df: pd.DataFrame, target_dtype=None) -> pd.DataFrame:
    """
    Normalize object columns that contain numpy arrays (e.g. action, observation)
    so they all use the same float dtype.  SIM datasets use float32; REAL datasets
    use float64.  Mixing them causes pyarrow to raise ArrowInvalid on concat.

    If target_dtype is None, the first non-None array value in each column decides
    the dtype (float32 wins over float64 if present, for storage efficiency).
    """
    import numpy as np
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != object:
            continue
        # Check if this column holds numpy arrays
        sample = next((v for v in df[col] if v is not None), None)
        if not isinstance(sample, np.ndarray):
            continue
        # Determine target dtype: prefer float32 for compactness
        dtype = target_dtype or np.float32
        df[col] = df[col].apply(
            lambda v: v.astype(dtype) if isinstance(v, np.ndarray) else v
        )
    return df


# ── Main builder ───────────────────────────────────────────────────────────────

def build_dataset(
    target: dict,
    sources: dict,          # {"sim": (local_path, info, eps_df), "real": ...}
    out_root: Path,
    dry_run: bool = False,
    seed: int = 42,
    rng: random.Random = None,
):
    repo_id = target["repo_id"]
    ep_plan = list(target["episodes"])   # [(source_key, orig_ep_idx), ...]

    # Shuffle with provided RNG (each dataset gets its own shuffled order)
    rng.shuffle(ep_plan)

    print(f"\n{'='*70}")
    print(f"  Building: {repo_id}")
    print(f"  Episodes ({len(ep_plan)} total):")
    for new_idx, (src_key, orig_idx) in enumerate(ep_plan):
        print(f"    ep{new_idx:02d} ← {src_key} ep{orig_idx:02d}")
    print(f"{'='*70}")

    out_dir = out_root / repo_id.replace("/", "__")
    if out_dir.exists():
        print(f"[OUT] Removing existing {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Grab FPS from first source (both sources must have same FPS)
    first_info = sources[ep_plan[0][0]][1]
    fps = first_info["fps"]

    # ── Per-episode: extract video clips and copy/reindex data parquets ────────
    clip_dir = out_dir / "_clips"
    clip_dir.mkdir()

    data_tables       = []    # pyarrow Tables, one per new episode
    new_ep_rows       = []    # rows for meta/episodes.parquet
    cumulative_frames = 0     # integer frame counter — drives all timestamps
    global_frame_idx  = 0     # running global frame counter

    for new_idx, (src_key, orig_idx) in enumerate(ep_plan):
        src_path, src_info, src_eps = sources[src_key]

        # Lookup this episode's metadata from source episodes parquet
        src_row = src_eps[src_eps["episode_index"] == orig_idx].iloc[0]
        n_frames  = int(src_row["length"])
        vid_start = float(src_row[VIDEO_TS_COL]) if VIDEO_TS_COL in src_eps.columns else (orig_idx * n_frames / fps)
        duration  = n_frames / fps

        # Extract video clip — use n_frames (not duration) to avoid off-by-one
        # on exact 30fps sources where -t n_frames/fps can include a boundary frame.
        src_vid  = video_path(src_path)
        clip_out = clip_dir / f"ep_{new_idx:04d}.mp4"
        print(f"  [VIDEO] ep{new_idx:02d} ← {src_key} ep{orig_idx:02d}  "
              f"start={vid_start:.3f}s  frames={n_frames}  dur≈{duration:.3f}s  → {clip_out.name}")
        extract_clip(src_vid, vid_start, n_frames, clip_out)

        # Verify the extracted clip has exactly the expected frame count.
        # If this fires, the timestamps in the output parquet will be wrong.
        actual_clip_dur = get_video_duration(clip_out)
        actual_clip_frames = round(actual_clip_dur * fps)
        if actual_clip_frames != n_frames:
            print(f"  [WARN] ep{new_idx:02d}: clip has ~{actual_clip_frames} frames "
                  f"(expected {n_frames}) — PTS timestamps may be misaligned!")

        # Load and reindex frame data for this episode
        ep_df = load_all_episode_data(src_path, orig_idx)
        ep_df = normalize_array_dtypes(ep_df)   # unify float32/float64 across sim+real
        actual_n_frames = len(ep_df)
        if actual_n_frames != n_frames:
            print(f"  [WARN] ep{new_idx:02d}: parquet length={n_frames} but "
                  f"actual data rows={actual_n_frames} — using actual row count")
            n_frames = actual_n_frames
        ep_df["episode_index"] = new_idx
        ep_df["frame_index"]   = range(len(ep_df))
        if "index" in ep_df.columns:
            ep_df["index"] = range(global_frame_idx, global_frame_idx + len(ep_df))
        data_tables.append(ep_df)

        # Build episodes parquet row.
        # Derive timestamps from the INTEGER frame counter so they match the
        # concat demuxer's integer-timebase arithmetic exactly (no float drift).
        #
        # The HF viewer loops an episode by checking currentTime >= to_timestamp
        # on every 'timeupdate' event, which fires every ~250 ms.  This means the
        # player can overshoot to_timestamp by up to 250 ms before the loop fires.
        #
        # Setting to_ts = last frame PTS (n_frames-1)/fps means:
        #   • Episodes 0-28: timeupdate fires ~250 ms AFTER to_ts, by which time
        #     the next episode's frames are already playing → visible flash.
        #   • Episode 29: video reaches EOF before the next timeupdate fires →
        #     episode just stops instead of looping.
        #
        # Fix: pull to_ts back by TO_TS_VIEWER_BUFFER frames so the loop always
        # fires while the current episode is still on-screen.
        # The buffer must be > 250 ms × fps = 7.5 frames; we use 9 (= 300 ms).
        # This only affects the viewer; training data is completely unaffected.
        TO_TS_VIEWER_BUFFER = 9   # frames (~300 ms at 30 fps); covers 250 ms timeupdate
        TO_TS_COL = VIDEO_TS_COL.replace("/from_timestamp", "/to_timestamp")
        from_ts = cumulative_frames / fps
        to_ts   = (cumulative_frames + max(0, n_frames - 1 - TO_TS_VIEWER_BUFFER)) / fps
        ep_row = {
            "episode_index": new_idx,
            "length":        n_frames,
            VIDEO_TS_COL:    from_ts,
            TO_TS_COL:       to_ts,
        }
        # Copy any extra columns from source EXCEPT the timestamp ones we set above
        skip_cols = {"episode_index", "length", VIDEO_TS_COL, TO_TS_COL}
        for col in src_eps.columns:
            if col not in skip_cols:
                ep_row[col] = src_row[col]
        new_ep_rows.append(ep_row)

        cumulative_frames += n_frames
        global_frame_idx += n_frames

    # ── Concatenate video clips ────────────────────────────────────────────────
    clips_ordered = [clip_dir / f"ep_{i:04d}.mp4" for i in range(len(ep_plan))]
    out_vid = out_dir / "videos" / VIDEO_CAM / "chunk-000" / "file-000.mp4"
    print(f"\n  [VIDEO] Concatenating {len(clips_ordered)} clips → {out_vid}")
    concatenate_clips(clips_ordered, out_vid)

    # ── Write data parquets — single file-000.parquet matching v3 layout ────
    data_dir = out_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    all_frames = pd.concat(data_tables, ignore_index=True)
    out_parquet = data_dir / "file-000.parquet"
    all_frames.to_parquet(str(out_parquet), index=False, compression="snappy")
    print(f"  [DATA] Wrote data/chunk-000/file-000.parquet  ({len(all_frames)} rows)")

    # ── Write meta/episodes/chunk-000/file-000.parquet  (actual v3 layout) ──
    meta_dir      = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    eps_chunk_dir = meta_dir / "episodes" / "chunk-000"
    eps_chunk_dir.mkdir(parents=True)
    eps_out = pd.DataFrame(new_ep_rows)
    # Drop stats/ columns — they can have incompatible schemas when mixing SIM and
    # REAL sources (one stores lists, the other scalars), causing PyArrow errors.
    # These columns are viewer metadata only; training is unaffected.
    stats_cols = [c for c in eps_out.columns if c.startswith("stats/")]
    if stats_cols:
        eps_out = eps_out.drop(columns=stats_cols)
        print(f"  [META] Dropped {len(stats_cols)} stats/ columns (schema mismatch across sources)")
    # Ensure column order: episode_index first, then VIDEO_TS_COL, then length, rest
    col_order = ["episode_index", VIDEO_TS_COL, "length"] + \
        [c for c in eps_out.columns if c not in ("episode_index", VIDEO_TS_COL, "length")]
    eps_out = eps_out[[c for c in col_order if c in eps_out.columns]]
    eps_out.to_parquet(eps_chunk_dir / "file-000.parquet", index=False)
    print(f"  [META] meta/episodes/chunk-000/file-000.parquet  ({len(eps_out)} episodes)")

    # ── Write meta/info.json ──────────────────────────────────────────────────
    ref_info = sources[ep_plan[0][0]][1]
    new_info = dict(ref_info)
    new_info["total_episodes"]  = len(ep_plan)
    new_info["total_frames"]    = global_frame_idx
    new_info["total_chunks"]    = 1
    new_info["chunks_size"]     = 1000
    new_info["splits"]          = {"train": f"0:{len(ep_plan)}"}
    new_info["dataset_type"]    = "LeRobotDataset"
    if "tasks" in new_info:
        new_info["tasks"] = {str(k): target["task"] for k in new_info["tasks"]}
    (meta_dir / "info.json").write_text(json.dumps(new_info, indent=2))
    print(f"  [META] info.json  total_frames={global_frame_idx}")

    # Copy over any other meta files from the reference source (stats, tasks, etc.)
    ref_meta = sources[ep_plan[0][0]][0] / "meta"
    for src_file in ref_meta.rglob("*"):
        if src_file.is_file() and src_file.name not in ("info.json",):
            # Skip the episodes parquet — we wrote our own above
            if "episodes" in src_file.parts:
                continue
            rel = src_file.relative_to(ref_meta)
            dst = meta_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_file, dst)

    # ── Clean up clips ────────────────────────────────────────────────────────
    shutil.rmtree(clip_dir)

    print(f"\n  ✓ Dataset built at {out_dir}")

    # ── Upload to HuggingFace ─────────────────────────────────────────────────
    if not dry_run:
        print(f"  [HF] Uploading {repo_id} …")
        api = HfApi()
        # Delete and recreate the repo so the upload is never skipped due to
        # "no files have been modified" — the HF SDK compares file hashes and
        # silently skips commits when bytes are identical to the previous version.
        # A fresh repo guarantees the new parquet/video always lands.
        try:
            api.delete_repo(repo_id=repo_id, repo_type="dataset", missing_ok=True)
            print(f"  [HF] Deleted existing repo, recreating fresh …")
        except Exception:
            pass
        api.create_repo(repo_id=repo_id, repo_type="dataset")
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"  [HF] ✓ Uploaded → https://huggingface.co/datasets/{repo_id}")
    else:
        print(f"  [DRY RUN] Skipping upload for {repo_id}")

    return out_dir


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build augmented LeRobot v3.0 datasets.")
    parser.add_argument("--dry-run",  action="store_true", help="Build locally, skip HF upload")
    parser.add_argument("--seed",     type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-cache", action="store_true", help="Force re-download source datasets")
    parser.add_argument("--out-dir",  type=str, default=str(Path.home() / "augmented_datasets"),
                        help="Directory for built datasets (default: ~/augmented_datasets)")
    parser.add_argument("--target",   type=str, default="all",
                        choices=["all", "real", "sim", "combo"],
                        help="Which dataset to build: real | sim | combo | all (default: all)")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  create_augmented_datasets.py  (seed={args.seed})")
    print(f"{'='*70}")

    # ── Filter targets by --target flag ──────────────────────────────────────
    target_map = {"real": 1, "sim": 0, "combo": 2}   # index into TARGETS list
    if args.target == "all":
        selected = list(enumerate(TARGETS))
    else:
        idx = target_map[args.target]
        selected = [(idx, TARGETS[idx])]

    # ── Determine which sources are actually needed ───────────────────────────
    needed_sources = set()
    for _, target in selected:
        for src_key, _ in target["episodes"]:
            needed_sources.add(src_key)

    # ── Load only the needed source datasets ──────────────────────────────────
    print("\n[SOURCES] Ensuring source datasets are available …")
    sources = {}
    if "sim" in needed_sources:
        sim_path = ensure_local(SIM_REPO, force=args.no_cache)
        sim_info, sim_eps = load_source(sim_path)
        sources["sim"] = (sim_path, sim_info, sim_eps)
        print(f"  SIM : {len(sim_eps)} episodes, {sim_info.get('total_frames','?')} frames, "
              f"{sim_info.get('fps','?')} fps")
    if "real" in needed_sources:
        real_path = ensure_local(REAL_REPO, force=args.no_cache)
        real_info, real_eps = load_source(real_path)
        sources["real"] = (real_path, real_info, real_eps)
        print(f"  REAL: {len(real_eps)} episodes, {real_info.get('total_frames','?')} frames, "
              f"{real_info.get('fps','?')} fps")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Each target gets its own RNG so their shuffles are independent but
    # reproducible from the same seed.
    results = []
    for i, target in selected:
        rng = random.Random(args.seed + i)   # seed+0, seed+1, seed+2
        out_dir = build_dataset(target, sources, out_root,
                                dry_run=args.dry_run, seed=args.seed, rng=rng)
        results.append((target["repo_id"], out_dir))

    print(f"\n{'='*70}")
    print("  All datasets complete!")
    print(f"{'='*70}")
    for repo_id, out_dir in results:
        print(f"  {repo_id}")
        print(f"    local: {out_dir}")
        if not args.dry_run:
            print(f"    hub:   https://huggingface.co/datasets/{repo_id}")
    print()


if __name__ == "__main__":
    main()