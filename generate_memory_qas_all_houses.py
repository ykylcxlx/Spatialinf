#!/usr/bin/env python3
"""Generate ProcTHOR memory QA datasets for every house in the dataset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

import prior


# python generate_memory_qas_all_houses.py   --splits train val   --episodes-per-house 3   --agent-prefix agent_1   --output-root /data5/zhuangyunhao/outputs/memory/per_house   --video-output-root /data5/zhuangyunhao/outputs/memory/per_house/videos   --plan-output-root /data5/zhuangyunhao/outputs/video/per_house   --plan-min-nav-distance 5.0   --generator-args --cooccur-frame-gap 6 --questions-per-size 10

DEFAULT_GENERATOR = Path(__file__).resolve().parent / "generate_memory_qas.py"
DEFAULT_OUTPUT_ROOT = Path("outputs") / "memory" / "per_house"
DEFAULT_VIDEO_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "videos"
DEFAULT_PLAN_OUTPUT_ROOT = Path("outputs") / "video" / "per_house"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator-script",
        type=Path,
        default=DEFAULT_GENERATOR,
        help="Path to the generate_memory_qas.py script (default: alongside this batch script).",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train"],
        help="Which ProcTHOR splits to process (default: train).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting house index (inclusive) for each selected split (default: 0).",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        help="Ending house index (exclusive). If omitted, processes the entire split.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        help="Maximum number of houses to process per split (useful for smoke testing).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for per-house QA JSON outputs (default: outputs/memory/per_house).",
    )
    parser.add_argument(
        "--video-output-root",
        type=Path,
        default=DEFAULT_VIDEO_OUTPUT_ROOT,
        help="Root directory for per-house rendered videos (default: outputs/memory/per_house/videos).",
    )
    parser.add_argument(
        "--plan-output-root",
        type=Path,
        default=DEFAULT_PLAN_OUTPUT_ROOT,
        help="Root directory where raw ProcTHOR frame dumps should be stored (default: outputs/video/per_house).",
    )
    parser.add_argument(
        "--agent-prefix",
        default="agent_1",
        help="Agent prefix forwarded to the generator script (default: agent_1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Base random seed forwarded to the generator script (default: 2025).",
    )
    parser.add_argument(
        "--plan-min-nav-distance",
        type=float,
        default=5.0,
        help="Minimum navigation distance (meters) forwarded to the generator script (default: 5.0).",
    )
    parser.add_argument(
        "--plan-max-nav-iterations",
        type=int,
        default=400,
        help="Maximum navigation iterations forwarded to the generator script (default: 400).",
    )
    parser.add_argument(
        "--episodes-per-house",
        type=int,
        default=1,
        help="Number of unique episodes to record per house (default: 1).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip houses whose JSON output already exists.",
    )
    parser.add_argument(
        "--generator-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded verbatim to generate_memory_qas.py.",
    )
    return parser.parse_args()


def load_scene_index() -> Dict[str, List[dict]]:
    dataset = prior.load_dataset("procthor-10k")

    splits: List[str] = []

    if hasattr(dataset, "keys"):
        try:
            splits = list(dataset.keys())  # type: ignore[arg-type]
        except Exception:
            splits = []

    if not splits and hasattr(dataset, "datasets"):
        datasets_attr = getattr(dataset, "datasets")
        if isinstance(datasets_attr, dict):
            splits = list(datasets_attr.keys())

    if not splits:
        fallback_candidates = [
            "train",
            "val",
            "validation",
            "test",
            "train_unseen",
            "val_unseen",
            "validation_unseen",
        ]
        seen = set()
        for name in fallback_candidates:
            if name in seen:
                continue
            try:
                _ = dataset[name]
            except Exception:
                continue
            splits.append(name)
            seen.add(name)

    scene_index: Dict[str, List[dict]] = {}
    for split in splits:
        try:
            scenes = dataset[split]
        except Exception:
            continue
        scene_index[split] = scenes

    if not scene_index:
        raise RuntimeError("Unable to enumerate splits from the ProcTHOR dataset.")

    return scene_index


def iter_scene_indices(
    scenes: List[dict],
    start_index: int,
    end_index: int | None,
    max_scenes: int | None,
) -> Iterable[int]:
    total = len(scenes)
    upper = end_index if end_index is not None else total
    upper = min(upper, total)
    count = 0
    for idx in range(start_index, upper):
        yield idx
        count += 1
        if max_scenes is not None and count >= max_scenes:
            break


def build_command(
    generator: Path,
    output_dir: Path,
    video_output_dir: Path,
    plan_output_dir: Path,
    seed: int,
    split: str,
    house_index: int,
    plan_min_nav_distance: float,
    plan_max_nav_iterations: int,
    extra_args: List[str] | None,
    agent_prefix: str,
    episode_id: str,
    conversation_prefix: str,
    output_name: str,
    video_id: str,
) -> tuple[List[str], Dict[str, str]]:
    env = os.environ.copy()
    env["PROC_THOR_SPLIT"] = split
    env["PROC_THOR_HOUSE_INDEX"] = str(house_index)

    cmd = [
        sys.executable,
        str(generator),
        "--auto-run",
        "--agent-prefix",
        agent_prefix,
        "--output-dir",
        str(output_dir),
        "--output-name",
        output_name,
        "--episode-id",
        episode_id,
        "--conversation-prefix",
        conversation_prefix,
        "--video-id",
        video_id,
        "--video-output-dir",
        str(video_output_dir),
        "--plan-output-base",
        str(plan_output_dir),
        "--plan-seed",
        str(seed),
        "--plan-min-nav-distance",
        str(plan_min_nav_distance),
        "--plan-max-nav-iterations",
        str(plan_max_nav_iterations),
    ]

    if extra_args:
        cmd.extend(extra_args)

    return cmd, env


def main() -> None:
    args = parse_args()
    generator = args.generator_script.resolve()
    if not generator.exists():
        raise FileNotFoundError(f"Generator script not found: {generator}")

    dataset = load_scene_index()
    requested_splits = args.splits or list(dataset.keys())

    successes: List[tuple[str, str]] = []
    failures: List[tuple[str, str, int]] = []

    for split in requested_splits:
        scenes = dataset.get(split)
        if not scenes:
            print(f"[WARN] Split '{split}' not found in dataset; skipping.")
            continue

        split_output_root = args.output_root.resolve() / split
        split_video_root = args.video_output_root.resolve() / split
        split_plan_root = args.plan_output_root.resolve() / split
        split_output_root.mkdir(parents=True, exist_ok=True)
        split_video_root.mkdir(parents=True, exist_ok=True)
        split_plan_root.mkdir(parents=True, exist_ok=True)

        for view in ("agent", "top_view"):
            (split_video_root / view).mkdir(parents=True, exist_ok=True)

        aggregated_path = split_output_root / "memory_conversations.json"
        aggregated_data: List[Dict[str, Any]] = []
        existing_episode_ids: set[str] = set()
        if aggregated_path.exists():
            try:
                loaded = json.loads(aggregated_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    aggregated_data = loaded
                    for entry in aggregated_data:
                        metadata = entry.get("metadata") if isinstance(entry, dict) else None
                        if isinstance(metadata, dict):
                            episode_id = metadata.get("episode_id")
                            if isinstance(episode_id, str):
                                existing_episode_ids.add(episode_id)
                else:
                    print(f"[WARN] Aggregated file at {aggregated_path} is not a list; starting fresh.")
            except Exception as exc:
                print(f"[WARN] Failed to load existing aggregated data ({exc}); starting fresh.")

        for house_index in iter_scene_indices(scenes, args.start_index, args.end_index, args.max_scenes):
            scene = scenes[house_index]
            scene_id = str(scene.get("id", f"{split}_{house_index}"))
            for episode_idx in range(args.episodes_per_house):
                episode_id = f"{split}_{scene_id}_ep{episode_idx:02d}"
                if args.skip_existing and episode_id in existing_episode_ids:
                    print(f"[SKIP] {split}:{scene_id} ep={episode_idx} (aggregated entry exists)")
                    continue

                plan_output_dir = split_plan_root / scene_id / f"episode_{episode_idx:02d}"
                plan_output_dir.mkdir(parents=True, exist_ok=True)

                seed = args.seed + house_index * args.episodes_per_house + episode_idx
                output_name = "memory_conversations.json"

                with tempfile.TemporaryDirectory(prefix="memory_qas_") as tmp_output_dir_str:
                    tmp_output_dir = Path(tmp_output_dir_str)
                    cmd, env = build_command(
                        generator=generator,
                        output_dir=tmp_output_dir,
                        video_output_dir=split_video_root,
                        plan_output_dir=plan_output_dir,
                        seed=seed,
                        split=split,
                        house_index=house_index,
                        plan_min_nav_distance=args.plan_min_nav_distance,
                        plan_max_nav_iterations=args.plan_max_nav_iterations,
                        extra_args=args.generator_args,
                        agent_prefix=args.agent_prefix,
                        episode_id=episode_id,
                        conversation_prefix=episode_id,
                        output_name=output_name,
                        video_id=episode_id,
                    )

                    print(
                        f"[INFO] Processing split={split} scene={scene_id} (index {house_index}) episode={episode_idx}"
                    )
                    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if result.returncode != 0:
                        failures.append((split, f"{scene_id}/episode_{episode_idx:02d}", result.returncode))
                        print(f"[ERROR] Failed for {split}:{scene_id} episode={episode_idx} exit={result.returncode}")
                        if result.stdout:
                            print(result.stdout)
                        if result.stderr:
                            print(result.stderr)
                        continue

                    output_file = tmp_output_dir / output_name
                    try:
                        run_entries = json.loads(output_file.read_text(encoding="utf-8"))
                    except FileNotFoundError:
                        failures.append((split, f"{scene_id}/episode_{episode_idx:02d}", -1))
                        print(f"[ERROR] Expected output file not found for {episode_id}.")
                        continue
                    except json.JSONDecodeError as exc:
                        failures.append((split, f"{scene_id}/episode_{episode_idx:02d}", -1))
                        print(f"[ERROR] JSON decode failure for {episode_id}: {exc}")
                        continue

                    if not isinstance(run_entries, list):
                        failures.append((split, f"{scene_id}/episode_{episode_idx:02d}", -1))
                        print(f"[ERROR] Unexpected output structure for {episode_id}; expected list of entries.")
                        continue

                    normalized_entries: List[Dict[str, Any]] = []
                    for idx, entry in enumerate(run_entries, start=1):
                        if not isinstance(entry, dict):
                            continue
                        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
                        if metadata.get("episode_id") != episode_id:
                            metadata = dict(metadata)
                            metadata["episode_id"] = episode_id
                        entry["metadata"] = metadata
                        if not isinstance(entry.get("id"), str):
                            entry["id"] = f"{episode_id}_q{idx:04d}"
                        normalized_entries.append(entry)

                    if not normalized_entries:
                        failures.append((split, f"{scene_id}/episode_{episode_idx:02d}", -1))
                        print(f"[ERROR] No usable conversation entries produced for {episode_id}.")
                        continue

                    run_entries = normalized_entries

                    if episode_id in existing_episode_ids:
                        filtered: List[Dict[str, Any]] = []
                        for entry in aggregated_data:
                            metadata = entry.get("metadata") if isinstance(entry, dict) else None
                            if isinstance(metadata, dict) and metadata.get("episode_id") == episode_id:
                                continue
                            filtered.append(entry)
                        aggregated_data = filtered
                        existing_episode_ids.discard(episode_id)

                    aggregated_data.extend(run_entries)
                    existing_episode_ids.add(episode_id)
                    aggregated_path.write_text(
                        json.dumps(aggregated_data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                    successes.append((split, f"{scene_id}/episode_{episode_idx:02d}"))
                    if result.stdout:
                        print(result.stdout.strip())

    if failures:
        print("\nCompleted with errors:")
        for split, scene_id, code in failures:
            print(f" - {split}:{scene_id} (exit {code})")
        raise SystemExit(1)

    print("\nAll requested houses processed successfully.")
    print(f"Total runs: {len(successes)}")


if __name__ == "__main__":
    main()
