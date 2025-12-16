#!/usr/bin/env python3
"""Batch-generate memory QA datasets for every episode directory under a video root."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

DEFAULT_GENERATOR = Path(__file__).resolve().parent / "generate_memory_qas.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("outputs") / "video",
        help="Root directory that holds per-episode subdirectories (default: outputs/video).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "memory",
        help="Root directory where per-episode QA JSON files will be written (default: outputs/memory).",
    )
    parser.add_argument(
        "--generator-script",
        type=Path,
        default=DEFAULT_GENERATOR,
        help="Path to the generate_memory_qas.py script (default: alongside this batch script).",
    )
    parser.add_argument(
        "--agent-prefix",
        default="agent_1",
        help="Prefix for agent subdirectories that hold frame dumps (default: agent_1).",
    )
    parser.add_argument("--min-objects", type=int, default=3, help="Minimum distinct objects for ordering questions.")
    parser.add_argument("--max-objects", type=int, default=4, help="Maximum distinct objects for ordering questions.")
    parser.add_argument(
        "--questions-per-size",
        type=int,
        default=8,
        help="Ordering questions to sample for each object count.",
    )
    parser.add_argument("--cooccur-questions", type=int, default=8, help="Number of co-occurrence questions.")
    parser.add_argument(
        "--cooccur-min-overlap",
        type=int,
        default=2,
        help="Minimum shared-object count required for co-occurrence questions.",
    )
    parser.add_argument(
        "--cooccur-max-options",
        type=int,
        default=4,
        help="Maximum answer options for co-occurrence questions.",
    )
    parser.add_argument(
        "--earliest-questions",
        type=int,
        default=8,
        help="Number of earliest-object questions to sample per object count.",
    )
    parser.add_argument(
        "--first-frame-questions",
        type=int,
        default=12,
        help="Number of first-frame questions to generate per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed forwarded to the generator script.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip episodes whose output JSON already exists.",
    )
    return parser.parse_args()


def discover_episodes(video_root: Path) -> List[Path]:
    episodes: List[Path] = []
    if not video_root.exists():
        return episodes
    for entry in sorted(video_root.iterdir()):
        if entry.is_dir():
            episodes.append(entry)
    return episodes


def main() -> None:
    args = parse_args()
    generator = args.generator_script.resolve()
    if not generator.exists():
        raise FileNotFoundError(f"Generator script not found: {generator}")

    episodes = discover_episodes(args.video_root.resolve())
    if not episodes:
        print(f"No episode directories found under {args.video_root}")
        return

    failures: List[tuple[Path, int]] = []
    for episode_dir in episodes:
        output_dir = (args.output_root / episode_dir.name).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"{episode_dir.name}_memory_qas.json"
        output_file = output_dir / output_name
        if args.skip_existing and output_file.exists():
            print(f"Skipping {episode_dir} (existing {output_file})")
            continue

        cmd = [
            sys.executable,
            str(generator),
            str(episode_dir),
            "--agent-prefix",
            args.agent_prefix,
            "--min-objects",
            str(args.min_objects),
            "--max-objects",
            str(args.max_objects),
            "--questions-per-size",
            str(args.questions_per_size),
            "--cooccur-questions",
            str(args.cooccur_questions),
            "--cooccur-min-overlap",
            str(args.cooccur_min_overlap),
            "--cooccur-max-options",
            str(args.cooccur_max_options),
            "--earliest-questions",
            str(args.earliest_questions),
            "--first-frame-questions",
            str(args.first_frame_questions),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(output_dir),
            "--output-name",
            output_name,
        ]

        print(f"Generating QA for {episode_dir} -> {output_file}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            failures.append((episode_dir, result.returncode))
            print(f"[ERROR] Failed on {episode_dir} (exit {result.returncode})")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
        else:
            if result.stdout:
                print(result.stdout.strip())

    if failures:
        print("\nCompleted with errors:")
        for episode_dir, code in failures:
            print(f" - {episode_dir} (exit code {code})")
        raise SystemExit(1)

    print("\nAll episodes processed successfully.")


if __name__ == "__main__":
    main()
