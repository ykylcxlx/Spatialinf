"""Capture visible object IDs from a ProcTHOR frame.

This script loads a ProcTHOR house description, starts an AI2-THOR controller,
optionally executes a sequence of actions, and saves the current RGB frame
along with the list of visible object IDs.
"""
# python procthordata/getItemLists.py \
#   procthordata/houses/train_000.json \
#   --output-dir outputs/demo_capture \
#   --width 800 --height 600 \
#   --actions RotateRight MoveAhead RotateRight \
#   --x-display :1 \
#   --connect-timeout 300 \
#   --visibility-distance 10

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from PIL import Image

from ai2thor.controller import Controller


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Load a ProcTHOR scene, capture the current frame, and store the visible"
			" object IDs."
		)
	)
	parser.add_argument(
		"house_json",
		type=Path,
		help="Path to a ProcTHOR house JSON file (e.g., a sample from procthor-10k).",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("captures"),
		help="Directory where the frame image and metadata will be saved.",
	)
	parser.add_argument(
		"--width",
		type=int,
		default=600,
		help="Viewport width passed to the AI2-THOR controller.",
	)
	parser.add_argument(
		"--height",
		type=int,
		default=600,
		help="Viewport height passed to the AI2-THOR controller.",
	)
	parser.add_argument(
		"--visibility-distance",
		type=float,
		default=5.0,
		help="Maximum distance (in meters) for an object to count as visible.",
	)
	parser.add_argument(
		"--actions",
		nargs="*",
		default=(),
		help="Optional sequence of AI2-THOR actions (e.g., RotateRight MoveAhead).",
	)
	parser.add_argument(
		"--connect-timeout",
		type=float,
		default=120.0,
		help="Seconds to wait for the AI2-THOR backend process to start (default 120).",
	)
	parser.add_argument(
		"--x-display",
		default=None,
		help=(
			"Optional X display string (e.g., ':1') for headless setups. If not"
			" provided, the controller uses the default active display."
		),
	)
	return parser.parse_args()


def load_house(house_path: Path) -> dict:
	if not house_path.exists():
		raise FileNotFoundError(f"Cannot find house description at {house_path}")
	with house_path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def save_rgb_frame(frame, destination: Path) -> None:
	destination.parent.mkdir(parents=True, exist_ok=True)
	Image.fromarray(frame).save(destination)


def extract_visible_object_ids(event) -> List[str]:
	objects: Iterable[dict] = event.metadata.get("objects", [])
	return [obj["objectId"] for obj in objects if obj.get("visible", False)]


def main() -> None:
	args = parse_args()
	scene = load_house(args.house_json)

	controller_kwargs = {
		"scene": scene,
		"width": args.width,
		"height": args.height,
	}
	if args.x_display is not None:
		controller_kwargs["x_display"] = args.x_display

	controller = Controller(
		connect_timeout=args.connect_timeout,
		visibilityDistance=args.visibility_distance,
		**controller_kwargs,
	)
	try:
		# Run the requested sequence of actions so the user can reposition the agent.
		for action in args.actions:
			controller.step(action=action)

		event = controller.last_event
		if event is None:
			raise RuntimeError("Controller did not return an event; cannot capture frame.")

		frame_path = args.output_dir / "frame.png"
		save_rgb_frame(event.frame, frame_path)

		visible_ids = extract_visible_object_ids(event)
		metadata_path = args.output_dir / "visible_objects.json"
		metadata_path.parent.mkdir(parents=True, exist_ok=True)
		metadata = {
			"house_json": str(args.house_json.resolve()),
			"actions": list(args.actions),
			"visible_object_ids": visible_ids,
		}
		metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

		print(f"Saved RGB frame to {frame_path}")
		print(f"Recorded {len(visible_ids)} visible object IDs in {metadata_path}")
	finally:
		controller.stop()


if __name__ == "__main__":
	main()
