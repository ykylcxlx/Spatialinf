"""Generate metric measurement QA pairs from recorded ProcTHOR agent videos.

This script generates questions about:
1. Camera-to-object distance
2. Object-to-object distance
3. Object size (width, height, depth, volume)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Unit configuration - centralized for easy modification
DISTANCE_UNIT = "cm"  # centimeters
SIZE_UNIT = "cm"      # centimeters
DECIMAL_PLACES = 2

# Question templates
CAMERA_TO_OBJECT_DISTANCE_TEMPLATES: Sequence[str] = (
    "Calculate the distance from the camera to the {object}.",
    "How far is the {object} from the camera?",
    "What is the distance between the camera and the {object}?",
    "Measure the distance from the viewpoint to the {object}.",
)

OBJECT_TO_OBJECT_DISTANCE_TEMPLATES: Sequence[str] = (
    "What is the distance between the {object1} and the {object2}?",
    "How far apart are the {object1} and the {object2}?",
    "Calculate the distance from the {object1} to the {object2}.",
    "Measure the separation between the {object1} and the {object2}.",
)

OBJECT_WIDTH_TEMPLATES: Sequence[str] = (
    "What is the width of the {object}?",
    "How wide is the {object}?",
    "Measure the {object}'s width.",
    "What is the horizontal span of the {object}?",
)

OBJECT_HEIGHT_TEMPLATES: Sequence[str] = (
    "What is the height of the {object}?",
    "How tall is the {object}?",
    "Measure the {object}'s height.",
    "What is the vertical extent of the {object}?",
)

OBJECT_DEPTH_TEMPLATES: Sequence[str] = (
    "What is the depth of the {object}?",
    "How deep is the {object}?",
    "Measure the {object}'s depth.",
    "What is the front-to-back distance of the {object}?",
)

OBJECT_VOLUME_TEMPLATES: Sequence[str] = (
    "What is the approximate volume of the {object}?",
    "Calculate the volume occupied by the {object}.",
    "How much space does the {object} take up?",
    "What is the {object}'s volume?",
)

Question = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episode_dir",
        type=Path,
        help="Path to the episode directory with frame dumps.",
    )
    parser.add_argument(
        "--agent-prefix",
        default="agent_1",
        help="Prefix for agent subdirectories (default: agent_1).",
    )
    parser.add_argument(
        "--camera-distance-questions",
        type=int,
        default=10,
        help="Number of camera-to-object distance questions.",
    )
    parser.add_argument(
        "--object-distance-questions",
        type=int,
        default=8,
        help="Number of object-to-object distance questions.",
    )
    parser.add_argument(
        "--size-questions",
        type=int,
        default=12,
        help="Number of object size questions (width/height/depth/volume).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "metric",
        help="Output directory.",
    )
    parser.add_argument(
        "--output-name",
        default="metric_questions.json",
        help="Output JSON filename.",
    )
    return parser.parse_args()


def meters_to_unit(meters: float) -> float:
    """Convert meters to the configured unit."""
    if DISTANCE_UNIT == "cm":
        return meters * 100
    elif DISTANCE_UNIT == "m":
        return meters
    elif DISTANCE_UNIT == "mm":
        return meters * 1000
    else:
        raise ValueError(f"Unknown distance unit: {DISTANCE_UNIT}")


def format_value(value: float) -> str:
    """Format a measurement value with configured decimal places."""
    return f"{value:.{DECIMAL_PLACES}f}"


def euclidean_distance_3d(pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
    """Calculate 3D Euclidean distance between two positions."""
    dx = pos1["x"] - pos2["x"]
    dy = pos1["y"] - pos2["y"]
    dz = pos1["z"] - pos2["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def normalize_object_name(obj_id: str) -> str:
    """Extract base object name from objectId."""
    return obj_id.split("|")[0] if "|" in obj_id else obj_id


def load_frame_metadata(frame_json: Path) -> Optional[Dict[str, Any]]:
    """Load frame metadata JSON."""
    if not frame_json.exists():
        return None
    try:
        with frame_json.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def collect_frame_data(episode_dir: Path, agent_prefix: str) -> List[Dict[str, Any]]:
    """Collect all frame data from agent directory."""
    agent_dir = episode_dir / agent_prefix
    if not agent_dir.exists():
        # Try to find agent directory by pattern
        candidates = list(episode_dir.glob(f"{agent_prefix}*"))
        if not candidates:
            raise FileNotFoundError(f"No agent directory found with prefix '{agent_prefix}'")
        agent_dir = candidates[0]

    frames = []
    frame_jsons = sorted(agent_dir.glob("img_*.json"))

    for frame_json in frame_jsons:
        metadata = load_frame_metadata(frame_json)
        if metadata:
            frames.append(metadata)

    return frames


def load_thor_metadata(episode_dir: Path) -> Optional[Dict[str, Any]]:
    """Load THOR scene metadata if available."""
    # Try to find metadata file
    metadata_candidates = [
        episode_dir / "scene_metadata.json",
        episode_dir / "metadata.json",
        episode_dir / "run_info.json",
    ]

    for candidate in metadata_candidates:
        if candidate.exists():
            try:
                with candidate.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def extract_object_info_from_thor(thor_metadata: Dict[str, Any], obj_id: str) -> Optional[Dict[str, Any]]:
    """Extract object position and bounding box from THOR metadata."""
    # This is a placeholder - actual implementation depends on metadata structure
    # For now, return None and we'll work with frame-level data
    return None


def generate_camera_distance_question(
    frame_data: Dict[str, Any],
    camera_pos: Dict[str, float],
    object_id: str,
    object_pos: Dict[str, float],
) -> Question:
    """Generate camera-to-object distance question."""
    # Calculate true distance
    distance_m = euclidean_distance_3d(camera_pos, object_pos)
    distance_unit = meters_to_unit(distance_m)
    correct_answer = format_value(distance_unit)

    # Generate distractors with >20% relative error
    distractors = []
    for _ in range(3):
        # Random error between 25% and 80%
        error_factor = random.uniform(0.25, 0.8)
        if random.random() > 0.5:
            distractor_value = distance_unit * (1 + error_factor)
        else:
            distractor_value = distance_unit * (1 - error_factor)
        distractor_value = max(0.01, distractor_value)  # Avoid negative or zero
        distractors.append(format_value(distractor_value))

    # Shuffle options
    options = [correct_answer] + distractors
    random.shuffle(options)

    # Select template and format
    object_name = normalize_object_name(object_id)
    template = random.choice(CAMERA_TO_OBJECT_DISTANCE_TEMPLATES)
    question = template.format(object=object_name)

    return {
        "question": question,
        "options": [f"{opt} {DISTANCE_UNIT}" for opt in options],
        "answer": f"{correct_answer} {DISTANCE_UNIT}",
        "metadata": {
            "type": "camera_to_object_distance",
            "object": object_name,
            "distance_value": float(correct_answer),
            "unit": DISTANCE_UNIT,
            "frame_index": frame_data.get("frame_index"),
        }
    }


def generate_object_distance_question(
    frame_data: Dict[str, Any],
    obj1_id: str,
    obj1_pos: Dict[str, float],
    obj2_id: str,
    obj2_pos: Dict[str, float],
) -> Question:
    """Generate object-to-object distance question."""
    distance_m = euclidean_distance_3d(obj1_pos, obj2_pos)
    distance_unit = meters_to_unit(distance_m)
    correct_answer = format_value(distance_unit)

    # Generate distractors
    distractors = []
    for _ in range(3):
        error_factor = random.uniform(0.25, 0.8)
        if random.random() > 0.5:
            distractor_value = distance_unit * (1 + error_factor)
        else:
            distractor_value = distance_unit * (1 - error_factor)
        distractor_value = max(0.01, distractor_value)
        distractors.append(format_value(distractor_value))

    options = [correct_answer] + distractors
    random.shuffle(options)

    obj1_name = normalize_object_name(obj1_id)
    obj2_name = normalize_object_name(obj2_id)
    template = random.choice(OBJECT_TO_OBJECT_DISTANCE_TEMPLATES)
    question = template.format(object1=obj1_name, object2=obj2_name)

    return {
        "question": question,
        "options": [f"{opt} {DISTANCE_UNIT}" for opt in options],
        "answer": f"{correct_answer} {DISTANCE_UNIT}",
        "metadata": {
            "type": "object_to_object_distance",
            "object1": obj1_name,
            "object2": obj2_name,
            "distance_value": float(correct_answer),
            "unit": DISTANCE_UNIT,
            "frame_index": frame_data.get("frame_index"),
        }
    }


def calculate_bbox_dimensions(bbox: Dict[str, Any]) -> Dict[str, float]:
    """Calculate width, height, depth from axis-aligned bounding box.

    bbox structure: {
        "center": {"x": ..., "y": ..., "z": ...},
        "size": {"x": ..., "y": ..., "z": ...}
    }
    """
    size = bbox.get("size", {})
    width = abs(size.get("x", 0))   # X-axis
    height = abs(size.get("y", 0))  # Y-axis
    depth = abs(size.get("z", 0))   # Z-axis
    volume = width * height * depth

    return {
        "width": width,
        "height": height,
        "depth": depth,
        "volume": volume,
    }


def generate_size_question(
    obj_id: str,
    bbox: Dict[str, Any],
    dimension: str,  # "width", "height", "depth", "volume"
) -> Question:
    """Generate object size question."""
    dims = calculate_bbox_dimensions(bbox)
    true_value_m = dims[dimension]
    true_value_unit = meters_to_unit(true_value_m)

    # For volume, cube the conversion factor
    if dimension == "volume":
        if DISTANCE_UNIT == "cm":
            true_value_unit = (true_value_m * 100) ** 3  # cm³
        elif DISTANCE_UNIT == "m":
            true_value_unit = true_value_m ** 3  # m³

    correct_answer = format_value(true_value_unit)

    # Generate distractors
    distractors = []
    for _ in range(3):
        error_factor = random.uniform(0.25, 0.8)
        if random.random() > 0.5:
            distractor_value = true_value_unit * (1 + error_factor)
        else:
            distractor_value = true_value_unit * (1 - error_factor)
        distractor_value = max(0.01, distractor_value)
        distractors.append(format_value(distractor_value))

    options = [correct_answer] + distractors
    random.shuffle(options)

    obj_name = normalize_object_name(obj_id)

    # Select template
    if dimension == "width":
        template = random.choice(OBJECT_WIDTH_TEMPLATES)
    elif dimension == "height":
        template = random.choice(OBJECT_HEIGHT_TEMPLATES)
    elif dimension == "depth":
        template = random.choice(OBJECT_DEPTH_TEMPLATES)
    else:  # volume
        template = random.choice(OBJECT_VOLUME_TEMPLATES)

    question = template.format(object=obj_name)

    unit_str = DISTANCE_UNIT if dimension != "volume" else f"{DISTANCE_UNIT}³"

    return {
        "question": question,
        "options": [f"{opt} {unit_str}" for opt in options],
        "answer": f"{correct_answer} {unit_str}",
        "metadata": {
            "type": f"object_{dimension}",
            "object": obj_name,
            "dimension": dimension,
            "value": float(correct_answer),
            "unit": unit_str,
        }
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # Load frame data
    print(f"Loading frames from {args.episode_dir}...")
    frames = collect_frame_data(args.episode_dir, args.agent_prefix)

    if not frames:
        raise RuntimeError(f"No frames found in {args.episode_dir}")

    print(f"Loaded {len(frames)} frames.")

    questions: List[Question] = []

    # Generate camera distance questions
    print(f"Generating {args.camera_distance_questions} camera distance questions...")
    camera_distance_count = 0
    attempts = 0
    max_attempts = len(frames) * 10

    while camera_distance_count < args.camera_distance_questions and attempts < max_attempts:
        attempts += 1
        frame = random.choice(frames)

        # Check if frame has required data
        camera_pos = frame.get("camera_position")
        visible_objects = frame.get("visible_objects", [])

        if not camera_pos or not visible_objects:
            continue

        # Pick a random visible object
        obj = random.choice(visible_objects)
        obj_pos = obj.get("position")
        obj_id = obj.get("objectId")

        if not obj_pos or not obj_id:
            continue

        try:
            q = generate_camera_distance_question(frame, camera_pos, obj_id, obj_pos)
            questions.append(q)
            camera_distance_count += 1
        except Exception as e:
            print(f"[WARNING] Failed to generate camera distance question: {e}")
            continue

    # Generate object-to-object distance questions
    print(f"Generating {args.object_distance_questions} object distance questions...")
    object_distance_count = 0
    attempts = 0

    while object_distance_count < args.object_distance_questions and attempts < max_attempts:
        attempts += 1
        frame = random.choice(frames)
        visible_objects = frame.get("visible_objects", [])

        if len(visible_objects) < 2:
            continue

        # Pick two random objects
        obj1, obj2 = random.sample(visible_objects, 2)
        obj1_pos = obj1.get("position")
        obj2_pos = obj2.get("position")
        obj1_id = obj1.get("objectId")
        obj2_id = obj2.get("objectId")

        if not all([obj1_pos, obj2_pos, obj1_id, obj2_id]):
            continue

        try:
            q = generate_object_distance_question(frame, obj1_id, obj1_pos, obj2_id, obj2_pos)
            questions.append(q)
            object_distance_count += 1
        except Exception as e:
            print(f"[WARNING] Failed to generate object distance question: {e}")
            continue

    # Generate size questions
    print(f"Generating {args.size_questions} size questions...")
    size_count = 0
    attempts = 0
    dimensions = ["width", "height", "depth", "volume"]

    while size_count < args.size_questions and attempts < max_attempts:
        attempts += 1
        frame = random.choice(frames)
        visible_objects = frame.get("visible_objects", [])

        if not visible_objects:
            continue

        obj = random.choice(visible_objects)
        obj_id = obj.get("objectId")
        bbox = obj.get("axisAlignedBoundingBox")

        if not obj_id or not bbox:
            continue

        # Check if bbox has valid size
        bbox_size = bbox.get("size", {})
        if not bbox_size or not all(k in bbox_size for k in ["x", "y", "z"]):
            continue

        # Randomly pick a dimension
        dimension = random.choice(dimensions)

        try:
            q = generate_size_question(obj_id, bbox, dimension)
            questions.append(q)
            size_count += 1
        except Exception as e:
            print(f"[WARNING] Failed to generate size question: {e}")
            continue

    # Save output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name

    output_data = {
        "metadata": {
            "episode_dir": str(args.episode_dir),
            "total_questions": len(questions),
            "camera_distance_questions": camera_distance_count,
            "object_distance_questions": object_distance_count,
            "size_questions": size_count,
            "distance_unit": DISTANCE_UNIT,
            "size_unit": SIZE_UNIT,
            "decimal_places": DECIMAL_PLACES,
        },
        "questions": questions,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated {len(questions)} total questions:")
    print(f"  - Camera distance: {camera_distance_count}")
    print(f"  - Object distance: {object_distance_count}")
    print(f"  - Size: {size_count}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
