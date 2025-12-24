"""Generate spatial relation QA pairs from recorded ProcTHOR agent videos.

This script generates questions about:
1. Egocentric spatial relations (object direction relative to camera)
2. Object distance comparison (which object is farther/closer)
3. Object size comparison (which object is larger/smaller)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Direction definitions (8 directions)
# Angles are measured from camera's forward direction (0 degrees)
DIRECTION_DEFINITIONS = {
    "front": (-30, 30),
    "front-right": (15, 75),
    "right": (45, 135),
    "back-right": (105, 165),
    "back": (135, -135),  # Special case: wraps around
    "back-left": (-165, -105),
    "left": (-135, -45),
    "front-left": (-75, -15),
}

# Question templates
EGOCENTRIC_DIRECTION_TEMPLATES: Sequence[str] = (
    "From the camera's perspective, in which direction is the {object}?",
    "Where is the {object} located relative to the camera?",
    "In what direction should the camera turn to face the {object}?",
    "Relative to the viewpoint, the {object} is positioned to the ___.",
)

DISTANCE_COMPARISON_TEMPLATES: Sequence[str] = (
    "Which object is farther from the camera: the {object1} or the {object2}?",
    "Between the {object1} and the {object2}, which one is closer to the camera?",
    "Comparing distances to the camera, which is more distant: {object1} or {object2}?",
    "Which object is nearer to the viewpoint: {object1} or {object2}?",
)

SIZE_COMPARISON_TEMPLATES: Sequence[str] = (
    "Which object is larger: the {object1} or the {object2}?",
    "Between the {object1} and the {object2}, which one is bigger?",
    "Comparing physical size, which is larger: {object1} or {object2}?",
    "Which object occupies more space: {object1} or {object2}?",
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
        "--egocentric-questions",
        type=int,
        default=12,
        help="Number of egocentric direction questions.",
    )
    parser.add_argument(
        "--distance-comparison-questions",
        type=int,
        default=10,
        help="Number of distance comparison questions.",
    )
    parser.add_argument(
        "--size-comparison-questions",
        type=int,
        default=10,
        help="Number of size comparison questions.",
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
        default=Path("outputs") / "spatial",
        help="Output directory.",
    )
    parser.add_argument(
        "--output-name",
        default="spatial_questions.json",
        help="Output JSON filename.",
    )
    return parser.parse_args()


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


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-180, 180] range."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def calculate_object_direction(camera_pos: Dict[str, float], camera_yaw: float, object_pos: Dict[str, float]) -> float:
    """Calculate the direction angle from camera to object.

    Returns angle in degrees, where:
    - 0 degrees = straight ahead
    - Positive = right side
    - Negative = left side
    """
    # Calculate vector from camera to object (in X-Z plane)
    dx = object_pos["x"] - camera_pos["x"]
    dz = object_pos["z"] - camera_pos["z"]

    # Calculate absolute angle (0 = +Z direction, counterclockwise)
    obj_angle = math.degrees(math.atan2(dx, dz))

    # Convert camera yaw to same coordinate system
    # In AI2-THOR, rotation.y is yaw (0 = +Z, clockwise is positive)
    # We need to invert it for our calculation
    camera_forward_angle = -camera_yaw

    # Calculate relative angle
    relative_angle = normalize_angle(obj_angle - camera_forward_angle)

    return relative_angle


def get_direction_label(angle: float) -> str:
    """Get direction label from angle, preferring more specific directions."""
    # Try 4-way directions first (more specific)
    four_way = ["front", "right", "back", "left"]
    eight_way = list(DIRECTION_DEFINITIONS.keys())

    # Check which directions match
    matches = []
    for direction, (min_angle, max_angle) in DIRECTION_DEFINITIONS.items():
        if direction == "back":
            # Special case for back (wraps around ±180)
            if angle >= 135 or angle <= -135:
                matches.append(direction)
        else:
            if min_angle <= angle <= max_angle:
                matches.append(direction)

    if not matches:
        # Fallback - shouldn't happen with proper definitions
        return "front"

    # Prefer 4-way directions if available
    for direction in four_way:
        if direction in matches:
            return direction

    # Otherwise return first match
    return matches[0]


def get_all_valid_directions(angle: float) -> List[str]:
    """Get all valid direction labels for an angle."""
    matches = []
    for direction, (min_angle, max_angle) in DIRECTION_DEFINITIONS.items():
        if direction == "back":
            if angle >= 135 or angle <= -135:
                matches.append(direction)
        else:
            if min_angle <= angle <= max_angle:
                matches.append(direction)
    return matches


def euclidean_distance_3d(pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
    """Calculate 3D Euclidean distance."""
    dx = pos1["x"] - pos2["x"]
    dy = pos1["y"] - pos2["y"]
    dz = pos1["z"] - pos2["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_object_volume(bbox: Dict[str, Any]) -> float:
    """Calculate object volume from bounding box."""
    size = bbox.get("size", {})
    width = abs(size.get("x", 0))
    height = abs(size.get("y", 0))
    depth = abs(size.get("z", 0))
    return width * height * depth


def generate_egocentric_question(
    frame_data: Dict[str, Any],
    camera_pos: Dict[str, float],
    camera_rotation: Dict[str, float],
    obj_id: str,
    obj_pos: Dict[str, float],
) -> Question:
    """Generate egocentric direction question."""
    camera_yaw = camera_rotation.get("y", 0)
    angle = calculate_object_direction(camera_pos, camera_yaw, obj_pos)

    # Get the primary direction
    correct_direction = get_direction_label(angle)

    # Get all valid directions to avoid ambiguity
    all_valid = get_all_valid_directions(angle)

    # Generate distractor directions (excluding all valid ones)
    all_directions = list(DIRECTION_DEFINITIONS.keys())
    distractors = [d for d in all_directions if d not in all_valid]
    random.shuffle(distractors)
    distractors = distractors[:3]

    # Build options
    options = [correct_direction] + distractors
    random.shuffle(options)

    obj_name = normalize_object_name(obj_id)
    template = random.choice(EGOCENTRIC_DIRECTION_TEMPLATES)
    question = template.format(object=obj_name)

    return {
        "question": question,
        "options": options,
        "answer": correct_direction,
        "metadata": {
            "type": "egocentric_direction",
            "object": obj_name,
            "direction": correct_direction,
            "angle": round(angle, 2),
            "all_valid_directions": all_valid,
            "frame_index": frame_data.get("frame_index"),
        }
    }


def generate_distance_comparison_question(
    frame_data: Dict[str, Any],
    camera_pos: Dict[str, float],
    obj1_id: str,
    obj1_pos: Dict[str, float],
    obj2_id: str,
    obj2_pos: Dict[str, float],
    question_type: str = "farther",  # "farther" or "closer"
) -> Question:
    """Generate distance comparison question."""
    dist1 = euclidean_distance_3d(camera_pos, obj1_pos)
    dist2 = euclidean_distance_3d(camera_pos, obj2_pos)

    obj1_name = normalize_object_name(obj1_id)
    obj2_name = normalize_object_name(obj2_id)

    if question_type == "farther":
        correct_answer = obj1_name if dist1 > dist2 else obj2_name
    else:  # closer
        correct_answer = obj1_name if dist1 < dist2 else obj2_name

    # Generate template based on question type
    if question_type == "farther":
        template = "Which object is farther from the camera: the {object1} or the {object2}?"
    else:
        template = "Which object is closer to the camera: the {object1} or the {object2}?"

    question = template.format(object1=obj1_name, object2=obj2_name)

    options = [obj1_name, obj2_name]
    random.shuffle(options)

    return {
        "question": question,
        "options": options,
        "answer": correct_answer,
        "metadata": {
            "type": f"distance_comparison_{question_type}",
            "object1": obj1_name,
            "object2": obj2_name,
            "distance1": round(dist1, 4),
            "distance2": round(dist2, 4),
            "frame_index": frame_data.get("frame_index"),
        }
    }


def generate_size_comparison_question(
    obj1_id: str,
    obj1_bbox: Dict[str, Any],
    obj2_id: str,
    obj2_bbox: Dict[str, Any],
) -> Question:
    """Generate size comparison question."""
    vol1 = calculate_object_volume(obj1_bbox)
    vol2 = calculate_object_volume(obj2_bbox)

    obj1_name = normalize_object_name(obj1_id)
    obj2_name = normalize_object_name(obj2_id)

    correct_answer = obj1_name if vol1 > vol2 else obj2_name

    template = random.choice(SIZE_COMPARISON_TEMPLATES)
    question = template.format(object1=obj1_name, object2=obj2_name)

    options = [obj1_name, obj2_name]
    random.shuffle(options)

    return {
        "question": question,
        "options": options,
        "answer": correct_answer,
        "metadata": {
            "type": "size_comparison",
            "object1": obj1_name,
            "object2": obj2_name,
            "volume1": round(vol1, 6),
            "volume2": round(vol2, 6),
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
    max_attempts = len(frames) * 10

    # Generate egocentric direction questions
    print(f"Generating {args.egocentric_questions} egocentric direction questions...")
    ego_count = 0
    attempts = 0

    while ego_count < args.egocentric_questions and attempts < max_attempts:
        attempts += 1
        frame = random.choice(frames)

        camera_pos = frame.get("camera_position")
        camera_rot = frame.get("camera_rotation")
        visible_objects = frame.get("visible_objects", [])

        if not camera_pos or not camera_rot or not visible_objects:
            continue

        obj = random.choice(visible_objects)
        obj_pos = obj.get("position")
        obj_id = obj.get("objectId")

        if not obj_pos or not obj_id:
            continue

        try:
            q = generate_egocentric_question(frame, camera_pos, camera_rot, obj_id, obj_pos)
            questions.append(q)
            ego_count += 1
        except Exception as e:
            print(f"[WARNING] Failed to generate egocentric question: {e}")
            continue

    # Generate distance comparison questions
    print(f"Generating {args.distance_comparison_questions} distance comparison questions...")
    dist_count = 0
    attempts = 0

    while dist_count < args.distance_comparison_questions and attempts < max_attempts:
        attempts += 1
        frame = random.choice(frames)

        camera_pos = frame.get("camera_position")
        visible_objects = frame.get("visible_objects", [])

        if not camera_pos or len(visible_objects) < 2:
            continue

        obj1, obj2 = random.sample(visible_objects, 2)
        obj1_pos = obj1.get("position")
        obj2_pos = obj2.get("position")
        obj1_id = obj1.get("objectId")
        obj2_id = obj2.get("objectId")

        if not all([obj1_pos, obj2_pos, obj1_id, obj2_id]):
            continue

        # Randomly choose question type
        q_type = random.choice(["farther", "closer"])

        try:
            q = generate_distance_comparison_question(
                frame, camera_pos, obj1_id, obj1_pos, obj2_id, obj2_pos, q_type
            )
            questions.append(q)
            dist_count += 1
        except Exception as e:
            print(f"[WARNING] Failed to generate distance comparison question: {e}")
            continue

    # Generate size comparison questions
    print(f"Generating {args.size_comparison_questions} size comparison questions...")
    size_count = 0
    attempts = 0

    while size_count < args.size_comparison_questions and attempts < max_attempts:
        attempts += 1
        frame = random.choice(frames)
        visible_objects = frame.get("visible_objects", [])

        if len(visible_objects) < 2:
            continue

        obj1, obj2 = random.sample(visible_objects, 2)
        obj1_bbox = obj1.get("axisAlignedBoundingBox")
        obj2_bbox = obj2.get("axisAlignedBoundingBox")
        obj1_id = obj1.get("objectId")
        obj2_id = obj2.get("objectId")

        if not all([obj1_bbox, obj2_bbox, obj1_id, obj2_id]):
            continue

        # Check if bboxes have valid size
        if not all(k in obj1_bbox.get("size", {}) for k in ["x", "y", "z"]):
            continue
        if not all(k in obj2_bbox.get("size", {}) for k in ["x", "y", "z"]):
            continue

        try:
            q = generate_size_comparison_question(obj1_id, obj1_bbox, obj2_id, obj2_bbox)
            questions.append(q)
            size_count += 1
        except Exception as e:
            print(f"[WARNING] Failed to generate size comparison question: {e}")
            continue

    # Save output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name

    output_data = {
        "metadata": {
            "episode_dir": str(args.episode_dir),
            "total_questions": len(questions),
            "egocentric_questions": ego_count,
            "distance_comparison_questions": dist_count,
            "size_comparison_questions": size_count,
        },
        "questions": questions,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated {len(questions)} total questions:")
    print(f"  - Egocentric direction: {ego_count}")
    print(f"  - Distance comparison: {dist_count}")
    print(f"  - Size comparison: {size_count}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
