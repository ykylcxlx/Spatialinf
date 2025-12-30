"""Generate camera motion QA pairs using ProcTHOR scenes."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

import numpy as np

from ai2thor.controller import Controller

# python procthordata/generate_camera_qas.py \
#   procthordata/houses/train \
#   --output-dir outputs/cam_motion/images \
#   --qa-root outputs/cam_motion \
#   --num-samples 3 \
#   --visibility-distance 10 \
#   --connect-timeout 300 \
#   --x-display :99 \
#   --qa-json outputs/cam_motion/all_tasks.json

# Action pools for random sequence generation (all rotations < 90°).
TRANSLATION_ACTIONS: Tuple[str, ...] = ("MoveAhead", "MoveBack", "MoveLeft", "MoveRight")
TRANSLATION_STEP_OPTIONS: Tuple[float, ...] = (0.25, 0.5, 0.75)
ROTATION_ACTIONS: Tuple[str, ...] = ("RotateLeft", "RotateRight")
ROTATION_DEGREE_OPTIONS: Tuple[int, ...] = (10, 15, 20, 30, 45, 60, 75, 80)
LOOK_ACTIONS: Tuple[str, ...] = ("LookUp", "LookDown")
LOOK_DEGREE_OPTIONS: Tuple[int, ...] = (5, 8, 10, 12, 15, 18, 20, 25, 30)
MAX_SEQUENCE_LENGTH = 2
OPPOSITE_TRANSLATIONS = {
    "MoveAhead": "MoveBack",
    "MoveBack": "MoveAhead",
    "MoveLeft": "MoveRight",
    "MoveRight": "MoveLeft",
}
TRANSLATION_MIN_PLANAR = 0.03  # meters
DISTANCE_MIN_THRESHOLD = 0.01  # meters
ROT_DIR_MIN_YAW = 5.0  # degrees
ROT_DIR_MIN_HORIZON = 5.0  # degrees
VERTICAL_MIN_THRESHOLD = 2.0  # degrees

DIR_TEMPLATES: Sequence[str] = (
    "[Cam Trans. Dir.] While capturing image 1, where is the camera for image 2 located relative to it?",
    "From the point of view of image 1, which way must I move to reach the spot that produced image 2?",
    "Considering the two shots, where does the second camera sit from the first camera's perspective?",
    "If I stand at the image-1 position, in what direction would I travel to arrive at the image-2 camera?",
)

DIST_TEMPLATES: Sequence[str] = (
    "[Cam Trans. Dist.] What's the measurement of the camera's movement vector's length?",
    "How far, in millimetres, did the camera translate between image 1 and image 2?",
    "Report the magnitude of the translation that carries the camera from shot one to shot two.",
    "Give the travel distance of the camera between these two captures.",
)

ROT_DIR_TEMPLATES: Sequence[str] = (
    "[Cam Rotation Dir.] From image 1 to 2, what is the correct camera rotation?",
    "Describe how the camera must rotate to change from the first frame to the second frame.",
    "When comparing these two shots, which rotation links the first to the second camera pose?",
    "How must the camera rotate to align the perspective of image 1 with that of image 2?",
)

DEG_TEMPLATES: Sequence[str] = (
    "[Cam Rotation Deg.] What is the magnitude of the vertical rotation between the two images?",
    "Report the absolute value of the pitch change that transforms image 1 into image 2.",
    "How many degrees separate the vertical orientations of the two camera poses?",
    "Give the degree value for the up/down rotation when moving from image 1 to image 2.",
)

PROMPT_PREFIX = "<image>Image 1.<image>Image 2."

CHOICE_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D")

DIRECTION_OPTION_POOL: Tuple[str, ...] = (
    "Front",
    "Back",
    "Left",
    "Right",
    "Front-Left",
    "Front-Right",
    "Back-Left",
    "Back-Right",
    "Same Position",
)

ROTATION_OPTION_POOL: Tuple[str, ...] = (
    "Rotate to left",
    "Rotate to right",
    "Look up",
    "Look down",
    "Rotate to left, look up",
    "Rotate to left, look down",
    "Rotate to right, look up",
    "Rotate to right, look down",
    "No rotation change.",
)

CATEGORY_TEMPLATES: Dict[str, Sequence[str]] = {
    "Dir": DIR_TEMPLATES,
    "Dist": DIST_TEMPLATES,
    "RotDir": ROT_DIR_TEMPLATES,
    "Deg": DEG_TEMPLATES,
}


def collect_house_paths(path: Path) -> List[Path]:
    if path.is_dir():
        candidates = sorted(p for p in path.glob("*.json") if p.is_file())
        if not candidates:
            raise FileNotFoundError(f"No JSON files found in directory: {path}")
        return candidates
    if path.is_file() and path.suffix.lower() == ".json":
        return [path]
    raise FileNotFoundError(f"Expected a JSON file or directory, got: {path}")


def write_combined_entries(entries: Sequence[dict], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def direction_option_text(label: str) -> str:
    if label == "Same Position":
        return "Stay at the same position"
    parts = label.split("-")
    if len(parts) == 2:
        return f"Move {parts[0]} then {parts[1]}"
    return f"Move {label}"


def format_distance_mm(value: int) -> str:
    return f"{value} mm"


def format_degrees(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-6):
        return "0 degrees"
    if float(value).is_integer():
        return f"{int(value)} degrees"
    return f"{value:.2f} degrees"


def build_choice_options(
    correct_text: str,
    distractor_texts: Sequence[str],
    rng: random.Random,
) -> Tuple[List[Tuple[str, str]], str]:
    unique_distractors: List[str] = []
    seen = {correct_text.lower()}
    for text in distractor_texts:
        candidate = text.strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered in seen:
            continue
        unique_distractors.append(candidate)
        seen.add(lowered)
        if len(unique_distractors) >= len(CHOICE_LETTERS) - 1:
            break

    while len(unique_distractors) < len(CHOICE_LETTERS) - 1:
        filler = f"Option {len(unique_distractors) + 1}"
        if filler.lower() in seen:
            filler = f"Choice {len(unique_distractors) + 1}"
        unique_distractors.append(filler)
        seen.add(filler.lower())

    options = [correct_text] + unique_distractors[: len(CHOICE_LETTERS) - 1]
    rng.shuffle(options)

    labeled: List[Tuple[str, str]] = []
    correct_label = ""
    for letter, text in zip(CHOICE_LETTERS, options):
        labeled.append((letter, text))
        if text.lower() == correct_text.lower():
            correct_label = letter

    if not correct_label and labeled:
        labeled[0] = (CHOICE_LETTERS[0], correct_text)
        correct_label = CHOICE_LETTERS[0]

    return labeled, correct_label


def build_direction_options(correct_label: str, rng: random.Random) -> Tuple[List[Tuple[str, str]], str]:
    pool = list(DIRECTION_OPTION_POOL)
    if correct_label not in pool:
        pool.append(correct_label)
    correct_text = direction_option_text(correct_label)
    distractors = [direction_option_text(entry) for entry in pool if entry != correct_label]
    rng.shuffle(distractors)
    return build_choice_options(correct_text, distractors, rng)


def build_distance_options(correct_mm: int, rng: random.Random) -> Tuple[List[Tuple[str, str]], str]:
    offsets = [-250, -180, -120, -60, 60, 120, 180, 250, 320, 420]
    candidates: List[int] = []
    for offset in offsets:
        candidate = max(1, correct_mm + offset)
        if candidate != correct_mm:
            candidates.append(candidate)
    while len(candidates) < len(CHOICE_LETTERS) - 1:
        scale = rng.uniform(0.6, 1.6)
        candidate = max(1, int(round(correct_mm * scale)))
        if candidate != correct_mm:
            candidates.append(candidate)
    distractors = [format_distance_mm(value) for value in candidates]
    return build_choice_options(format_distance_mm(correct_mm), distractors, rng)


def build_rotation_options(correct_label: str, rng: random.Random) -> Tuple[List[Tuple[str, str]], str]:
    pool = list(ROTATION_OPTION_POOL)
    if correct_label not in pool:
        pool.append(correct_label)
    distractors = [entry for entry in pool if entry != correct_label]
    rng.shuffle(distractors)
    return build_choice_options(correct_label, distractors, rng)


def build_degree_options(correct_value: float, rng: random.Random) -> Tuple[List[Tuple[str, str]], str]:
    base = abs(correct_value)
    deltas = [3.0, 5.0, 8.0, 10.0, 12.0, 15.0]
    candidates: List[float] = []
    for delta in deltas:
        plus = base + delta
        minus = base - delta
        candidates.append(max(0.5, plus))
        if minus > 0.5:
            candidates.append(minus)
    while len(candidates) < len(CHOICE_LETTERS) - 1:
        candidate = max(0.5, base * rng.uniform(0.5, 1.6))
        if not math.isclose(candidate, base, rel_tol=1e-3, abs_tol=0.1):
            candidates.append(candidate)
    distractors = [format_degrees(value) for value in candidates]
    return build_choice_options(format_degrees(base), distractors, rng)


def format_options_text(options: Sequence[Tuple[str, str]]) -> str:
    return " ".join(f"({label}) {text}" for label, text in options)


def get_answer_text(options: Sequence[Tuple[str, str]], label: str) -> str:
    for option_label, text in options:
        if option_label == label:
            return text
    raise ValueError(f"Missing answer label {label} in options: {options}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "house_json",
        type=Path,
        help="Path to a ProcTHOR house JSON file or a directory containing multiple JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("camera_motion_dataset"),
        help="Directory to store captured frames.",
    )
    parser.add_argument(
        "--qa-root",
        type=Path,
        default=Path("camera_motion_dataset"),
        help="Root directory to store category-specific QA JSON files.",
    )
    parser.add_argument(
        "--qa-json",
        type=Path,
        help="Optional combined QA JSON file (aggregated across categories).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of QA sample sets to generate per house.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for the AI2-THOR backend process to start.",
    )
    parser.add_argument(
        "--visibility-distance",
        type=float,
        default=5.0,
        help="Maximum distance (meters) for objects to be considered visible.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=600,
        help="Viewport width for rendered frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Viewport height for rendered frames.",
    )
    parser.add_argument(
        "--sequences",
        type=Path,
        help=(
            "Optional JSON file listing custom action sequences. Supply an array of sequences, each sequence "
            "being a list of actions. An action can be a string (e.g. 'MoveAhead') or an object like {\"action\": "
            "\"RotateRight\", \"params\": {\"degrees\": 30}}."
        ),
    )
    parser.add_argument(
        "--x-display",
        default=None,
        help="Optional X display string (e.g., ':1') for headless setups.",
    )
    return parser.parse_args()


def create_choice_entry(
    sample_tag: str,
    question_type: str,
    question_text: str,
    options: Sequence[Tuple[str, str]],
    correct_label: str,
    images: Sequence[str],
    depths: Sequence[str],
    metadata: Dict[str, Any],
) -> dict:
    options_text = format_options_text(options)
    answer_text = get_answer_text(options, correct_label)
    return {
        "id": f"{sample_tag}_{question_type.lower()}",
        "question_type": question_type,
        "images": list(images),
        "depths": list(depths),
        "metadata": copy.deepcopy(metadata),
        "messages": [
            {
                "role": "user",
                "content": f"{PROMPT_PREFIX} {question_text} {options_text}",
            },
            {
                "role": "assistant",
                "content": f"({correct_label}) {answer_text}",
            },
        ],
    }
def generate_samples_for_house(
    house_path: Path,
    args: argparse.Namespace,
    sequences: Optional[Sequence[Tuple[ActionSpec, ...]]],
    rng: random.Random,
    start_index: int,
) -> Tuple[List[dict], int, Dict[str, List[dict]]]:
    print(f"Processing house: {house_path}")
    house = load_house(house_path)

    controller_kwargs = {
        "scene": house,
        "width": args.width,
        "height": args.height,
        "connect_timeout": args.connect_timeout,
        "visibilityDistance": args.visibility_distance,
        "renderDepthImage": True,
    }
    if args.x_display is not None:
        controller_kwargs["x_display"] = args.x_display

    controller = Controller(**controller_kwargs)
    qa_entries: List[dict] = []
    category_entries: Dict[str, List[dict]] = {key: [] for key in CATEGORY_TEMPLATES}
    generated_samples = 0
    failure_count = 0
    max_failures = args.num_samples * 12

    try:
        while generated_samples < args.num_samples and failure_count < max_failures:
            randomize_start(controller, rng)
            before_event = controller.last_event
            if before_event is None:
                raise RuntimeError("Controller did not return an event after Teleport.")

            actions = sample_action_sequence(sequences, rng)

            try:
                attempt_actions(controller, actions)
            except Exception as exc:  # noqa: BLE001
                print(f"Action sequence failed for {house_path.name}: {exc}")
                failure_count += 1
                continue

            after_event = controller.last_event
            if after_event is None:
                raise RuntimeError("Controller did not produce an event after actions.")

            agent_before = before_event.metadata["agent"]
            agent_after = after_event.metadata["agent"]

            sequence_index = start_index + generated_samples
            sample_tag = f"camera_{sequence_index:06d}"

            img1_path = args.output_dir / f"{sample_tag}_img1.png"
            img2_path = args.output_dir / f"{sample_tag}_img2.png"
            save_frame(before_event.frame, img1_path)
            save_frame(after_event.frame, img2_path)

            depth1_path = args.output_dir / f"{sample_tag}_img1_depth.png"
            depth2_path = args.output_dir / f"{sample_tag}_img2_depth.png"
            save_depth_image(before_event.depth_frame, depth1_path)
            save_depth_image(after_event.depth_frame, depth2_path)

            base_dir = args.qa_json.parent if args.qa_json else args.qa_root
            rel_img1 = os.path.relpath(img1_path, base_dir).replace("\\", "/")
            rel_img2 = os.path.relpath(img2_path, base_dir).replace("\\", "/")
            images_rel = [rel_img1, rel_img2]
            rel_depth1 = os.path.relpath(depth1_path, base_dir).replace("\\", "/")
            rel_depth2 = os.path.relpath(depth2_path, base_dir).replace("\\", "/")
            depths_rel = [rel_depth1, rel_depth2]

            motion_metrics = compute_motion_metrics(agent_before, agent_after)
            rotation_summary = summarize_action_rotations(actions)
            action_yaw = rotation_summary["action_yaw_deg"]
            action_pitch = rotation_summary["action_pitch_deg"]

            dir_answer = motion_metrics["direction_label"]
            dist_answer_mm = motion_metrics["distance_mm"]
            rot_dir_answer = rotation_direction(
                motion_metrics["delta_yaw_deg"],
                motion_metrics["delta_horizon_deg"],
                action_yaw,
                action_pitch,
            )

            vertical_value = (
                action_pitch if abs(action_pitch) > VERTICAL_MIN_THRESHOLD else motion_metrics["delta_horizon_deg"]
            )
            deg_answer_value = abs(vertical_value)

            motion_metadata = {
                key: motion_metrics[key]
                for key in (
                    "delta_position_meters",
                    "distance_meters",
                    "distance_mm",
                    "planar_distance_meters",
                    "initial_yaw_deg",
                    "final_yaw_deg",
                    "delta_yaw_deg",
                    "delta_horizon_deg",
                    "direction_label",
                )
            }
            motion_metadata.update(
                {
                    "action_yaw_deg": action_yaw,
                    "action_pitch_deg": action_pitch,
                    "rotation_label": rot_dir_answer,
                }
            )

            metadata = {
                "house": house_path.stem,
                "actions": [
                    {"action": spec["action"], "params": dict(spec.get("params", {}))}
                    for spec in actions
                ],
                "motion": motion_metadata,
            }

            created_any = False

            if motion_metrics["planar_distance_meters"] > TRANSLATION_MIN_PLANAR:
                question_text = rng.choice(CATEGORY_TEMPLATES["Dir"])
                options, correct_label = build_direction_options(dir_answer, rng)
                entry = create_choice_entry(
                    sample_tag,
                    "Dir",
                    question_text,
                    options,
                    correct_label,
                    images_rel,
                    depths_rel,
                    metadata,
                )
                qa_entries.append(entry)
                category_entries["Dir"].append(copy.deepcopy(entry))
                created_any = True

            if motion_metrics["distance_meters"] > DISTANCE_MIN_THRESHOLD:
                question_text = rng.choice(CATEGORY_TEMPLATES["Dist"])
                options, correct_label = build_distance_options(dist_answer_mm, rng)
                entry = create_choice_entry(
                    sample_tag,
                    "Dist",
                    question_text,
                    options,
                    correct_label,
                    images_rel,
                    depths_rel,
                    metadata,
                )
                qa_entries.append(entry)
                category_entries["Dist"].append(copy.deepcopy(entry))
                created_any = True

            if (
                max(abs(motion_metrics["delta_yaw_deg"]), abs(action_yaw)) > ROT_DIR_MIN_YAW
                or max(abs(motion_metrics["delta_horizon_deg"]), abs(action_pitch)) > ROT_DIR_MIN_HORIZON
            ):
                question_text = rng.choice(CATEGORY_TEMPLATES["RotDir"])
                options, correct_label = build_rotation_options(rot_dir_answer, rng)
                entry = create_choice_entry(
                    sample_tag,
                    "RotDir",
                    question_text,
                    options,
                    correct_label,
                    images_rel,
                    depths_rel,
                    metadata,
                )
                qa_entries.append(entry)
                category_entries["RotDir"].append(copy.deepcopy(entry))
                created_any = True

            if max(abs(motion_metrics["delta_horizon_deg"]), abs(action_pitch)) > VERTICAL_MIN_THRESHOLD:
                question_text = rng.choice(CATEGORY_TEMPLATES["Deg"])
                options, correct_label = build_degree_options(deg_answer_value, rng)
                entry = create_choice_entry(
                    sample_tag,
                    "Deg",
                    question_text,
                    options,
                    correct_label,
                    images_rel,
                    depths_rel,
                    metadata,
                )
                qa_entries.append(entry)
                category_entries["Deg"].append(copy.deepcopy(entry))
                created_any = True

            if created_any:
                generated_samples += 1
            else:
                failure_count += 1

    finally:
        controller.stop()

    if generated_samples < args.num_samples:
        print(
            f"Warning: only generated {generated_samples} / {args.num_samples} sample sets for house {house_path.name}."
        )

    return qa_entries, generated_samples, category_entries
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.qa_root.mkdir(parents=True, exist_ok=True)
    if args.qa_json is not None:
        args.qa_json.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    rng = random.Random(args.seed)

    sequences = load_sequences(args.sequences)
    house_paths = collect_house_paths(args.house_json)

    qa_entries: List[dict] = []
    category_totals: Dict[str, List[dict]] = {key: [] for key in CATEGORY_TEMPLATES}
    total_samples = 0

    output_json = args.qa_json or (args.qa_root / "qa_pairs.json")

    for house_path in house_paths:
        try:
            house_entries, produced, category_entries = generate_samples_for_house(
                house_path, args, sequences, rng, total_samples
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to generate samples for {house_path}: {exc}")
            continue

        qa_entries.extend(house_entries)
        for key in CATEGORY_TEMPLATES:
            category_totals[key].extend(category_entries[key])
        total_samples += produced

        write_combined_entries(qa_entries, output_json)
        print(f"Appended {len(house_entries)} entries -> {output_json}")

    for category, entries in category_totals.items():
        category_dir = args.qa_root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        category_path = category_dir / "qa_pairs.json"
        write_combined_entries(entries, category_path)
        print(f"Saved {len(entries)} entries for category {category} -> {category_path}")

    print(f"Saved total {len(qa_entries)} QA pairs to {output_json}")


def load_house(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"House JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


ActionSpec = Dict[str, Any]


def normalize_action_spec(spec: Any) -> ActionSpec:
    if isinstance(spec, str):
        return {"action": spec, "params": {}}
    if isinstance(spec, dict):
        action_name = spec.get("action") or spec.get("name")
        if not isinstance(action_name, str):
            raise ValueError("Action specification must include an 'action' string field.")
        params = spec.get("params")
        if params is None:
            params = {k: v for k, v in spec.items() if k not in {"action", "name", "params"}}
        if not isinstance(params, dict):
            raise ValueError("Action 'params' must be a dictionary when provided.")
        return {"action": action_name, "params": params}
    raise ValueError("Each action must be a string or an object with an 'action' field.")


def normalize_sequence(raw_sequence: Sequence[Any]) -> Tuple[ActionSpec, ...]:
    if not raw_sequence:
        raise ValueError("Action sequences must not be empty.")
    return tuple(normalize_action_spec(action) for action in raw_sequence)


def load_sequences(path: Path | None) -> Optional[Sequence[Tuple[ActionSpec, ...]]]:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Custom sequences file must contain a JSON list.")
    sequences: List[Tuple[ActionSpec, ...]] = []
    for seq in data:
        if not isinstance(seq, list):
            raise ValueError("Each custom sequence must be a list of actions.")
        sequences.append(normalize_sequence(seq))
    return sequences
def save_frame(frame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(destination)


def save_depth_image(depth_frame: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if depth_frame is None:
        return
    depth_m = np.nan_to_num(depth_frame, nan=0.0, posinf=0.0, neginf=0.0)
    depth_mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
    Image.fromarray(depth_mm).save(destination)


def wrapped_angle(delta: float) -> float:
    """Wrap delta yaw to [-180, 180] degrees."""
    wrapped = (delta + 180.0) % 360.0 - 180.0
    return wrapped


def translation_direction(delta_pos: dict, initial_yaw: float) -> str:
    dx = delta_pos["x"]
    dz = delta_pos["z"]
    yaw_rad = math.radians(initial_yaw)
    local_forward = math.cos(yaw_rad) * dz + math.sin(yaw_rad) * dx
    local_right = -math.sin(yaw_rad) * dz + math.cos(yaw_rad) * dx

    threshold = 0.02
    horizontal = None
    vertical = None
    if local_right > threshold:
        horizontal = "Right"
    elif local_right < -threshold:
        horizontal = "Left"
    if local_forward > threshold:
        vertical = "Front"
    elif local_forward < -threshold:
        vertical = "Back"

    if horizontal and vertical:
        return f"{horizontal}-{vertical}"
    if horizontal:
        return horizontal
    if vertical:
        return vertical
    return "Same Position"


def summarize_action_rotations(actions: Sequence[ActionSpec]) -> Dict[str, float]:
    yaw_total = 0.0
    pitch_total = 0.0
    for spec in actions:
        params = spec.get("params", {})
        action = spec["action"]
        if action == "RotateLeft":
            yaw_total += float(params.get("degrees", 90.0))
        elif action == "RotateRight":
            yaw_total -= float(params.get("degrees", 90.0))
        elif action == "LookUp":
            pitch_total -= float(params.get("degrees", 30.0))
        elif action == "LookDown":
            pitch_total += float(params.get("degrees", 30.0))
    return {"action_yaw_deg": yaw_total, "action_pitch_deg": pitch_total}


def rotation_direction(
    delta_yaw: float,
    delta_horizon: float,
    action_yaw: float,
    action_pitch: float,
) -> str:
    effective_yaw = action_yaw if abs(action_yaw) > ROT_DIR_MIN_YAW else delta_yaw
    effective_pitch = action_pitch if abs(action_pitch) > ROT_DIR_MIN_HORIZON else delta_horizon

    parts: List[str] = []
    if effective_yaw > ROT_DIR_MIN_YAW:
        parts.append("rotate to left")
    elif effective_yaw < -ROT_DIR_MIN_YAW:
        parts.append("rotate to right")

    if effective_pitch < -ROT_DIR_MIN_HORIZON:
        parts.append("look up")
    elif effective_pitch > ROT_DIR_MIN_HORIZON:
        parts.append("look down")

    if not parts:
        return "No rotation change."
    answer = ", ".join(parts)
    return answer[0].upper() + answer[1:]


def horizon_delta(initial_agent: dict, final_agent: dict) -> float:
    return final_agent.get("cameraHorizon", 0.0) - initial_agent.get("cameraHorizon", 0.0)


def attempt_actions(controller: Controller, actions: Sequence[ActionSpec]) -> None:
    for spec in actions:
        params = dict(spec.get("params", {}))
        controller.step(action=spec["action"], raise_for_failure=True, **params)


def sample_translation_action(rng: random.Random) -> ActionSpec:
    action = rng.choice(TRANSLATION_ACTIONS)
    params: Dict[str, Any] = {}
    if rng.random() < 0.7:
        params["moveMagnitude"] = rng.choice(TRANSLATION_STEP_OPTIONS)
    return {"action": action, "params": params}


def sample_rotation_action(rng: random.Random) -> ActionSpec:
    action = rng.choice(ROTATION_ACTIONS)
    degrees = rng.choice(ROTATION_DEGREE_OPTIONS)
    return {"action": action, "params": {"degrees": degrees}}


def sample_look_action(rng: random.Random) -> ActionSpec:
    action = rng.choice(LOOK_ACTIONS)
    degrees = rng.choice(LOOK_DEGREE_OPTIONS)
    return {"action": action, "params": {"degrees": degrees}}


def generate_random_sequence(rng: random.Random) -> Tuple[ActionSpec, ...]:
    length = 1 if rng.random() < 0.45 else MAX_SEQUENCE_LENGTH
    sequence: List[ActionSpec] = []
    for idx in range(length):
        category_pool = ["translate", "rotate", "look"]
        # Bias the first action toward translations for richer motion context.
        if idx == 0:
            category_pool = ["translate", "translate", "rotate", "translate", "look"]

        candidate: Optional[ActionSpec] = None
        attempts = 0
        while attempts < 8 and candidate is None:
            category = rng.choice(category_pool)
            if category == "translate":
                action_spec = sample_translation_action(rng)
                if sequence:
                    prev_action = sequence[-1]["action"]
                    if prev_action in OPPOSITE_TRANSLATIONS and action_spec["action"] == OPPOSITE_TRANSLATIONS[prev_action]:
                        attempts += 1
                        continue
                candidate = action_spec
            elif category == "rotate":
                candidate = sample_rotation_action(rng)
            else:
                candidate = sample_look_action(rng)
            attempts += 1

        if candidate is None:
            # Fallback to a small yaw adjustment to guarantee progress.
            candidate = sample_rotation_action(rng)

        sequence.append(candidate)

    return tuple(sequence)


def sample_action_sequence(
    sequences: Optional[Sequence[Tuple[ActionSpec, ...]]], rng: random.Random
) -> Tuple[ActionSpec, ...]:
    if sequences:
        return rng.choice(sequences)
    return generate_random_sequence(rng)


def randomize_start(controller: Controller, rng: random.Random) -> None:
    event = controller.step(action="GetReachablePositions", raise_for_failure=True)
    reachable: Iterable[dict] = event.metadata["actionReturn"]
    positions = list(reachable)
    if not positions:
        raise RuntimeError("No reachable positions returned by the scene.")
    position = rng.choice(positions)
    rotation_y = rng.choice([0, 90, 180, 270])
    controller.step(
        action="Teleport",
        position=position,
        rotation={"x": 0, "y": rotation_y, "z": 0},
        horizon=0.0,
        raise_for_failure=True,
    )


def compute_motion_metrics(agent_before: dict, agent_after: dict) -> dict:
    delta_pos = {
        "x": agent_after["position"]["x"] - agent_before["position"]["x"],
        "y": agent_after["position"]["y"] - agent_before["position"]["y"],
        "z": agent_after["position"]["z"] - agent_before["position"]["z"],
    }
    distance = math.sqrt(
        delta_pos["x"] ** 2 + delta_pos["y"] ** 2 + delta_pos["z"] ** 2
    )
    planar_distance = math.sqrt(delta_pos["x"] ** 2 + delta_pos["z"] ** 2)
    distance_mm = round(distance * 1000.0)

    initial_yaw = agent_before["rotation"]["y"]
    final_yaw = agent_after["rotation"]["y"]
    delta_yaw = wrapped_angle(final_yaw - initial_yaw)

    delta_horizon = horizon_delta(agent_before, agent_after)

    direction_label = translation_direction(delta_pos, initial_yaw)

    return {
        "delta_position_meters": delta_pos,
        "distance_meters": distance,
        "distance_mm": distance_mm,
        "planar_distance_meters": planar_distance,
        "initial_yaw_deg": initial_yaw,
        "final_yaw_deg": final_yaw,
        "delta_yaw_deg": delta_yaw,
        "delta_horizon_deg": delta_horizon,
        "direction_label": direction_label,
    }


if __name__ == "__main__":
    main()
