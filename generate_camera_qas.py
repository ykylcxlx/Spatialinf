"""Generate camera motion QA pairs using ProcTHOR scenes."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from ai2thor.controller import Controller

# python procthordata/generate_camera_qas.py \
#   procthordata/houses/train_000.json \
#   --output-dir outputs/cam_motion/images \
#   --qa-root outputs/cam_motion \
#   --num-samples 3 \
#   --visibility-distance 10 \
#   --connect-timeout 300 \
#   --x-display :1 \
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
    "Which rotation steps connect image 1 with image 2?",
    "How did the camera turn (yaw/pitch) when it moved from shot one to shot two?",
)

DEG_TEMPLATES: Sequence[str] = (
    "[Cam Rotation Deg.] What is the total vertical rotation angle from one shot to another?",
    "State the overall pitch change between image 1 and image 2.",
    "How large is the up/down rotation required to align the first view with the second?",
    "Give the vertical rotation angle that separates the two camera poses.",
)

CATEGORY_TEMPLATES = {
    "Dir": DIR_TEMPLATES,
    "Dist": DIST_TEMPLATES,
    "RotDir": ROT_DIR_TEMPLATES,
    "Deg": DEG_TEMPLATES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("house_json", type=Path, help="Path to a ProcTHOR house JSON file.")
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
        help="Number of QA pairs to generate.",
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
            "Optional JSON file listing custom action sequences. Supply an array of"
            " sequences, each sequence being a list of actions. An action can be a"
            " string (e.g. 'MoveAhead') or an object like {\"action\": \"RotateRight\","
            " \"params\": {\"degrees\": 30}}."
        ),
    )
    parser.add_argument(
        "--x-display",
        default=None,
        help="Optional X display string (e.g., ':1') for headless setups.",
    )
    return parser.parse_args()


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


def format_distance_mm(value: int) -> str:
    return f"`{value}` mm"


def format_degrees(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-6):
        return "0 degrees"
    if float(value).is_integer():
        return f"{int(value)} degrees"
    return f"{value:.2f} degrees"


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    rng = random.Random(args.seed)

    sequences = load_sequences(args.sequences)
    house = load_house(args.house_json)

    controller_kwargs = {
        "scene": house,
        "width": args.width,
        "height": args.height,
        "connect_timeout": args.connect_timeout,
        "visibilityDistance": args.visibility_distance,
    }
    if args.x_display is not None:
        controller_kwargs["x_display"] = args.x_display

    controller = Controller(**controller_kwargs)
    qa_entries: List[dict] = []
    category_entries = {key: [] for key in CATEGORY_TEMPLATES}

    try:
        for idx in range(args.num_samples):
            randomize_start(controller, rng)

            before_event = controller.last_event
            if before_event is None:
                raise RuntimeError("Controller did not return an event after Teleport.")

            agent_before = before_event.metadata["agent"]

            image1_path = args.output_dir / f"sample_{idx:04d}_before.png"
            save_frame(before_event.frame, image1_path)

            actions = sample_action_sequence(sequences, rng)
            attempt_actions(controller, actions)

            after_event = controller.last_event
            if after_event is None:
                raise RuntimeError("Controller did not produce an event after actions.")

            agent_after = after_event.metadata["agent"]
            image2_path = args.output_dir / f"sample_{idx:04d}_after.png"
            save_frame(after_event.frame, image2_path)

            motion_metrics = compute_motion_metrics(agent_before, agent_after)
            action_rotation_summary = summarize_action_rotations(actions)
            action_yaw = action_rotation_summary["action_yaw_deg"]
            action_pitch = action_rotation_summary["action_pitch_deg"]

            dir_question = rng.choice(CATEGORY_TEMPLATES["Dir"])
            dist_question = rng.choice(CATEGORY_TEMPLATES["Dist"])
            rot_dir_question = rng.choice(CATEGORY_TEMPLATES["RotDir"])
            deg_question = rng.choice(CATEGORY_TEMPLATES["Deg"])

            dir_answer = motion_metrics["direction_label"]
            dist_answer = format_distance_mm(motion_metrics["distance_mm"])
            rot_dir_answer = rotation_direction(
                motion_metrics["delta_yaw_deg"],
                motion_metrics["delta_horizon_deg"],
                action_yaw,
                action_pitch,
            )

            vertical_value = (
                action_pitch
                if abs(action_pitch) > VERTICAL_MIN_THRESHOLD
                else motion_metrics["delta_horizon_deg"]
            )
            deg_answer = format_degrees(abs(vertical_value))

            shared_payload = {
                "image_1_path": str(image1_path.resolve()),
                "image_2_path": str(image2_path.resolve()),
                "actions": [
                    {"action": spec["action"], "params": dict(spec.get("params", {}))}
                    for spec in actions
                ],
            }

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
            motion_metadata["action_yaw_deg"] = action_yaw
            motion_metadata["action_pitch_deg"] = action_pitch
            motion_metadata["rotation_label"] = rot_dir_answer

            qa_list: List[dict] = []

            if motion_metrics["planar_distance_meters"] > TRANSLATION_MIN_PLANAR:
                entry = {
                    **shared_payload,
                    "question": dir_question,
                    "answer": dir_answer,
                    "metadata": motion_metadata,
                }
                category_entries["Dir"].append(entry)
                qa_list.append({"question": dir_question, "answer": dir_answer, "category": "Dir"})

            if motion_metrics["distance_meters"] > DISTANCE_MIN_THRESHOLD:
                entry = {
                    **shared_payload,
                    "question": dist_question,
                    "answer": dist_answer,
                    "metadata": motion_metadata,
                }
                category_entries["Dist"].append(entry)
                qa_list.append({"question": dist_question, "answer": dist_answer, "category": "Dist"})

            if (
                max(abs(motion_metrics["delta_yaw_deg"]), abs(action_yaw)) > ROT_DIR_MIN_YAW
                or max(abs(motion_metrics["delta_horizon_deg"]), abs(action_pitch))
                > ROT_DIR_MIN_HORIZON
            ):
                entry = {
                    **shared_payload,
                    "question": rot_dir_question,
                    "answer": rot_dir_answer,
                    "metadata": motion_metadata,
                }
                category_entries["RotDir"].append(entry)
                qa_list.append({"question": rot_dir_question, "answer": rot_dir_answer, "category": "RotDir"})

            if max(abs(motion_metrics["delta_horizon_deg"]), abs(action_pitch)) > VERTICAL_MIN_THRESHOLD:
                entry = {
                    **shared_payload,
                    "question": deg_question,
                    "answer": deg_answer,
                    "metadata": motion_metadata,
                }
                category_entries["Deg"].append(entry)
                qa_list.append({"question": deg_question, "answer": deg_answer, "category": "Deg"})

            if qa_list:
                qa_entries.append(
                    {
                        **shared_payload,
                        "metadata": motion_metadata,
                        "qa": qa_list,
                    }
                )

        qa_root = args.qa_root
        for category, entries in category_entries.items():
            category_dir = qa_root / category
            category_dir.mkdir(parents=True, exist_ok=True)
            category_path = category_dir / "qa_pairs.json"
            category_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            print(f"Saved {len(entries)} QA pairs to {category_path}")

        if args.qa_json is not None:
            args.qa_json.parent.mkdir(parents=True, exist_ok=True)
            args.qa_json.write_text(json.dumps(qa_entries, indent=2), encoding="utf-8")
            print(f"Saved combined QA log to {args.qa_json}")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
