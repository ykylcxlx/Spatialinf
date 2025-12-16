"""Generate correspondence-point QA pairs using ProcTHOR scenes."""



# python generate_corr_point_qas.py houses/train_000.json --output-dir ../outputs/corr_point/images --qa-root ../outputs/corr_point --num-samples 20 --visibility-distance 10 --connect-timeout 300 --x-display :1 --qa-json ../outputs/corr_point/all_tasks.json
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from ai2thor.controller import Controller

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

QUESTION_TEMPLATES: Sequence[str] = (
    "[Corr Point] Match the point from image 1 with the correct point in image 2.",
    "[Corr Point] Identify which label in picture 2 lines up with the red dot in picture 1.",
    "[Corr Point] 图1中的红点在图2对应哪个字母标记？",
    "[Corr Point] 请选择图2中与第一张图高亮红点位置相同的蓝色字母点。",
    "[Corr Point] Determine the labeled point in image 2 that coincides with the highlighted point in image 1.",
    "[Corr Point] 请找出第二张图里与第一张图红点重合的标记点。",
)

COORD_TEMPLATES: Sequence[str] = (
    "[Corr Coord]The object at {bbox} in image 1 is at which bbox in image 2?",
    "[Corr Coord]Select the image-2 bounding box that aligns with image-1 region {bbox}.",
    "[Corr Coord] 图1 中框 {bbox} 在图2 对应哪个选项框？",
    "[Corr Coord] 请选择图2中与图1 区域 {bbox} 相匹配的边界框。",
    "[Corr Coord]Which candidate bbox in image 2 matches the object covering {bbox} in image 1?",
)

OBJECT_TEMPLATES: Sequence[str] = (
    "[Corr Object]What object exists in both images?",
    "[Corr Object]Name the item visible in image 1 and image 2.",
    "[Corr Object]Which object is shared between the two shots?",
    "[Corr Object] 图1 和图2 都出现了哪件物体？",
    "[Corr Object] 请选择两张图里共同存在的物体。",
)

OBJECT_DISTRACTOR_POOL: Sequence[str] = (
    "Chair",
    "Table",
    "Bottle",
    "Cup",
    "Spoon",
    "Plate",
    "Microwave",
    "Fridge",
    "Lamp",
    "Book",
    "Laptop",
    "Sink",
)

POINT_RADIUS_PIXELS = 8
DISTRACTOR_TARGET = 3
MIN_POINT_SPACING = 24.0
MAX_SAMPLE_ATTEMPTS = 12


@dataclass(frozen=True)
class ActionSpec:
    action: str
    params: Dict[str, Any]


@dataclass
class VisibleObject:
    object_id: str
    object_type: str
    name: str
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]


@dataclass
class LabeledPoint:
    label: str
    x: float
    y: float
    is_correct: bool
    source: str
    object_id: Optional[str]
    object_type: Optional[str]
    bbox: Optional[Tuple[float, float, float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("house_json", type=Path, help="Path to a ProcTHOR house JSON file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("corr_point_dataset"),
        help="Directory to store annotated frames.",
    )
    parser.add_argument(
        "--qa-root",
        type=Path,
        default=Path("corr_point_dataset"),
        help="Root directory to store QA JSON outputs.",
    )
    parser.add_argument(
        "--qa-json",
        type=Path,
        help="Optional combined QA JSON file (aggregated across samples).",
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
        default=13,
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
            "Optional JSON file listing custom action sequences. Supply an array of sequences, each sequence being a list of actions. "
            "An action can be a string (e.g. 'MoveAhead') or an object like {\"action\": \"RotateRight\", \"params\": {\"degrees\": 30}}."
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


def normalize_action_spec(spec: Any) -> ActionSpec:
    if isinstance(spec, str):
        return ActionSpec(action=spec, params={})
    if isinstance(spec, dict):
        action_name = spec.get("action") or spec.get("name")
        if not isinstance(action_name, str):
            raise ValueError("Action specification must include an 'action' string field.")
        params = spec.get("params")
        if params is None:
            params = {k: v for k, v in spec.items() if k not in {"action", "name", "params"}}
        if not isinstance(params, dict):
            raise ValueError("Action 'params' must be a dictionary when provided.")
        return ActionSpec(action=action_name, params=dict(params))
    raise ValueError("Each action must be a string or an object with an 'action' field.")


def normalize_sequence(raw_sequence: Sequence[Any]) -> Tuple[ActionSpec, ...]:
    if not raw_sequence:
        raise ValueError("Action sequences must not be empty.")
    return tuple(normalize_action_spec(action) for action in raw_sequence)


def load_sequences(path: Optional[Path]) -> Optional[Sequence[Tuple[ActionSpec, ...]]]:
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


def sample_translation_action(rng: random.Random) -> ActionSpec:
    action = rng.choice(TRANSLATION_ACTIONS)
    params: Dict[str, Any] = {}
    if rng.random() < 0.7:
        params["moveMagnitude"] = rng.choice(TRANSLATION_STEP_OPTIONS)
    return ActionSpec(action=action, params=params)


def sample_rotation_action(rng: random.Random) -> ActionSpec:
    action = rng.choice(ROTATION_ACTIONS)
    degrees = rng.choice(ROTATION_DEGREE_OPTIONS)
    return ActionSpec(action=action, params={"degrees": degrees})


def sample_look_action(rng: random.Random) -> ActionSpec:
    action = rng.choice(LOOK_ACTIONS)
    degrees = rng.choice(LOOK_DEGREE_OPTIONS)
    return ActionSpec(action=action, params={"degrees": degrees})


def generate_random_sequence(rng: random.Random) -> Tuple[ActionSpec, ...]:
    length = 1 if rng.random() < 0.55 else MAX_SEQUENCE_LENGTH
    sequence: List[ActionSpec] = []
    for idx in range(length):
        category_pool = ["translate", "rotate", "look"]
        if idx == 0:
            category_pool = ["translate", "translate", "rotate", "look", "translate"]
        candidate: Optional[ActionSpec] = None
        attempts = 0
        while attempts < 8 and candidate is None:
            category = rng.choice(category_pool)
            if category == "translate":
                action_spec = sample_translation_action(rng)
                if sequence:
                    prev_action = sequence[-1].action
                    if prev_action in OPPOSITE_TRANSLATIONS and action_spec.action == OPPOSITE_TRANSLATIONS[prev_action]:
                        attempts += 1
                        continue
                candidate = action_spec
            elif category == "rotate":
                candidate = sample_rotation_action(rng)
            else:
                candidate = sample_look_action(rng)
            attempts += 1
        if candidate is None:
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


def attempt_actions(controller: Controller, actions: Sequence[ActionSpec]) -> bool:
    for spec in actions:
        try:
            controller.step(action=spec.action, raise_for_failure=True, **spec.params)
        except RuntimeError as exc:
            print(f"Action {spec.action} failed due to runtime error: {exc}")
            return False
        except Exception as exc:  # noqa: BLE001
            print(f"Action {spec.action} failed due to unexpected error: {exc}")
            return False
    return True


def extract_visible_objects(event) -> Dict[str, VisibleObject]:
    detections = event.instance_detections2D or {}
    result: Dict[str, VisibleObject] = {}
    for obj in event.metadata.get("objects", []):
        if not obj.get("visible"):
            continue
        obj_id = obj.get("objectId")
        if not obj_id:
            continue
        bbox = detections.get(obj_id)
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        result[obj_id] = VisibleObject(
            object_id=obj_id,
            object_type=obj.get("objectType", "Unknown"),
            name=obj.get("name", obj_id),
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            center=(float(center[0]), float(center[1])),
        )
    return result


def too_close(point: Tuple[float, float], others: Sequence[Tuple[float, float]]) -> bool:
    return any(math.hypot(point[0] - ox, point[1] - oy) < MIN_POINT_SPACING for ox, oy in others)


def random_screen_point(width: int, height: int, rng: random.Random, existing: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    padding = max(POINT_RADIUS_PIXELS * 2, 20)
    for _ in range(32):
        x = rng.uniform(padding, width - padding)
        y = rng.uniform(padding, height - padding)
        candidate = (x, y)
        if not too_close(candidate, existing):
            return candidate
    return (width / 2.0, height / 2.0)


def clamp_bbox(bbox: Tuple[float, float, float, float], width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (
        max(0.0, min(width, x1)),
        max(0.0, min(height, y1)),
        max(0.0, min(width, x2)),
        max(0.0, min(height, y2)),
    )


def normalize_bbox(bbox: Tuple[float, float, float, float], width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
    if width <= 0 or height <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (x1 / width, y1 / height, x2 / width, y2 / height)


def format_bbox(bbox: Tuple[float, float, float, float], width: int, height: int) -> str:
    nx1, ny1, nx2, ny2 = normalize_bbox(bbox, width, height)
    return f"[{nx1:.2f}, {ny1:.2f}, {nx2:.2f}, {ny2:.2f}]"


def select_common_object(
    before_event, after_event, rng: random.Random
) -> Optional[Tuple[VisibleObject, VisibleObject]]:
    before_objects = extract_visible_objects(before_event)
    after_objects = extract_visible_objects(after_event)
    common_ids = [obj_id for obj_id in before_objects if obj_id in after_objects]
    if not common_ids:
        return None
    target_id = rng.choice(common_ids)
    return before_objects[target_id], after_objects[target_id]


def build_point_choices(
    after_objects: Dict[str, VisibleObject],
    target_after: VisibleObject,
    width: int,
    height: int,
    rng: random.Random,
) -> Tuple[List[LabeledPoint], str]:
    letters = ["A", "B", "C", "D"]
    points: List[LabeledPoint] = []
    centers_so_far: List[Tuple[float, float]] = []

    points.append(
        LabeledPoint(
            label="",
            x=target_after.center[0],
            y=target_after.center[1],
            is_correct=True,
            source="object",
            object_id=target_after.object_id,
            object_type=target_after.object_type,
            bbox=target_after.bbox,
        )
    )
    centers_so_far.append(target_after.center)

    distractors: List[VisibleObject] = [
        obj for obj_id, obj in after_objects.items() if obj_id != target_after.object_id
    ]
    rng.shuffle(distractors)
    for obj in distractors[:DISTRACTOR_TARGET]:
        points.append(
            LabeledPoint(
                label="",
                x=obj.center[0],
                y=obj.center[1],
                is_correct=False,
                source="object",
                object_id=obj.object_id,
                object_type=obj.object_type,
                bbox=obj.bbox,
            )
        )
        centers_so_far.append(obj.center)

    while len(points) < DISTRACTOR_TARGET + 1:
        random_center = random_screen_point(width, height, rng, centers_so_far)
        radius = POINT_RADIUS_PIXELS * 1.5
        synthetic_bbox = (
            random_center[0] - radius,
            random_center[1] - radius,
            random_center[0] + radius,
            random_center[1] + radius,
        )
        points.append(
            LabeledPoint(
                label="",
                x=random_center[0],
                y=random_center[1],
                is_correct=False,
                source="random",
                object_id=None,
                object_type=None,
                bbox=synthetic_bbox,
            )
        )
        centers_so_far.append(random_center)

    rng.shuffle(points)
    labeled_points: List[LabeledPoint] = []
    correct_label = ""
    for idx, point in enumerate(points[: len(letters)]):
        label = letters[idx]
        point.label = label
        labeled_points.append(point)
        if point.is_correct:
            correct_label = label
    if not correct_label:
        raise RuntimeError("Failed to assign correct label to target point.")
    return labeled_points, correct_label


def draw_annotations(
    before_image: Image.Image,
    after_image: Image.Image,
    target_before: VisibleObject,
    labeled_points: Sequence[LabeledPoint],
) -> Tuple[Image.Image, Image.Image]:
    before_copy = before_image.copy().convert("RGB")
    after_copy = after_image.copy().convert("RGB")

    draw_before = ImageDraw.Draw(before_copy)
    draw_after = ImageDraw.Draw(after_copy)
    font = ImageFont.load_default()

    bx, by = target_before.center
    bx_i, by_i = int(round(bx)), int(round(by))
    draw_before.ellipse(
        [
            bx_i - POINT_RADIUS_PIXELS,
            by_i - POINT_RADIUS_PIXELS,
            bx_i + POINT_RADIUS_PIXELS,
            by_i + POINT_RADIUS_PIXELS,
        ],
        fill=(220, 30, 30),
        outline=(255, 255, 255),
        width=2,
    )

    for point in labeled_points:
        px_i, py_i = int(round(point.x)), int(round(point.y))
        draw_after.ellipse(
            [
                px_i - POINT_RADIUS_PIXELS,
                py_i - POINT_RADIUS_PIXELS,
                px_i + POINT_RADIUS_PIXELS,
                py_i + POINT_RADIUS_PIXELS,
            ],
            fill=(40, 120, 255) if point.is_correct else (20, 170, 255),
            outline=(255, 255, 255),
            width=2,
        )
        text_position = (px_i + POINT_RADIUS_PIXELS + 4, py_i - POINT_RADIUS_PIXELS - 4)
        draw_after.text(text_position, point.label, fill=(255, 255, 0), font=font)

    return before_copy, after_copy


def build_question(rng: random.Random) -> str:
    return rng.choice(QUESTION_TEMPLATES)


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
        "renderInstanceSegmentation": True,
    }
    if args.x_display is not None:
        controller_kwargs["x_display"] = args.x_display

    controller = Controller(**controller_kwargs)
    category_entries: Dict[str, List[dict]] = {"CorrPoint": [], "CorrCoord": [], "CorrObject": []}
    combined_entries: List[dict] = []

    try:
        failure_count = 0
        max_failures = args.num_samples * MAX_SAMPLE_ATTEMPTS
        while len(category_entries["CorrPoint"]) < args.num_samples:
            if failure_count >= max_failures:
                print("Stopping early: exceeded maximum failure attempts.")
                break

            randomize_start(controller, rng)
            before_event = controller.last_event
            if before_event is None:
                raise RuntimeError("Controller did not return an event after Teleport.")

            actions = sample_action_sequence(sequences, rng)
            if not attempt_actions(controller, actions):
                failure_count += 1
                continue
            after_event = controller.last_event
            if after_event is None:
                raise RuntimeError("Controller did not produce an event after actions.")

            selection = select_common_object(before_event, after_event, rng)
            if selection is None:
                failure_count += 1
                continue
            target_before, target_after = selection

            after_objects = extract_visible_objects(after_event)
            labeled_points, correct_label = build_point_choices(
                after_objects, target_after, args.width, args.height, rng
            )

            before_image = Image.fromarray(before_event.frame)
            after_image = Image.fromarray(after_event.frame)
            annotated_before, annotated_after = draw_annotations(
                before_image, after_image, target_before, labeled_points
            )

            sample_idx = len(category_entries["CorrPoint"])
            img1_path = args.output_dir / f"sample_{sample_idx:04d}_img1.png"
            img2_path = args.output_dir / f"sample_{sample_idx:04d}_img2.png"
            img1_path.parent.mkdir(parents=True, exist_ok=True)
            annotated_before.save(img1_path)
            annotated_after.save(img2_path)

            question_text = build_question(rng)
            target_bbox_before_pixels = clamp_bbox(target_before.bbox, args.width, args.height)
            target_bbox_after_pixels = clamp_bbox(target_after.bbox, args.width, args.height)

            points_metadata: List[Dict[str, Any]] = []
            for point in labeled_points:
                point_bbox_pixels = clamp_bbox(point.bbox, args.width, args.height) if point.bbox else clamp_bbox(
                    (
                        point.x - POINT_RADIUS_PIXELS,
                        point.y - POINT_RADIUS_PIXELS,
                        point.x + POINT_RADIUS_PIXELS,
                        point.y + POINT_RADIUS_PIXELS,
                    ),
                    args.width,
                    args.height,
                )
                point_bbox_normalized = normalize_bbox(point_bbox_pixels, args.width, args.height)
                points_metadata.append(
                    {
                        "label": point.label,
                        "x": point.x,
                        "y": point.y,
                        "is_correct": point.is_correct,
                        "source": point.source,
                        "object_id": point.object_id,
                        "object_type": point.object_type,
                        "bbox_pixels": point_bbox_pixels,
                        "bbox_normalized": point_bbox_normalized,
                    }
                )

            coord_options_strings: List[str] = []
            coord_answer_string = ""
            for point_info in points_metadata:
                option_string = f"{point_info['label']}. {format_bbox(point_info['bbox_pixels'], args.width, args.height)}"
                coord_options_strings.append(option_string)
                if point_info["is_correct"]:
                    coord_answer_string = option_string

            if not coord_answer_string and coord_options_strings:
                coord_answer_string = coord_options_strings[0]

            target_object_display = target_after.object_type or target_after.name or target_after.object_id
            target_object_name = target_after.name or target_object_display
            target_object_type = target_after.object_type or target_object_display

            object_option_candidates: List[str] = []
            for obj in after_objects.values():
                candidate_name = obj.object_type or obj.name or obj.object_id
                if not candidate_name:
                    continue
                candidate_name_str = str(candidate_name)
                if candidate_name_str not in object_option_candidates:
                    object_option_candidates.append(candidate_name_str)

            object_options_list: List[str] = []
            if target_object_type:
                object_options_list.append(str(target_object_type))
            for candidate in object_option_candidates:
                if len(object_options_list) >= 4:
                    break
                if candidate == target_object_type:
                    continue
                object_options_list.append(candidate)

            if len(object_options_list) < 4:
                for fallback in OBJECT_DISTRACTOR_POOL:
                    if len(object_options_list) >= 4:
                        break
                    if fallback == target_object_type or fallback in object_options_list:
                        continue
                    object_options_list.append(fallback)

            rng.shuffle(object_options_list)
            object_answer_string = str(target_object_type or "Unknown")
            if object_answer_string not in object_options_list:
                object_options_list = [object_answer_string] + [opt for opt in object_options_list if opt != object_answer_string]
                object_options_list = object_options_list[:4]
                rng.shuffle(object_options_list)

            object_options_list = object_options_list[:4]
            if not object_options_list:
                object_options_list = [object_answer_string or "Unknown"]

            metadata = {
                "target_object_id": target_after.object_id,
                "target_object_type": target_object_type,
                "target_object_display": target_object_display,
                "target_object_name": target_object_name,
                "target_point_before": {
                    "x": target_before.center[0],
                    "y": target_before.center[1],
                },
                "target_point_after": {
                    "x": target_after.center[0],
                    "y": target_after.center[1],
                },
                "target_bbox_before_pixels": target_bbox_before_pixels,
                "target_bbox_before_normalized": normalize_bbox(target_bbox_before_pixels, args.width, args.height),
                "target_bbox_after_pixels": target_bbox_after_pixels,
                "target_bbox_after_normalized": normalize_bbox(target_bbox_after_pixels, args.width, args.height),
                "points": points_metadata,
                "coord_options": coord_options_strings,
                "object_options": list(object_options_list),
            }

            shared_payload = {
                "image_1_path": str(img1_path.resolve()),
                "image_2_path": str(img2_path.resolve()),
                "actions": [
                    {"action": spec.action, "params": dict(spec.params)} for spec in actions
                ],
                "metadata": metadata,
            }

            corr_point_entry = {
                **shared_payload,
                "category": "CorrPoint",
                "question": question_text,
                "answer": correct_label,
                "options": [point.label for point in labeled_points],
            }
            category_entries["CorrPoint"].append(corr_point_entry)
            combined_entries.append(corr_point_entry)

            coord_question = rng.choice(COORD_TEMPLATES).format(
                bbox=format_bbox(target_bbox_before_pixels, args.width, args.height)
            )
            coord_entry = {
                **shared_payload,
                "category": "CorrCoord",
                "question": coord_question,
                "answer": coord_answer_string,
                "options": list(coord_options_strings),
            }
            category_entries["CorrCoord"].append(coord_entry)
            combined_entries.append(coord_entry)

            object_question = rng.choice(OBJECT_TEMPLATES)
            object_entry = {
                **shared_payload,
                "category": "CorrObject",
                "question": object_question,
                "answer": object_answer_string,
                "options": list(object_options_list),
            }
            category_entries["CorrObject"].append(object_entry)
            combined_entries.append(object_entry)

            print(
                f"Generated sample {sample_idx:04d} across CorrPoint/CorrCoord/CorrObject with target {target_object_display}."
            )

        for category, entries in category_entries.items():
            category_dir = args.qa_root / category
            category_dir.mkdir(parents=True, exist_ok=True)
            qa_path = category_dir / "qa_pairs.json"
            qa_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Saved {len(entries)} {category} QA pairs to {qa_path}")

        if args.qa_json is not None:
            args.qa_json.parent.mkdir(parents=True, exist_ok=True)
            args.qa_json.write_text(
                json.dumps(combined_entries, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"Saved combined QA log to {args.qa_json}")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
