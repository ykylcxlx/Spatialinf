"""Generate correspondence-point QA pairs using ProcTHOR scenes."""



# python generate_corr_point_qas.py houses/train_000.json --output-dir ../outputs/corr_point/images --qa-root ../outputs/corr_point --num-samples 20 --visibility-distance 10 --connect-timeout 300 --x-display :99 --qa-json ../outputs/corr_point/all_tasks.json
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
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
LABEL_FONT_SIZE = 28

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

CHOICE_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D")

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

# 某些对象类型（例如墙面）不适合作为正确答案，提前过滤掉
EXCLUDED_OBJECT_TYPES: Tuple[str, ...] = ("Wall",)


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
    parser.add_argument(
        "house_json",
        type=Path,
        help="Path to a ProcTHOR house JSON file or a directory containing multiple JSON files.",
    )
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
        default=512,
        help="Viewport width for rendered frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
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
    target.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )


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


def load_label_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def format_bbox(
    bbox: Optional[Tuple[float, float, float, float]],
    width: int,
    height: int,
) -> str:
    if bbox is None:
        return "[unknown]"
    x1, y1, x2, y2 = bbox
    denom_w = max(float(width) - 1.0, 1.0)
    denom_h = max(float(height) - 1.0, 1.0)
    normalized = [
        max(0.0, min(1.0, x1 / denom_w)),
        max(0.0, min(1.0, y1 / denom_h)),
        max(0.0, min(1.0, x2 / denom_w)),
        max(0.0, min(1.0, y2 / denom_h)),
    ]
    return "[" + ", ".join(f"{value:.2f}" for value in normalized) + "]"


def humanize_object_name(raw_name: Optional[str]) -> str:
    if not raw_name:
        return "Object"
    name = raw_name.replace("_", " ")
    name = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    words = [word.capitalize() for word in name.split()]
    return " ".join(words) if words else "Object"


def build_object_options(
    target_object_type: Optional[str],
    labeled_points: Sequence[LabeledPoint],
    rng: random.Random,
) -> Tuple[List[Tuple[str, str]], str]:
    correct_name = humanize_object_name(target_object_type)
    options: List[Tuple[str, bool]] = [(correct_name, True)]
    seen = {correct_name.lower()}

    for point in labeled_points:
        if point.is_correct:
            continue
        if point.object_type:
            candidate = humanize_object_name(point.object_type)
            lowered = candidate.lower()
            if lowered not in seen:
                options.append((candidate, False))
                seen.add(lowered)

    pool = list(OBJECT_DISTRACTOR_POOL)
    rng.shuffle(pool)
    for candidate in pool:
        candidate_name = humanize_object_name(candidate)
        lowered = candidate_name.lower()
        if lowered in seen:
            continue
        options.append((candidate_name, False))
        seen.add(lowered)
        if len(options) >= len(CHOICE_LETTERS):
            break

    while len(options) < len(CHOICE_LETTERS):
        filler = f"Item {len(options) + 1}"
        if filler.lower() in seen:
            filler = f"Choice {len(options) + 1}"
        options.append((filler, False))
        seen.add(options[-1][0].lower())

    rng.shuffle(options)

    try:
        correct_index = next(i for i, (_, flag) in enumerate(options) if flag)
    except StopIteration:
        correct_index = 0

    if correct_index >= len(CHOICE_LETTERS):
        options[0], options[correct_index] = options[correct_index], options[0]
        correct_index = 0

    trimmed = options[: len(CHOICE_LETTERS)]
    labeled_options: List[Tuple[str, str]] = []
    correct_label = ""
    for letter, (name, is_correct) in zip(CHOICE_LETTERS, trimmed):
        labeled_options.append((letter, name))
        if is_correct:
            correct_label = letter

    if not correct_label:
        labeled_options[0] = (CHOICE_LETTERS[0], correct_name)
        correct_label = CHOICE_LETTERS[0]

    return labeled_options, correct_label


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


def select_common_object(
    before_event, after_event, rng: random.Random
) -> Optional[Tuple[VisibleObject, VisibleObject]]:
    before_objects = extract_visible_objects(before_event)
    after_objects = extract_visible_objects(after_event)
    common_ids: List[str] = []
    for obj_id, before_obj in before_objects.items():
        after_obj = after_objects.get(obj_id)
        if after_obj is None:
            continue
        # 跳过指定类型（如墙面），避免生成墙作为正确答案
        if before_obj.object_type in EXCLUDED_OBJECT_TYPES or after_obj.object_type in EXCLUDED_OBJECT_TYPES:
            continue
        common_ids.append(obj_id)
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
    letters = list(CHOICE_LETTERS)
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
    font = load_label_font(LABEL_FONT_SIZE)

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
        text_position = (
            px_i + POINT_RADIUS_PIXELS + 6,
            py_i - (POINT_RADIUS_PIXELS + LABEL_FONT_SIZE // 2),
        )
        draw_after.text(text_position, point.label, fill=(255, 255, 0), font=font)

    return before_copy, after_copy


def build_question(rng: random.Random) -> str:
    return rng.choice(QUESTION_TEMPLATES)


def save_depth_image(depth_frame: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if depth_frame is None:
        return
    depth_m = np.nan_to_num(depth_frame, nan=0.0, posinf=0.0, neginf=0.0)
    depth_mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
    Image.fromarray(depth_mm).save(destination)


def generate_samples_for_house(
    house_path: Path,
    args: argparse.Namespace,
    sequences: Optional[Sequence[Tuple[ActionSpec, ...]]],
    rng: random.Random,
    start_index: int,
) -> Tuple[List[dict], int]:
    print(f"Processing house: {house_path}")
    house = load_house(house_path)

    controller_kwargs = {
        "scene": house,
        "width": args.width,
        "height": args.height,
        "connect_timeout": args.connect_timeout,
        "visibilityDistance": args.visibility_distance,
        "renderInstanceSegmentation": True,
        "renderDepthImage": True,
    }
    if args.x_display is not None:
        controller_kwargs["x_display"] = args.x_display

    controller = Controller(**controller_kwargs)
    qa_entries: List[dict] = []
    generated_samples = 0

    try:
        failure_count = 0
        max_failures = args.num_samples * MAX_SAMPLE_ATTEMPTS
        while generated_samples < args.num_samples:
            if failure_count >= max_failures:
                print(
                    f"Stopping early for {house_path.name}: exceeded maximum failure attempts ({max_failures})."
                )
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

            global_sample_idx = start_index + generated_samples
            sample_tag = f"sample_{global_sample_idx:06d}"
            img1_path = args.output_dir / f"{sample_tag}_img1.png"
            img2_path = args.output_dir / f"{sample_tag}_img2.png"
            img1_path.parent.mkdir(parents=True, exist_ok=True)
            annotated_before.save(img1_path)
            annotated_after.save(img2_path)

            depth1_path = args.output_dir / f"{sample_tag}_img1_depth.png"
            depth2_path = args.output_dir / f"{sample_tag}_img2_depth.png"
            save_depth_image(before_event.depth_frame, depth1_path)
            save_depth_image(after_event.depth_frame, depth2_path)

            base_dir = args.qa_json.parent if args.qa_json else args.output_dir
            rel_img1 = os.path.relpath(img1_path, base_dir)
            rel_img2 = os.path.relpath(img2_path, base_dir)
            rel_depth1 = os.path.relpath(depth1_path, base_dir)
            rel_depth2 = os.path.relpath(depth2_path, base_dir)
            images_rel = [rel_img1.replace("\\", "/"), rel_img2.replace("\\", "/")]
            depths_rel = [rel_depth1.replace("\\", "/"), rel_depth2.replace("\\", "/")]

            correct_point = next((point for point in labeled_points if point.is_correct), None)
            if correct_point is None:
                raise RuntimeError("No correct labeled point was generated.")
            id_prefix = f"corr_point_{global_sample_idx:06d}"
            prompt_prefix = "<image>Image 1.<image>Image 2."

            point_question = build_question(rng)
            point_options_text = " ".join(
                f"({point.label}) point {point.label}" for point in labeled_points
            )
            point_answer = f"({correct_label}) point {correct_label}"
            qa_entries.append(
                {
                    "id": f"{id_prefix}_point",
                    "images": images_rel,
                    "depths": depths_rel,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{prompt_prefix} {point_question} {point_options_text}",
                        },
                        {"role": "assistant", "content": point_answer},
                    ],
                }
            )

            coord_question = rng.choice(COORD_TEMPLATES).format(
                bbox=format_bbox(target_before.bbox, args.width, args.height)
            )
            coord_options_text = " ".join(
                f"({point.label}) bbox {format_bbox(point.bbox, args.width, args.height)}"
                for point in labeled_points
            )
            coord_answer = (
                f"({correct_label}) bbox {format_bbox(correct_point.bbox, args.width, args.height)}"
            )
            qa_entries.append(
                {
                    "id": f"{id_prefix}_bbox",
                    "images": images_rel,
                    "depths": depths_rel,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{prompt_prefix} {coord_question} {coord_options_text}",
                        },
                        {"role": "assistant", "content": coord_answer},
                    ],
                }
            )

            object_options, object_correct_label = build_object_options(
                target_after.object_type, labeled_points, rng
            )
            object_question = rng.choice(OBJECT_TEMPLATES)
            object_options_text = " ".join(
                f"({letter}) {name}" for letter, name in object_options
            )
            default_object_name = humanize_object_name(target_after.object_type)
            object_answer_name = next(
                (name for letter, name in object_options if letter == object_correct_label),
                default_object_name,
            )
            object_answer = f"({object_correct_label}) {object_answer_name}"
            qa_entries.append(
                {
                    "id": f"{id_prefix}_object",
                    "images": images_rel,
                    "depths": depths_rel,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{prompt_prefix} {object_question} {object_options_text}",
                        },
                        {"role": "assistant", "content": object_answer},
                    ],
                }
            )

            print(
                "House {house} sample {idx:04d}: point={point_label}, bbox={bbox_label}, object={object_label} ({object_name}).".format(
                    house=house_path.stem,
                    idx=generated_samples,
                    point_label=correct_label,
                    bbox_label=correct_label,
                    object_label=object_correct_label,
                    object_name=object_answer_name,
                )
            )

            generated_samples += 1

    finally:
        controller.stop()

    if generated_samples < args.num_samples:
        print(
            f"Warning: only generated {generated_samples} / {args.num_samples} sample sets for house {house_path.name}."
        )

    return qa_entries, generated_samples


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    rng = random.Random(args.seed)

    sequences = load_sequences(args.sequences)
    house_paths = collect_house_paths(args.house_json)

    qa_entries: List[dict] = []
    total_samples = 0
    output_json = args.qa_json or (args.output_dir / "qa_pairs.json")

    for house_path in house_paths:
        try:
            house_entries, produced = generate_samples_for_house(
                house_path, args, sequences, rng, total_samples
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to generate samples for {house_path}: {exc}")
            continue

        qa_entries.extend(house_entries)
        total_samples += produced
        write_combined_entries(qa_entries, output_json)
        print(f"Appended {len(house_entries)} entries -> {output_json}")

    print(f"Saved total {len(qa_entries)} QA pairs to {output_json}")


if __name__ == "__main__":
    main()
