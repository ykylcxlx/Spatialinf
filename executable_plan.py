import argparse
import math
import re
import shutil
import subprocess
import time
import threading
import cv2
import numpy as np
from ai2thor.controller import Controller
from scipy.spatial import distance
from typing import Any, Dict, List, Optional, Sequence, Tuple
import random
import os
import copy
import json
from pathlib import Path

from typing import Set


import prior


def _resolve_split(default: str = "train") -> str:
    for key in ("PROC_THOR_SPLIT", "PROCTHOR_SPLIT"):
        value = os.environ.get(key)
        if value:
            return value
    return default


def _resolve_house_index(default: int = 0) -> int:
    for key in ("PROC_THOR_HOUSE_INDEX", "PROCTHOR_HOUSE_INDEX"):
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return max(0, int(value))
        except ValueError:
            print(f"[WARN] Invalid {key}='{value}'; falling back to {default}")
            break
    return default


PROC_THOR_SPLIT = _resolve_split()
PROC_THOR_HOUSE_INDEX = _resolve_house_index()
_DATASET = prior.load_dataset("procthor-10k")
try:
    _SPLIT_SCENES = _DATASET[PROC_THOR_SPLIT]
except Exception as exc:
    raise KeyError(f"Split '{PROC_THOR_SPLIT}' not found in ProcTHOR dataset") from exc
if not 0 <= PROC_THOR_HOUSE_INDEX < len(_SPLIT_SCENES):
    raise IndexError(
        f"House index {PROC_THOR_HOUSE_INDEX} is out of range for split '{PROC_THOR_SPLIT}'"
    )
_PROC_SCENE = _SPLIT_SCENES[PROC_THOR_HOUSE_INDEX]
_SCENE_ID = str(_PROC_SCENE.get("id", f"{PROC_THOR_SPLIT}_{PROC_THOR_HOUSE_INDEX}"))


def _compute_polygon_area(points: Sequence[Dict[str, Any]]) -> float:
    if not points or len(points) < 3:
        return 0.0
    coords: List[Tuple[float, float]] = []
    for entry in points:
        try:
            x_val = float(entry.get("x", 0.0))
            z_val = float(entry.get("z", 0.0))
        except (TypeError, ValueError):
            continue
        coords.append((x_val, z_val))
    if len(coords) < 3:
        return 0.0
    area = 0.0
    for idx, (x0, z0) in enumerate(coords):
        x1, z1 = coords[(idx + 1) % len(coords)]
        area += (x0 * z1) - (x1 * z0)
    return abs(area) * 0.5


def _build_room_lookup(scene_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for index, room in enumerate(scene_payload.get("rooms", []) or []):
        raw_id = room.get("id")
        if isinstance(raw_id, str) and raw_id.strip():
            room_id = raw_id.strip()
        else:
            room_id = f"room|{index}"
        floor_polygon = room.get("floorPolygon") or []
        lookup[room_id] = {
            "id": room_id,
            "roomType": room.get("roomType"),
            "floorMaterial": room.get("floorMaterial"),
            "area": _compute_polygon_area(floor_polygon),
        }
    return lookup


def _normalize_room_id(room_identifier: Optional[str]) -> Optional[str]:
    if not room_identifier:
        return None
    if not isinstance(room_identifier, str):
        return None
    trimmed = room_identifier.strip()
    if not trimmed:
        return None
    if "|" not in trimmed:
        return trimmed
    parts = trimmed.split("|")
    if len(parts) >= 2 and parts[0].lower() == "room":
        return "|".join(parts[:2])
    return trimmed


_ROOM_LOOKUP: Dict[str, Dict[str, Any]] = _build_room_lookup(_PROC_SCENE)
_TOTAL_ROOMS = len(_ROOM_LOOKUP)

_ROOM_VISIT_LOCK = threading.Lock()
_ROOM_VISITED_SET: Set[str] = set()
_ROOM_VISIT_ORDER: List[str] = []
_ROOM_TIMELINE: List[Dict[str, Any]] = []
_AGENT_ROOM_SEQUENCES: Dict[int, List[str]] = {}


def _reset_room_tracking() -> None:
    global _ROOM_VISITED_SET, _ROOM_VISIT_ORDER, _ROOM_TIMELINE, _AGENT_ROOM_SEQUENCES
    with _ROOM_VISIT_LOCK:
        _ROOM_VISITED_SET = set()
        _ROOM_VISIT_ORDER = []
        _ROOM_TIMELINE = []
        _AGENT_ROOM_SEQUENCES = {}


def _record_room_visit(
    agent_index: int,
    agent_name: str,
    raw_room_id: Optional[str],
    raw_room_type: Optional[str],
    frame_index: int,
) -> None:
    normalized_id = _normalize_room_id(raw_room_id)
    resolved_id = normalized_id or (raw_room_id.strip() if isinstance(raw_room_id, str) else None)
    room_info = _ROOM_LOOKUP.get(resolved_id) if resolved_id else None
    resolved_type = (
        raw_room_type
        or (room_info.get("roomType") if isinstance(room_info, dict) else None)
    )
    if not resolved_id and not resolved_type:
        return
    with _ROOM_VISIT_LOCK:
        if resolved_id:
            if resolved_id not in _ROOM_VISITED_SET:
                _ROOM_VISITED_SET.add(resolved_id)
                _ROOM_VISIT_ORDER.append(resolved_id)
        timeline_entry: Dict[str, Any] = {
            "frame_index": frame_index,
            "agent_index": agent_index,
            "agent_name": agent_name,
        }
        if resolved_id:
            timeline_entry["room_id"] = resolved_id
        if resolved_type:
            timeline_entry["room_type"] = resolved_type
        if room_info:
            timeline_entry["room_area"] = room_info.get("area")
        _ROOM_TIMELINE.append(timeline_entry)
        if resolved_id:
            seq = _AGENT_ROOM_SEQUENCES.setdefault(agent_index, [])
            if not seq or seq[-1] != resolved_id:
                seq.append(resolved_id)


# floor_no = 15

def _extract_scene_object_basenames(scene_payload: dict):
    names = set()
    for entry in (scene_payload or {}).get("objects", []):
        obj_id = entry.get("id", "")
        if obj_id:
            names.add(obj_id.split("|")[0])
        for child in entry.get("children", []) or []:
            child_id = child.get("id", "")
            if child_id:
                names.add(child_id.split("|")[0])
    return sorted(names)


def _choose_random_object_name(scene_payload: dict, excluded=None) -> str:
    excluded = set(excluded or [])
    available = [
        name for name in _extract_scene_object_basenames(scene_payload)
        if name not in excluded
    ]
    if not available:
        raise RuntimeError("No eligible objects found for navigation target selection")
    return random.choice(available)


def _sanitize_label(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", (value or "unknown"))


def _visible_object_ids(event):
    seg_frame = getattr(event, "instance_segmentation_frame", None)
    seg_objects = getattr(event, "instance_segmentation_objects", None)
    if seg_frame is not None and seg_objects:
        try:
            color_map = {
                tuple(int(c) for c in obj.get("color", ())): obj.get("objectId")
                for obj in seg_objects
            }
            flat = seg_frame.reshape(-1, seg_frame.shape[-1])
            unique_colors = np.unique(flat, axis=0)
            visible_ids = []
            for color in unique_colors:
                color_key = tuple(int(c) for c in color)
                if color_key == (0, 0, 0):
                    continue
                obj_id = color_map.get(color_key)
                if obj_id:
                    visible_ids.append(obj_id)
            if visible_ids:
                return sorted(set(visible_ids))
        except Exception:
            pass
    metadata = getattr(event, "metadata", {})
    objects = (metadata or {}).get("objects", [])
    return [obj.get("objectId") for obj in objects if obj.get("visible")]


def closest_node(node, nodes, no_robot, clost_node_location):
    crps = []
    distances = distance.cdist([node], nodes)[0]
    dist_indices = np.argsort(np.array(distances))
    for i in range(no_robot):
        pos_index = dist_indices[(i * 5) + clost_node_location[i]]
        crps.append (nodes[pos_index])
    return crps

def distance_pts(p1: Tuple[float, float, float], p2: Tuple[float, float, float]):
    return ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5


def _select_navigation_target(
    controller: Controller,
    exclude: Sequence[str],
    min_distance: float = 4.5,
) -> Tuple[str, str]:
    event = controller.last_event
    objects = event.metadata.get("objects", [])
    agent_positions: List[Tuple[float, float, float]] = []
    agent_room_types: List[str] = []
    for agent_event in getattr(event, "events", []) or []:
        agent_meta = agent_event.metadata.get("agent", {})
        position = agent_meta.get("position", {})
        agent_positions.append(
            (
                float(position.get("x", 0.0)),
                float(position.get("y", 0.0)),
                float(position.get("z", 0.0)),
            )
        )
        room_type = agent_meta.get("room") or agent_meta.get("roomType")
        if isinstance(room_type, str):
            agent_room_types.append(room_type)

    excluded = set(exclude)
    candidates: List[Tuple[str, str, float]] = []
    for obj in objects:
        object_id = obj.get("objectId")
        if not object_id:
            continue
        base_name = object_id.split("|")[0]
        if base_name in excluded:
            continue
        aabb = obj.get("axisAlignedBoundingBox", {})
        center = aabb.get("center") if isinstance(aabb, dict) else None
        if not center:
            continue
        center_pos = (
            float(center.get("x", 0.0)),
            float(center.get("y", 0.0)),
            float(center.get("z", 0.0)),
        )
        room_type = obj.get("roomType") or obj.get("room")
        if agent_room_types and isinstance(room_type, str) and room_type in agent_room_types:
            continue
        if agent_positions:
            min_dist = min(distance_pts(agent_pos, center_pos) for agent_pos in agent_positions)
            if min_dist < min_distance:
                continue
        else:
            min_dist = 0.0
        candidates.append((object_id, base_name, min_dist))

    if candidates:
        max_dist = max(candidate[2] for candidate in candidates)
        far_candidates = [candidate for candidate in candidates if max_dist - candidate[2] < 0.5]
        chosen_id, chosen_name, _ = random.choice(far_candidates or candidates)
        return chosen_id, chosen_name

    # Fallback to previous behaviour if no candidate satisfies the distance constraint.
    fallback_name = _choose_random_object_name(_PROC_SCENE, excluded=excluded)
    for obj in objects:
        object_id = obj.get("objectId")
        if object_id and object_id.startswith(f"{fallback_name}|"):
            return object_id, fallback_name
    return fallback_name, fallback_name

def generate_video(frame_rate: int, destinations: Optional[Dict[str, Path]] = None) -> Dict[str, Path]:
    produced: Dict[str, Path] = {}
    if not _RUN_OUTPUT_ROOT.exists():
        print(f"The output directory {_RUN_OUTPUT_ROOT} does not exist; skipping video generation.")
        return produced
    for imgs_folder in sorted(_RUN_OUTPUT_ROOT.iterdir()):
        if not imgs_folder.is_dir():
            continue
        if not any(imgs_folder.glob("img_*.png")):
            continue
        view = imgs_folder.name
        output_path = _RUN_OUTPUT_ROOT / f"video_{view}.mp4"
        command_set = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(frame_rate),
            "-i",
            str(imgs_folder / "img_%05d.png"),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        subprocess.call(command_set)
        dest_key = None
        if view.startswith("agent_"):
            dest_key = "agent"
        elif view.startswith("top_view"):
            dest_key = "top_view"

        final_path = output_path
        if destinations and dest_key and dest_key in destinations:
            target_path = destinations[dest_key]
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path != output_path:
                shutil.copy2(output_path, target_path)
            final_path = target_path
        produced[dest_key or view] = final_path
    return produced
        



objects = [{'name': 'Apple', 'mass': 0.20000000298023224}, {'name': 'Blinds', 'mass': 0.0}, {'name': 'Bowl', 'mass': 0.4699999988079071}, {'name': 'Bread', 'mass': 0.699999988079071}, {'name': 'ButterKnife', 'mass': 0.07999999821186066}, {'name': 'Cabinet', 'mass': 0.0}, {'name': 'Cabinet', 'mass': 0.0}, {'name': 'Cabinet', 'mass': 0.0}, {'name': 'Cabinet', 'mass': 0.0}, {'name': 'Cabinet', 'mass': 0.0}, {'name': 'Cabinet', 'mass': 0.0}, {'name': 'Cabinet', 'mass': 0.0}, {'name': 'Cabinet', 'mass': 0.0}, {'name': 'Cabinet', 'mass': 0.0}, {'name': 'Cabinet', 'mass': 0.0}, {'name': 'Cabinet', 'mass': 0.0}, {'name': 'CoffeeMachine', 'mass': 5.0}, {'name': 'CounterTop', 'mass': 0.0}, {'name': 'CounterTop', 'mass': 0.0}, {'name': 'Cup', 'mass': 0.4000000059604645}, {'name': 'DiningTable', 'mass': 85.0}, {'name': 'DishSponge', 'mass': 0.029999997466802597}, {'name': 'Drawer', 'mass': 0.0}, {'name': 'Drawer', 'mass': 0.0}, {'name': 'Drawer', 'mass': 0.0}, {'name': 'Egg', 'mass': 0.054999999701976776}, {'name': 'Faucet', 'mass': 0.0}, {'name': 'Floor', 'mass': 1.0}, {'name': 'Fork', 'mass': 0.03999999910593033}, {'name': 'Fridge', 'mass': 0.0}, {'name': 'GarbageBag', 'mass': 3.499999761581421}, {'name': 'GarbageCan', 'mass': 0.699999988079071}, {'name': 'Kettle', 'mass': 0.800000011920929}, {'name': 'Knife', 'mass': 0.18000000715255737}, {'name': 'Lettuce', 'mass': 0.4699999988079071}, {'name': 'LightSwitch', 'mass': 0.0}, {'name': 'Microwave', 'mass': 6.999999523162842}, {'name': 'Mug', 'mass': 1.0}, {'name': 'Pan', 'mass': 0.6700000166893005}, {'name': 'PepperShaker', 'mass': 0.14000000059604645}, {'name': 'Plate', 'mass': 0.6200000047683716}, {'name': 'Pot', 'mass': 0.5699999928474426}, {'name': 'Potato', 'mass': 0.18000000715255737}, {'name': 'SaltShaker', 'mass': 0.4000000059604645}, {'name': 'Sink', 'mass': 0.0}, {'name': 'SinkBasin', 'mass': 0.0}, {'name': 'SoapBottle', 'mass': 0.4000000059604645}, {'name': 'Spatula', 'mass': 0.06499999761581421}, {'name': 'Spoon', 'mass': 0.03999999910593033}, {'name': 'Stool', 'mass': 3.180000066757202}, {'name': 'StoveBurner', 'mass': 0.0}, {'name': 'StoveBurner', 'mass': 0.0}, {'name': 'StoveBurner', 'mass': 0.0}, {'name': 'StoveBurner', 'mass': 0.0}, {'name': 'StoveKnob', 'mass': 0.0}, {'name': 'StoveKnob', 'mass': 0.0}, {'name': 'StoveKnob', 'mass': 0.0}, {'name': 'StoveKnob', 'mass': 0.0}, {'name': 'Toaster', 'mass': 5.0}, {'name': 'Tomato', 'mass': 0.11999998986721039}, {'name': 'Window', 'mass': 0.0}, {'name': 'WineBottle', 'mass': 1.2000000476837158}]
# floor_no = 15

ground_truth = [{'name': 'CoffeeMachine', 'contains': ['Mug'], 'state': 'None'}, {'name': 'Mug', 'contains': [], 'state': 'HOT'}]
no_trans_gt = 1
max_trans = 1

# ---- single-agent configuration (auto-injected) ----
robots = [
    {
        'name': 'robot1',
        'skills': [
            'GoToObject', 'OpenObject', 'CloseObject', 'PickupObject',
            'PutObject', 'DropHandObject', 'ThrowObject', 'PushObject', 'PullObject'
        ],
        'mass': 100
    }
]
no_robot = 1
# ---- end single-agent configuration ----


try:
    _PLAN_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _PLAN_DIR = os.getcwd()

DEFAULT_OUTPUT_BASE = Path(
    os.environ.get(
        "PROCTHOR_VIDEO_OUTPUT",
        Path(__file__).resolve().parents[1] / "outputs" / "video",
    )
)

_RANDOM_NAV_TARGET: Optional[str] = None
_RANDOM_NAV_TARGET_ID: Optional[str] = None
_RUN_OUTPUT_ROOT: Optional[Path] = None
_RUN_INFO: Dict[str, Any] = {}
_RUN_SUFFIX: Optional[str] = None


_DISPLAY_CANDIDATES = []
_env_display = os.environ.get('DISPLAY')
if _env_display:
    _DISPLAY_CANDIDATES.append(_env_display)
_DISPLAY_CANDIDATES.extend([':99.0', ':1.0'])

def _create_controller(**kwargs):
    errors = []
    defaults = {
        "renderInstanceSegmentation": True,
    }
    default_kwargs = {**defaults, **kwargs}
    for disp in _DISPLAY_CANDIDATES:
        if not disp:
            continue
        try:
            os.environ['DISPLAY'] = disp
            ctrl = Controller(x_display=disp, **default_kwargs)
            print(f"[INFO] AI2-THOR using DISPLAY {disp}")
            return ctrl
        except Exception as exc:
            errors.append(f"{disp}: {exc}")
    raise RuntimeError("Failed to initialize Controller. Tried displays:\n" + "\n".join(errors))



total_exec = 0
success_exec = 0

c: Optional[Controller] = None
no_robot = len(robots)
reachable_positions: List[Tuple[float, float, float]] = []
action_queue: List[Dict[str, Any]] = []
task_over = False
recp_id: Optional[str] = None
_HAS_TOP_VIEW = False

def exec_actions():
    global total_exec, success_exec
    if _RUN_OUTPUT_ROOT is None:
        raise RuntimeError("_RUN_OUTPUT_ROOT is not set before exec_actions start")
    if _RUN_SUFFIX is None:
        raise RuntimeError("_RUN_SUFFIX is not set before exec_actions start")
    run_dir = _RUN_OUTPUT_ROOT
    run_dir.mkdir(parents=True, exist_ok=True)

    agent_dirs = []
    for i in range(no_robot):
        folder_path = run_dir / f"agent_{i + 1}_{_RUN_SUFFIX}"
        if folder_path.exists():
            shutil.rmtree(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        agent_dirs.append(folder_path)

    top_view_dir = run_dir / f"top_view_{_RUN_SUFFIX}"
    if top_view_dir.exists():
        shutil.rmtree(top_view_dir)
    top_view_dir.mkdir(parents=True, exist_ok=True)

    info_path = run_dir / "run_info.json"
    try:
        info_path.write_text(json.dumps(_RUN_INFO, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as info_err:
        print(f"[WARN] Failed to write run info: {info_err}")

    img_counter = 0
    
    while not task_over:
        if len(action_queue) > 0:
            try:
                act = action_queue[0]
                if act['action'] == 'ObjectNavExpertAction':
                    multi_agent_event = c.step(dict(action=act['action'], position=act['position'], agentId=act['agent_id']))
                    next_action = multi_agent_event.metadata['actionReturn']

                    if next_action != None:
                        multi_agent_event = c.step(action=next_action, agentId=act['agent_id'], forceAction=True)
                
                elif act['action'] == 'MoveAhead':
                    multi_agent_event = c.step(action="MoveAhead", agentId=act['agent_id'])
                    
                elif act['action'] == 'MoveBack':
                    multi_agent_event = c.step(action="MoveBack", agentId=act['agent_id'])
                        
                elif act['action'] == 'RotateLeft':
                    multi_agent_event = c.step(action="RotateLeft", degrees=act['degrees'], agentId=act['agent_id'])
                    
                elif act['action'] == 'RotateRight':
                    multi_agent_event = c.step(action="RotateRight", degrees=act['degrees'], agentId=act['agent_id'])
                    
                elif act['action'] == 'PickupObject':
                    total_exec += 1
                    multi_agent_event = c.step(action="PickupObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                    else:
                        success_exec += 1
 
                elif act['action'] == 'PutObject':
                    total_exec += 1
                    multi_agent_event = c.step(action="PutObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                    else:
                        success_exec += 1
 
                elif act['action'] == 'ToggleObjectOn':
                    total_exec += 1
                    multi_agent_event = c.step(action="ToggleObjectOn", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                    else:
                        success_exec += 1
                
                elif act['action'] == 'ToggleObjectOff':
                    total_exec += 1
                    multi_agent_event = c.step(action="ToggleObjectOff", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                    else:
                        success_exec += 1
                    
                elif act['action'] == 'OpenObject':
                    total_exec += 1
                    multi_agent_event = c.step(action="OpenObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                    else:
                        success_exec += 1
 
                    
                elif act['action'] == 'CloseObject':
                    total_exec += 1
                    multi_agent_event = c.step(action="CloseObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                    else:
                        success_exec += 1
                        
                elif act['action'] == 'SliceObject':
                    total_exec += 1
                    multi_agent_event = c.step(action="SliceObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                    else:
                        success_exec += 1
                        
                elif act['action'] == 'ThrowObject':
                    total_exec += 1
                    multi_agent_event = c.step(action="ThrowObject", moveMagnitude=7, agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                    else:
                        success_exec += 1
                        
                elif act['action'] == 'BreakObject':
                    total_exec += 1
                    multi_agent_event = c.step(action="BreakObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
                    if multi_agent_event.metadata['errorMessage'] != "":
                        print (multi_agent_event.metadata['errorMessage'])
                    else:
                        success_exec += 1
 
                
                elif act['action'] == 'Done':
                    multi_agent_event = c.step(action="Done")
                    
                    
            except Exception as e:
                print (e)
                
            for i, e in enumerate(multi_agent_event.events):
                cv2.imshow('agent%s' % i, e.cv2img)
                frame_basename = f"img_{img_counter:05d}"
                frame_root = agent_dirs[i] / frame_basename
                cv2.imwrite(str(frame_root.with_suffix(".png")), e.cv2img)
                visible_ids = _visible_object_ids(e)
                # attempt to read agent room info from event metadata
                agent_meta = (getattr(e, "metadata", {}) or {}).get("agent", {})
                raw_room_id = agent_meta.get("room") or agent_meta.get("roomType")
                room_id = _normalize_room_id(raw_room_id)
                room_info = _ROOM_LOOKUP.get(room_id) if room_id else None
                room_type = agent_meta.get("roomType") or (room_info.get("roomType") if room_info else None)
                room_area = room_info.get("area") if room_info else None
                # record visit into module-level trackers
                try:
                    _record_room_visit(i, robots[i]["name"], raw_room_id, room_type, img_counter)
                except Exception:
                    pass
                metadata = {
                    "agent_id": i,
                    "agent_name": robots[i]["name"],
                    "frame_index": img_counter,
                    "itemlist": visible_ids,
                    "house_id": _SCENE_ID,
                    "nav_target": _RANDOM_NAV_TARGET,
                }
                if room_id:
                    metadata["room_id"] = room_id
                if room_type:
                    metadata["room_type"] = room_type
                if room_area is not None:
                    metadata["room_area"] = room_area
                with open(frame_root.with_suffix(".json"), "w", encoding="utf-8") as meta_file:
                    json.dump(metadata, meta_file, ensure_ascii=False, indent=2)
            frames = []
            if _HAS_TOP_VIEW:
                try:
                    frames = c.last_event.events[0].third_party_camera_frames
                except (AttributeError, IndexError, TypeError):
                    frames = []
            if frames:
                top_view_rgb = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB)
                cv2.imshow('Top View', top_view_rgb)
                f_name = top_view_dir / f"img_{img_counter:05d}.png"
                cv2.imwrite(str(f_name), top_view_rgb)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
            img_counter += 1    
            action_queue.pop(0)
       
actions_thread = threading.Thread(target=exec_actions)
actions_thread.start()

def GoToObject(robots, dest_obj, max_iterations: int = 400):
    global recp_id
    
    # check if robots is a list
    
    if not isinstance(robots, list):
        # convert robot to a list
        robots = [robots]
    no_agents = len (robots)
    # robots distance to the goal 
    dist_goals = [10.0] * len(robots)
    prev_dist_goals = [10.0] * len(robots)
    count_since_update = [0] * len(robots)
    clost_node_location = [0] * len(robots)
    
    # list of objects in the scene and their centers
    objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])
    objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])
    dest_obj_center = None
    if "|" in dest_obj:
        # full object id supplied; attempt to use metadata center
        dest_obj_id = dest_obj
        for idx, obj in enumerate(objs):
            if obj == dest_obj_id:
                candidate_center = objs_center[idx]
                if candidate_center and candidate_center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                    dest_obj_center = candidate_center
                break
        if dest_obj_center is None:
            pos_components = dest_obj_id.split("|")[1:]
            coords: List[float] = []
            for comp in pos_components:
                try:
                    coords.append(float(comp))
                except (TypeError, ValueError):
                    continue
                if len(coords) == 3:
                    break
            if len(coords) == 3:
                dest_obj_center = {'x': coords[0], 'y': coords[1], 'z': coords[2]}
    else:
        for idx, obj in enumerate(objs):
            
            match = re.match(dest_obj, obj)
            if match is not None:
                dest_obj_id = obj
                dest_obj_center = objs_center[idx]
                if dest_obj_center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                    break # find the first instance

    if dest_obj_center is None:
        raise RuntimeError(f"Unable to determine center for object '{dest_obj}'.")
        
    print ("Going to ", dest_obj_id, dest_obj_center)
        
    dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']] 
    
    # closest reachable position for each robot
    # all robots cannot reach the same spot 
    # differt close points needs to be found for each robot
    crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
    goal_thresh = 1.0
    # at least one robot is far away from the goal
    
    iteration = 0
    while all(d > goal_thresh for d in dist_goals):
        iteration += 1
        if iteration > max_iterations:
            print("[WARN] Navigation loop exceeded max_iterations; stopping early.")
            break
        for ia, robot in enumerate(robots):
            robot_name = robot['name']
            agent_id = int(robot_name[-1]) - 1
            
            # get the pose of robot        
            metadata = c.last_event.events[agent_id].metadata
            location = {
                "x": metadata["agent"]["position"]["x"],
                "y": metadata["agent"]["position"]["y"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"],
                "horizon": metadata["agent"]["cameraHorizon"]}
            
            prev_dist_goals[ia] = dist_goals[ia] # store the previous distance to goal
            dist_goals[ia] = distance_pts([location['x'], location['y'], location['z']], crp[ia])
            
            dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])
            # print (ia, "Dist to Goal: ", dist_goals[ia], dist_del, clost_node_location[ia])
            if dist_del < 0.2:
                # robot did not move 
                count_since_update[ia] += 1
            else:
                # robot moving 
                count_since_update[ia] = 0
                
            if count_since_update[ia] < 8:
                action_queue.append({'action':'ObjectNavExpertAction', 'position':dict(x=crp[ia][0], y=crp[ia][1], z=crp[ia][2]), 'agent_id':agent_id})
            else:    
                #updating goal
                clost_node_location[ia] += 1
                count_since_update[ia] = 0
                crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
            time.sleep(0.5)

    # align the robot once goal is reached
    # compute angle between robot heading and object
    metadata = c.last_event.events[agent_id].metadata
    robot_location = {
        "x": metadata["agent"]["position"]["x"],
        "y": metadata["agent"]["position"]["y"],
        "z": metadata["agent"]["position"]["z"],
        "rotation": metadata["agent"]["rotation"]["y"],
        "horizon": metadata["agent"]["cameraHorizon"]}
    
    robot_object_vec = [dest_obj_pos[0] -robot_location['x'], dest_obj_pos[2]-robot_location['z']]
    y_axis = [0, 1]
    unit_y = y_axis / np.linalg.norm(y_axis)
    unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)
    
    angle = math.atan2(np.linalg.det([unit_vector,unit_y]),np.dot(unit_vector,unit_y))
    angle = 360*angle/(2*np.pi)
    angle = (angle + 360) % 360
    rot_angle = angle - robot_location['rotation']
    
    if rot_angle > 0:
        action_queue.append({'action':'RotateRight', 'degrees':abs(rot_angle), 'agent_id':agent_id})
    else:
        action_queue.append({'action':'RotateLeft', 'degrees':abs(rot_angle), 'agent_id':agent_id})
        
    print ("Reached: ", dest_obj)
    if dest_obj == "Cabinet" or dest_obj == "Fridge" or dest_obj == "CounterTop":
        recp_id = dest_obj_id
    
def PickupObject(robots, pick_obj):
    if not isinstance(robots, list):
        # convert robot to a list
        robots = [robots]
    no_agents = len (robots)
    # robots distance to the goal 
    for idx in range(no_agents):
        robot = robots[idx]
        print ("PIcking: ", pick_obj)
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        # list of objects in the scene and their centers
        objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])
        objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])
        
        for idx, obj in enumerate(objs):
            match = re.match(pick_obj, obj)
            if match is not None:
                pick_obj_id = obj
                dest_obj_center = objs_center[idx]
                if dest_obj_center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                    break # find the first instance
        # GoToObject(robot, pick_obj_id)
        # time.sleep(1)
        print ("Picking Up ", pick_obj_id, dest_obj_center)
        action_queue.append({'action':'PickupObject', 'objectId':pick_obj_id, 'agent_id':agent_id})
        time.sleep(1)
    
def PutObject(robot, put_obj, recp):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])
    objs_dists = list([obj["distance"] for obj in c.last_event.metadata["objects"]])

    metadata = c.last_event.events[agent_id].metadata
    robot_location = [metadata["agent"]["position"]["x"], metadata["agent"]["position"]["y"], metadata["agent"]["position"]["z"]]
    dist_to_recp = 9999999 # distance b/w robot and the recp obj
    for idx, obj in enumerate(objs):
        match = re.match(recp, obj)
        if match is not None:
            dist = objs_dists[idx]
            if dist < dist_to_recp:
                recp_obj_id = obj
                dest_obj_center = objs_center[idx]
                dist_to_recp = dist
                
    
    global recp_id         
    # if recp_id is not None:
    #     recp_obj_id = recp_id
    # GoToObject(robot, recp_obj_id)
    # time.sleep(1)
    action_queue.append({'action':'PutObject', 'objectId':recp_obj_id, 'agent_id':agent_id})
    time.sleep(1)
         
def SwitchOn(robot, sw_obj):
    print ("Switching On: ", sw_obj)
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    # turn on all stove burner
    if sw_obj == "StoveKnob":
        for obj in objs:
            match = re.match(sw_obj, obj)
            if match is not None:
                sw_obj_id = obj
                GoToObject(robot, sw_obj_id)
                # time.sleep(1)
                action_queue.append({'action':'ToggleObjectOn', 'objectId':sw_obj_id, 'agent_id':agent_id})
                time.sleep(0.1)
    
    # all objects apart from Stove Burner
    else:
        for obj in objs:
            match = re.match(sw_obj, obj)
            if match is not None:
                sw_obj_id = obj
                break # find the first instance
        GoToObject(robot, sw_obj_id)
        time.sleep(1)
        action_queue.append({'action':'ToggleObjectOn', 'objectId':sw_obj_id, 'agent_id':agent_id})
        time.sleep(1)            
        
def SwitchOff(robot, sw_obj):
    print ("Switching Off: ", sw_obj)
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    # turn on all stove burner
    if sw_obj == "StoveKnob":
        for obj in objs:
            match = re.match(sw_obj, obj)
            if match is not None:
                sw_obj_id = obj
                action_queue.append({'action':'ToggleObjectOff', 'objectId':sw_obj_id, 'agent_id':agent_id})
                time.sleep(0.1)
    
    # all objects apart from Stove Burner
    else:
        for obj in objs:
            match = re.match(sw_obj, obj)
            if match is not None:
                sw_obj_id = obj
                break # find the first instance
        GoToObject(robot, sw_obj_id)
        time.sleep(1)
        action_queue.append({'action':'ToggleObjectOff', 'objectId':sw_obj_id, 'agent_id':agent_id})
        time.sleep(1)      
    
def OpenObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
        
    global recp_id
    if recp_id is not None:
        sw_obj_id = recp_id
    
    GoToObject(robot, sw_obj_id)
    time.sleep(1)
    action_queue.append({'action':'OpenObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
    time.sleep(1)
    
def CloseObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
        
    global recp_id
    if recp_id is not None:
        sw_obj_id = recp_id
        
    GoToObject(robot, sw_obj_id)
    time.sleep(1)
    
    action_queue.append({'action':'CloseObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 
    
    if recp_id is not None:
        recp_id = None
    time.sleep(1)
    
def BreakObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    GoToObject(robot, sw_obj_id)
    time.sleep(1)
    action_queue.append({'action':'BreakObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 
    time.sleep(1)
    
def SliceObject(robot, sw_obj):
    print ("Slicing: ", sw_obj)
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    GoToObject(robot, sw_obj_id)
    time.sleep(1)
    action_queue.append({'action':'SliceObject', 'objectId':sw_obj_id, 'agent_id':agent_id})      
    time.sleep(1)
    
def CleanObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    GoToObject(robot, sw_obj_id)
    time.sleep(1)
    action_queue.append({'action':'CleanObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 
    time.sleep(1)
    
def ThrowObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    
    action_queue.append({'action':'ThrowObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 
    time.sleep(1)
def goto(robot, max_iterations: int = 400):
    target_label = _RANDOM_NAV_TARGET
    target_identifier = _RANDOM_NAV_TARGET_ID or target_label
    if target_identifier is None:
        raise RuntimeError("Navigation target has not been selected before calling goto().")
    print(f"[INFO] Navigating to selected object: {target_label or target_identifier}")
    GoToObject(robot, target_identifier, max_iterations=max_iterations)

import threading as _aq_th
import json as _aq_json
from datetime import datetime as _aq_dt


class _ActionQueue(list):
    def __init__(self, *args, _log_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_path = _log_path
        self._lock = _aq_th.Lock()
        self._counts: Dict[str, int] = {}
        self._per_agent: Dict[int, int] = {}
        self._total = 0

    def append(self, item):
        with self._lock:
            self._total += 1
            act = item.get('action') if isinstance(item, dict) else None
            ag = item.get('agent_id') if isinstance(item, dict) else None
            if act is not None:
                self._counts[act] = self._counts.get(act, 0) + 1
            if ag is not None:
                self._per_agent[ag] = self._per_agent.get(ag, 0) + 1
            if self._log_path and isinstance(item, dict):
                try:
                    with open(self._log_path, 'a', encoding='utf-8') as f:
                        rec = dict(ts=_aq_dt.now().isoformat(), **item)
                        f.write(_aq_json.dumps(rec, ensure_ascii=False) + '\n')
                except Exception:
                    # non-fatal
                    pass
        return super().append(item)

    def stats(self):
        with self._lock:
            return dict(total=self._total, by_action=self._counts, by_agent=self._per_agent)


def instrument_action_queue(queue: List[Dict[str, Any]], log_dir: Optional[Path] = None) -> _ActionQueue:
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent
    action_log_file = log_dir / 'actions_log.jsonl'
    try:
        if action_log_file.exists():
            action_log_file.unlink()
    except Exception:
        pass
    if isinstance(queue, _ActionQueue):
        queue._log_path = str(action_log_file)
        return queue
    return _ActionQueue(queue, _log_path=str(action_log_file))


def _next_video_index(video_output_dir: Path) -> int:
    max_index = 0
    for subdir in ("agent", "top_view"):
        folder = video_output_dir / subdir
        if not folder.exists():
            continue
        for candidate in folder.glob("*.mp4"):
            stem = candidate.stem
            if stem.isdigit():
                max_index = max(max_index, int(stem))
    return max_index + 1


def run_episode(
    output_base: Path = DEFAULT_OUTPUT_BASE,
    video_output_dir: Optional[Path] = None,
    agent_prefix: str = "agent_1",
    frame_rate: int = 5,
    done_repetitions: int = 25,
    nav_exclude: Optional[Sequence[str]] = None,
    random_seed: Optional[int] = None,
    min_nav_distance: float = 4.5,
    max_nav_iterations: int = 400,
    video_filename_stem: Optional[str] = None,
) -> Dict[str, Any]:
    global total_exec, success_exec, c, no_robot, action_queue, task_over, _RUN_OUTPUT_ROOT, _RUN_INFO, _RUN_SUFFIX, _RANDOM_NAV_TARGET, recp_id, _HAS_TOP_VIEW, reachable_positions, _RANDOM_NAV_TARGET_ID
    global _ROOM_VISITED_SET, _ROOM_VISIT_ORDER, _ROOM_TIMELINE, _AGENT_ROOM_SEQUENCES

    if nav_exclude is None:
        nav_exclude_set = {"Mug"}
    else:
        nav_exclude_set = set(nav_exclude)
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    sanitized_scene_id = _sanitize_label(_SCENE_ID)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    total_exec = 0
    success_exec = 0
    task_over = False
    recp_id = None
    action_queue = []
    _reset_room_tracking()

    video_destinations: Optional[Dict[str, Path]] = None
    video_index: Optional[int] = None
    sanitized_stem: Optional[str] = None
    if video_output_dir is not None:
        video_output_dir = Path(video_output_dir)
        if video_filename_stem:
            sanitized_stem = _sanitize_label(video_filename_stem)
        if sanitized_stem:
            video_destinations = {
                "agent": video_output_dir / "agent" / f"{sanitized_stem}.mp4",
                "top_view": video_output_dir / "top_view" / f"{sanitized_stem}.mp4",
            }
        else:
            sanitized_stem = None
            video_index = _next_video_index(video_output_dir)
            video_destinations = {
                "agent": video_output_dir / "agent" / f"{video_index:04d}.mp4",
                "top_view": video_output_dir / "top_view" / f"{video_index:04d}.mp4",
            }

    actions_thread: Optional[threading.Thread] = None
    produced_videos: Dict[str, Path] = {}
    run_output_root: Optional[Path] = None
    run_suffix: str = ""
    run_info: Dict[str, Any] = {}
    random_nav_target: Optional[str] = None

    try:
        c = _create_controller(height=1000, width=1000)
        c.reset(scene=_PROC_SCENE)
        no_robot = len(robots)

        multi_agent_event = c.step(dict(action='Initialize', agentMode="default", snapGrid=False, gridSize=0.5, rotateStepDegrees=20, visibilityDistance=1000, fieldOfView=180, agentCount=no_robot))

        _HAS_TOP_VIEW = False
        try:
            event = c.step(action="GetMapViewCameraProperties", raise_for_failure=True)
            top_view_props = copy.deepcopy(event.metadata["actionReturn"])
            bounds = event.metadata.get("sceneBounds", {}).get("size", {})
            max_bound = max(bounds.get("x", 0), bounds.get("z", 0), 1)
            top_view_props["orthographic"] = False
            top_view_props["fieldOfView"] = 50
            top_view_props["position"]["y"] += 1.1 * max_bound
            top_view_props["farClippingPlane"] = 50
            top_view_props.pop("orthographicSize", None)
            c.step(
                action="AddThirdPartyCamera",
                skyboxColor="white",
                raise_for_failure=True,
                **top_view_props,
            )
            _HAS_TOP_VIEW = True
        except Exception as _camera_err:
            print(f"[WARN] Failed to add third-party camera: {_camera_err}")
            _HAS_TOP_VIEW = False

        reachable_positions_ = c.step(action="GetReachablePositions").metadata["actionReturn"]
        reachable_positions = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]

        for i in range(no_robot):
            init_pos = random.choice(reachable_positions_)
            c.step(dict(action="Teleport", position=init_pos, agentId=i))

        for i in range(no_robot):
            c.step(action="LookDown", degrees=35, agentId=i)

        target_id, target_name = _select_navigation_target(c, sorted(nav_exclude_set), min_distance=min_nav_distance)
        sanitized_target = _sanitize_label(target_name or target_id)
        run_suffix = f"{sanitized_scene_id}_{sanitized_target}_{timestamp}"
        run_output_root = output_base / run_suffix
        run_output_root.mkdir(parents=True, exist_ok=True)

        run_info = {
            "house_id": _SCENE_ID,
            "split": PROC_THOR_SPLIT,
            "house_index": PROC_THOR_HOUSE_INDEX,
            "nav_target": target_name,
            "run_suffix": run_suffix,
            "output_root": str(run_output_root),
        }
        # include static room information discovered from the scene payload
        try:
            rooms_list = []
            for rid, info in _ROOM_LOOKUP.items():
                rooms_list.append({"id": rid, "roomType": info.get("roomType"), "area": info.get("area")})
            run_info["total_rooms"] = _TOTAL_ROOMS
            run_info["rooms"] = rooms_list
        except Exception:
            pass

        _RUN_OUTPUT_ROOT = run_output_root
        _RUN_INFO = run_info
        _RUN_SUFFIX = run_suffix
        random_nav_target = target_name
        _RANDOM_NAV_TARGET = target_name
        _RANDOM_NAV_TARGET_ID = target_id

        actions_thread = threading.Thread(target=exec_actions, daemon=True)
        actions_thread.start()

        goto(robots[0], max_iterations=max_nav_iterations)

        action_queue = instrument_action_queue(action_queue, run_output_root)

        for _ in range(done_repetitions):
            action_queue.append({'action':'Done'})
            action_queue.append({'action':'Done'})
            action_queue.append({'action':'Done'})
            time.sleep(0.1)

        task_over = True
        if actions_thread.is_alive():
            actions_thread.join()

        produced_videos = generate_video(frame_rate, video_destinations)
    finally:
        task_over = True
        if actions_thread and actions_thread.is_alive():
            actions_thread.join()
        cv2.destroyAllWindows()
        try:
            c.stop()
        except Exception:
            pass

    if run_output_root is None:
        raise RuntimeError("Run output directory was not initialized.")
    agent_dirs = [run_output_root / f"agent_{i + 1}_{run_suffix}" for i in range(no_robot)]
    top_view_dir = run_output_root / f"top_view_{run_suffix}"

    random_nav_target = random_nav_target or _RANDOM_NAV_TARGET
    run_info = run_info or _RUN_INFO

    result = {
        "run_output_root": str(run_output_root),
        "agent_dirs": [str(path) for path in agent_dirs if path.exists()],
        "top_view_dir": str(top_view_dir) if top_view_dir.exists() else None,
        "video_paths": {key: str(path) for key, path in produced_videos.items()},
        "run_suffix": run_suffix,
        "random_nav_target": random_nav_target,
        "run_info": run_info,
        "video_index": video_index,
        "agent_prefix": agent_prefix,
        "video_filename_stem": sanitized_stem,
    }
    # enrich run_info with recorded room visits and timeline
    try:
        enriched_info = dict(run_info or {})
        enriched_info.setdefault("visited_rooms", list(_ROOM_VISIT_ORDER))
        enriched_info.setdefault("visited_rooms_set", list(sorted(_ROOM_VISITED_SET)))
        enriched_info.setdefault("room_timeline", list(_ROOM_TIMELINE))
        # write back to disk for downstream consumers
        try:
            info_path = run_output_root / "run_info.json"
            info_path.write_text(json.dumps(enriched_info, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        result["run_info"] = enriched_info
    except Exception:
        pass

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute ProcTHOR plan and record episode data.")
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Directory where per-run frame dumps are stored.",
    )
    parser.add_argument(
        "--video-output-dir",
        type=Path,
        help="Directory where rendered videos should be saved (agent/ and top_view/ subdirectories will be created).",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=5,
        help="Frame rate used when rendering MP4 videos.",
    )
    parser.add_argument(
        "--done-repetitions",
        type=int,
        default=25,
        help="Number of Done action triplets to append before terminating the episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--nav-exclude",
        nargs="*",
        default=["Mug"],
        help="Object basenames to exclude when sampling the navigation target.",
    )
    parser.add_argument(
        "--agent-prefix",
        default="agent_1",
        help="Agent directory prefix (default: agent_1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_episode(
        output_base=args.output_base,
        video_output_dir=args.video_output_dir,
        agent_prefix=args.agent_prefix,
        frame_rate=args.frame_rate,
        done_repetitions=args.done_repetitions,
        nav_exclude=args.nav_exclude,
        random_seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
