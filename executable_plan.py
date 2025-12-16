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
from typing import Tuple
from collections import deque
import random
import os
import copy
import json
from pathlib import Path


import prior


PROC_THOR_SPLIT = os.environ.get("PROCTHOR_SPLIT", "train")
PROC_THOR_HOUSE_INDEX = 15
_DATASET = prior.load_dataset("procthor-10k")
_PROC_SCENE = _DATASET[PROC_THOR_SPLIT][PROC_THOR_HOUSE_INDEX]
_SCENE_ID = str(_PROC_SCENE.get("id", f"{PROC_THOR_SPLIT}_{PROC_THOR_HOUSE_INDEX}"))


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

def generate_video():
    frame_rate = 5
    if not _RUN_OUTPUT_ROOT.exists():
        print(f"The output directory {_RUN_OUTPUT_ROOT} does not exist; skipping video generation.")
        return
    for imgs_folder in sorted(_RUN_OUTPUT_ROOT.iterdir()):
        if not imgs_folder.is_dir():
            continue
        if not any(imgs_folder.glob("img_*.png")):
            continue
        view = imgs_folder.name
        command_set = [
            'ffmpeg',
            '-i', str(imgs_folder / 'img_%05d.png'),
            '-framerate', str(frame_rate),
            '-pix_fmt', 'yuv420p',
            str(_RUN_OUTPUT_ROOT / f'video_{view}.mp4'),
        ]
        subprocess.call(command_set)
        



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

_RANDOM_NAV_TARGET = _choose_random_object_name(_PROC_SCENE, excluded={"Mug"})

_OUTPUT_BASE = Path("/data5/zhuangyunhao/outputs/video")
_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
_SANITIZED_SCENE_ID = _sanitize_label(_SCENE_ID)
_SANITIZED_TARGET = _sanitize_label(_RANDOM_NAV_TARGET)
_RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
_RUN_SUFFIX = f"{_SANITIZED_SCENE_ID}_{_SANITIZED_TARGET}_{_RUN_TIMESTAMP}"
_RUN_OUTPUT_ROOT = _OUTPUT_BASE / _RUN_SUFFIX
_RUN_INFO = {
    "house_id": _SCENE_ID,
    "split": PROC_THOR_SPLIT,
    "house_index": PROC_THOR_HOUSE_INDEX,
    "nav_target": _RANDOM_NAV_TARGET,
    "run_suffix": _RUN_SUFFIX,
    "output_root": str(_RUN_OUTPUT_ROOT),
}


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

c = _create_controller(height=1000, width=1000)
c.reset(scene=_PROC_SCENE)
no_robot = len(robots)

# initialize n agents into the scene
multi_agent_event = c.step(dict(action='Initialize', agentMode="default", snapGrid=False, gridSize=0.5, rotateStepDegrees=20, visibilityDistance=1000, fieldOfView=180, agentCount=no_robot))

# add a top view camera
_HAS_TOP_VIEW = False
event = c.step(action="GetMapViewCameraProperties", raise_for_failure=True)
try:
    top_view_props = copy.deepcopy(event.metadata["actionReturn"])
    bounds = event.metadata.get("sceneBounds", {}).get("size", {})
    max_bound = max(bounds.get("x", 0), bounds.get("z", 0), 1)
    top_view_props["orthographic"] = False
    top_view_props["fieldOfView"] = 50
    top_view_props["position"]["y"] += 1.1 * max_bound
    top_view_props["farClippingPlane"] = 50
    top_view_props.pop("orthographicSize", None)
    event = c.step(
        action="AddThirdPartyCamera",
        skyboxColor="white",
        raise_for_failure=True,
        **top_view_props,
    )
    _HAS_TOP_VIEW = True
except Exception as _camera_err:
    print(f"[WARN] Failed to add third-party camera: {_camera_err}")
    _HAS_TOP_VIEW = False

# get reachabel positions
reachable_positions_ = c.step(action="GetReachablePositions").metadata["actionReturn"]
reachable_positions = positions_tuple = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]

# randomize postions of the agents
for i in range (no_robot):
    init_pos = random.choice(reachable_positions_)
    c.step(dict(action="Teleport", position=init_pos, agentId=i))
    
objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])
# print (objs)
    
# x = c.step(dict(action="RemoveFromScene", objectId='Lettuce|+01.11|+00.83|-01.43'))
#c.step({"action":"InitialRandomSpawn", "excludedReceptacles":["Microwave", "Pan", "Chair", "Plate", "Fridge", "Cabinet", "Drawer", "GarbageCan"]})
# c.step({"action":"InitialRandomSpawn", "excludedReceptacles":["Cabinet", "Drawer", "GarbageCan"]})

action_queue = []

task_over = False

recp_id = None

for i in range (no_robot):
    multi_agent_event = c.step(action="LookDown", degrees=35, agentId=i)
    # c.step(action="LookUp", degrees=30, 'agent_id':i)

def exec_actions():
    global total_exec, success_exec
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
                metadata = {
                    "agent_id": i,
                    "agent_name": robots[i]["name"],
                    "frame_index": img_counter,
                    "itemlist": visible_ids,
                    "house_id": _SCENE_ID,
                    "nav_target": _RANDOM_NAV_TARGET,
                }
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

def GoToObject(robots, dest_obj):
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
    if "|" in dest_obj:
        # obj alredy given
        dest_obj_id = dest_obj
        pos_arr = dest_obj_id.split("|")
        dest_obj_center = {'x': float(pos_arr[1]), 'y': float(pos_arr[2]), 'z': float(pos_arr[3])}
    else:
        for idx, obj in enumerate(objs):
            
            match = re.match(dest_obj, obj)
            if match is not None:
                dest_obj_id = obj
                dest_obj_center = objs_center[idx]
                if dest_obj_center != {'x': 0.0, 'y': 0.0, 'z': 0.0}:
                    break # find the first instance
        
    print ("Going to ", dest_obj_id, dest_obj_center)
        
    dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']] 
    
    # closest reachable position for each robot
    # all robots cannot reach the same spot 
    # differt close points needs to be found for each robot
    crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
    goal_thresh = 0.5
    # at least one robot is far away from the goal
    
    while all(d > goal_thresh for d in dist_goals):
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
def goto(robot):
    print(f"[INFO] Navigating to randomly selected object: {_RANDOM_NAV_TARGET}")
    GoToObject(robot, _RANDOM_NAV_TARGET)

# Execute the task using Robot 2
goto(robots[0])

# Task "Put mug in the coffee machine and switch on the coffee machine" is done

# ---- Action queue instrumentation (auto-injected) ----
import threading as _aq_th
import json as _aq_json
from datetime import datetime as _aq_dt

class _ActionQueue(list):
    def __init__(self, *args, _log_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_path = _log_path
        self._lock = _aq_th.Lock()
        self._counts = {}
        self._per_agent = {}
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
                    with open(self._log_path, 'a') as f:
                        rec = dict(ts=_aq_dt.now().isoformat(), **item)
                        f.write(_aq_json.dumps(rec, ensure_ascii=False) + '\n')
                except Exception as _e:
                    # non-fatal
                    pass
        return super().append(item)

    def stats(self):
        with self._lock:
            return dict(total=self._total, by_action=self._counts, by_agent=self._per_agent)

_action_log_file = os.path.join(os.path.dirname(__file__), 'actions_log.jsonl')
try:
    if os.path.exists(_action_log_file):
        os.remove(_action_log_file)
except Exception:
    pass

try:
    action_queue = _ActionQueue(action_queue, _log_path=_action_log_file)
except NameError:
    # In case action_queue not yet defined, create a new one
    action_queue = _ActionQueue([], _log_path=_action_log_file)
# ---- end instrumentation ----
no_trans = 1

for i in range(25):
    action_queue.append({'action':'Done'})
    action_queue.append({'action':'Done'})
    action_queue.append({'action':'Done'})
    time.sleep(0.1)

task_over = True
time.sleep(5)


generate_video()
