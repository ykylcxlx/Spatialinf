#!/usr/bin/env python3
"""Build QA pairs from train_multiturn_9390_pickup_and_put.json using task metadata.

Reads records from the filtered train JSON, locates the corresponding FloorPlan metadata
under `procthordata/pickup_and_put_task_metadata`, and generates QA pairs similar to
`generate_pickup_put_qas.py`.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DATA_ROOT = Path("/data5/zhuangyunhao")
INPUT_JSON = DATA_ROOT / "procthordata" / "train_multiturn_9390_pickup_and_put.json"
METADATA_DIR = DATA_ROOT / "procthordata" / "pickup_and_put_task_metadata"
IMAGES_ROOT = DATA_ROOT / "procthordata" / "data" / "images" / "pickup_and_put"
OUTPUT_DIR = DATA_ROOT / "outputs" / "pickup_put_from_train"
OUTPUT_NAME = "pickup_put_qas_from_train.json"
SEED = 2026
MAX_Q_PER_EP = 6
MAX_CHOICES = 4

FIRST_ACTION_TEMPLATES = [
    "What is the first action the agent executes in this episode?",
    "Which step kicks off the agent's plan?",
    "At the start of the task, what does the agent do?",
    "Which action does the agent perform first in this run?",
    "Identify the initial action taken by the agent.",
]

PICKUP_OBJECT_TEMPLATES = [
    "Which object does the agent pick up during this task?",
    "What item is collected when the agent performs the pickup action?",
    "During the pickup step, which object is targeted?",
    "Which object is grabbed by the agent in the pickup step?",
    "Name the item the agent picks up as part of the episode.",
]

PUT_TARGET_TEMPLATES = [
    "Where does the agent place the picked-up item?",
    "Which destination object receives the item at the end?",
    "What container does the agent use to finish the task?",
    "Into which object does the agent put the item?",
    "Which receptacle is used to place the picked item?",
]

ACTION_COUNT_TEMPLATES = [
    "How many rewarded steps (excluding the final 'end' action) does the agent complete?",
    "Counting every rewarded move except the concluding 'end', how many steps occur?",
    "Excluding the 'end' signal, how many reward-yielding actions are executed?",
    "How many actionable steps that yield reward are performed before 'end'?",
    "What is the total number of reward-bearing actions (not counting 'end')?",
]

FINAL_STATE_TEMPLATES = [
    "At the conclusion, which object holds the picked-up item?",
    "By the end of the episode, where does the picked item end up?",
    "Looking at the final state, which object contains the picked-up item?",
    "When the episode finishes, which destination contains the item?",s
    "In the final scene, which object holds the item the agent picked up?",
]

NEXT_ACTION_TEMPLATES = [
    "After performing '{current}', what does the agent do next?",
    "Which step immediately follows '{current}'?",
    "What action succeeds '{current}' in the sequence?",
    "Following '{current}', which action comes next?",
    "What is the subsequent action after '{current}'?",
]

ACTION_ORDER_TEMPLATES = [
    "Arrange these key steps in the correct order of execution: {actions}.",
    "Which option lists the highlighted actions in the order the agent performs them?",
    "From earliest to latest, how do these actions unfold: {actions}?",
    "Put the following actions in the sequence the agent carries them out: {actions}.",
    "Select the choice that shows the actions {actions} in execution order.",
]


def safe_load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_floorplan_from_images(images: List[str]) -> Optional[str]:
    # look for FloorPlanNN in any image path
    for img in images:
        m = re.search(r"(FloorPlan\d+)", img)
        if m:
            return m.group(1)
    return None


def parse_task_instruction_from_messages(messages: List[Dict[str, Any]]) -> Optional[str]:
    # Look for a message that contains 'Task:' and return the task string
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content", "")
        found = re.search(r"Task:\s*\"([^\"]+)\"", content)
        if found:
            return found.group(1)
        # fallback: look for "put the ... in the ..."
        found2 = re.search(r"put the (.+?) in the (.+?)[\?\.]", content, flags=re.I)
        if found2:
            return f"put the {found2.group(1)} in the {found2.group(2)}"
    return None


def load_floorplan_metadata(floorplan: str):
    if not floorplan:
        return []
    path = METADATA_DIR / f"{floorplan}.json"
    if not path.exists():
        return []
    data = safe_load_json(path)
    # flatten, many files are nested lists
    flat: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, list):
            for sub in item:
                if isinstance(sub, dict):
                    flat.append(sub)
        elif isinstance(item, dict):
            flat.append(item)
    return flat


def find_best_metadata_entry(metadata_list: List[Dict[str, Any]], task_instruction: Optional[str]):
    if not metadata_list:
        return None
    if not task_instruction:
        return metadata_list[0]
    instr = task_instruction.lower()
    for meta in metadata_list:
        tn = (meta.get("taskname") or "").lower()
        if tn and all(tok in tn for tok in re.findall(r"\w+", instr)):
            return meta
    # fallback: match on pickup and put tokens if possible
    m = re.search(r"put the (.+?) in the (.+)", instr)
    if m:
        pick = m.group(1).strip().lower()
        cont = m.group(2).strip().lower()
        for meta in metadata_list:
            tn = (meta.get("taskname") or "").lower()
            if pick in tn and cont in tn:
                return meta
    return metadata_list[0]


def format_action_label(row: Dict[str, Any]) -> str:
    action = (row.get("action") or "").strip()
    objtype = (row.get("objectType") or "").strip()
    if not objtype:
        objtype = (row.get("objectId") or "").split("|")[0]
    if objtype:
        return f"{action} {objtype}".strip()
    return action or "unknown"


def extract_object_pool_from_metadata(meta: Dict[str, Any]) -> List[str]:
    pool = []
    for row in meta.get("actions", []) or []:
        for key in ("objectType", "objectId"):
            v = row.get(key) or ""
            if not v:
                continue
            name = v.split("|")[0]
            if name and name not in pool:
                pool.append(name)
        for related in row.get("relatedObject", []) or []:
            name = (related or "").split("|")[0]
            if name and name not in pool:
                pool.append(name)
    return pool


def assemble_choices(rng: random.Random, correct: str, pool: Sequence[str], max_options: int = MAX_CHOICES) -> Optional[List[str]]:
    distractors = [p for p in pool if p != correct]
    rng.shuffle(distractors)
    options = [correct]
    for d in distractors:
        if d not in options:
            options.append(d)
        if len(options) >= max_options:
            break
    if len(options) < 2:
        return None
    rng.shuffle(options)
    return options


def make_questions_for_record(rng: random.Random, record: Dict[str, Any]) -> List[Dict[str, Any]]:
    images = record.get("images") or []
    floorplan = extract_floorplan_from_images(images)
    metadata_list = load_floorplan_metadata(floorplan)
    task_instr = parse_task_instruction_from_messages(record.get("messages", []))
    meta = find_best_metadata_entry(metadata_list, task_instr)
    if not meta:
        return []
    action_rows = meta.get("actions", []) or []
    action_labels = [format_action_label(r) for r in action_rows]
    unique_labels = []
    seen = set()
    for lab in action_labels:
        low = lab.lower()
        if lab and low not in seen and low != "end":
            unique_labels.append(lab)
            seen.add(low)

    object_pool = extract_object_pool_from_metadata(meta)
    task_pool = [meta.get("taskname", "")]

    questions: List[Dict[str, Any]] = []

    # first action
    if len(action_labels) >= 1:
        correct = action_labels[0]
        opts = assemble_choices(rng, correct, unique_labels or object_pool)
        if opts:
            questions.append({
                "question": rng.choice(FIRST_ACTION_TEMPLATES),
                "choices": opts,
                "answer_index": opts.index(correct),
                "answer": correct,
                "metadata": {"source_file": str(meta.get("metadatapath", "")), "images": images},
            })

    # pickup object
    pickup_row = next((r for r in action_rows if (r.get("action") or "").strip().lower() == "pickup"), None)
    if pickup_row:
        answer = (pickup_row.get("objectType") or pickup_row.get("objectId") or "").split("|")[0]
        if answer:
            opts = assemble_choices(rng, answer, object_pool)
            if opts:
                questions.append({
                    "question": rng.choice(PICKUP_OBJECT_TEMPLATES),
                    "choices": opts,
                    "answer_index": opts.index(answer),
                    "answer": answer,
                    "metadata": {"source_file": str(meta.get("metadatapath", "")), "images": images},
                })

    # put target
    put_row = next((r for r in action_rows if (r.get("action") or "").strip().lower() == "put"), None)
    if put_row:
        answer = (put_row.get("objectType") or put_row.get("objectId") or "").split("|")[0]
        if answer:
            opts = assemble_choices(rng, answer, object_pool)
            if opts:
                questions.append({
                    "question": rng.choice(PUT_TARGET_TEMPLATES),
                    "choices": opts,
                    "answer_index": opts.index(answer),
                    "answer": answer,
                    "metadata": {"source_file": str(meta.get("metadatapath", "")), "images": images},
                })

    # action count
    rewarded = [r for r in action_rows if (r.get("action") or "").strip().lower() != "end"]
    if rewarded:
        answer_val = len(rewarded)
        opts_raw = {answer_val}
        if answer_val > 1:
            opts_raw.add(answer_val - 1)
        opts_raw.add(answer_val + 1)
        while len(opts_raw) < MAX_CHOICES:
            opts_raw.add(answer_val + rng.randint(2, 4))
        opts = [str(x) for x in sorted(list(opts_raw))[:MAX_CHOICES]]
        rng.shuffle(opts)
        questions.append({
            "question": rng.choice(ACTION_COUNT_TEMPLATES),
            "choices": opts,
            "answer_index": opts.index(str(answer_val)),
            "answer": str(answer_val),
            "metadata": {"source_file": str(meta.get("metadatapath", "")), "images": images},
        })

    # final state
    if pickup_row and put_row:
        item = (pickup_row.get("objectType") or pickup_row.get("objectId") or "").split("|")[0]
        container = (put_row.get("objectType") or put_row.get("objectId") or "").split("|")[0]
        if item and container:
            ask_item = rng.random() < 0.5
            if ask_item:
                prompt = rng.choice(FINAL_STATE_TEMPLATES)
                correct = item
            else:
                prompt = rng.choice(FINAL_STATE_TEMPLATES)
                correct = container
            opts = assemble_choices(rng, correct, object_pool)
            if opts:
                questions.append({
                    "question": prompt,
                    "choices": opts,
                    "answer_index": opts.index(correct),
                    "answer": correct,
                    "metadata": {"item": item, "container": container, "images": images},
                })

    # next action questions
    labels = [l for l in action_labels if l]
    for i in range(len(labels) - 1):
        cur = labels[i]
        nxt = labels[i + 1]
        if cur == nxt:
            continue
        opts = assemble_choices(rng, nxt, unique_labels or labels)
        if not opts:
            continue
        questions.append({
            "question": rng.choice(NEXT_ACTION_TEMPLATES).format(current=cur),
            "choices": opts,
            "answer_index": opts.index(nxt),
            "answer": nxt,
            "metadata": {"step_index": i, "images": images},
        })

    # action order (if enough)
    if len(unique_labels) >= 3:
        focus = unique_labels[:4]
        # create simple permutations
        perms = [" -> ".join(focus)]
        # add up to MAX_CHOICES-1 shuffled permutations
        import itertools

        for perm in itertools.permutations(focus):
            s = " -> ".join(perm)
            if s not in perms:
                perms.append(s)
            if len(perms) >= MAX_CHOICES:
                break
        rng.shuffle(perms)
        correct = " -> ".join(focus)
        if correct in perms:
            questions.append({
                "question": rng.choice(ACTION_ORDER_TEMPLATES).format(actions=", ".join(focus)),
                "choices": perms,
                "answer_index": perms.index(correct),
                "answer": correct,
                "metadata": {"images": images},
            })

    # limit questions
    rng.shuffle(questions)
    return questions[:MAX_Q_PER_EP]


def main():
    rng = random.Random(SEED)
    data = safe_load_json(INPUT_JSON)
    all_qs: List[Dict[str, Any]] = []
    for rec in data:
        qs = make_questions_for_record(rng, rec)
        if qs:
            all_qs.extend(qs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / OUTPUT_NAME
    out_path.write_text(json.dumps(all_qs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_qs)} questions to {out_path}")

if __name__ == "__main__":
    main()
