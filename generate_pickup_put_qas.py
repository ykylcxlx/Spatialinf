"""Generate QA pairs from pickup-and-put task episodes."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Question = Dict[str, Any]

FIRST_ACTION_TEMPLATES: Sequence[str] = (
    "What is the first action the agent executes in this episode?",
    "Which step kicks off the agent's plan?",
    "At the start of the task, what does the agent do?",
)

PICKUP_OBJECT_TEMPLATES: Sequence[str] = (
    "Which object does the agent pick up during this task?",
    "What item is collected when the agent performs the pickup action?",
    "During the pickup step, which object is targeted?",
)

PUT_TARGET_TEMPLATES: Sequence[str] = (
    "Where does the agent place the picked-up item?",
    "Which destination object receives the item at the end?",
    "What container does the agent use to finish the task?",
)

ACTION_ORDER_TEMPLATES: Sequence[str] = (
    "Arrange these key steps in the correct order of execution: {actions}.",
    "Which option lists the highlighted actions in the order the agent performs them?",
    "From earliest to latest, how do these actions unfold: {actions}?",
)

NEXT_ACTION_TEMPLATES: Sequence[str] = (
    "After performing '{current}', what does the agent do next?",
    "Which step immediately follows '{current}'?",
    "What action succeeds '{current}' in the sequence?",
)

ACTION_COUNT_TEMPLATES: Sequence[str] = (
    "How many rewarded steps (excluding the final 'end' action) does the agent complete?",
    "Counting every rewarded move except the concluding 'end', how many steps occur?",
    "Excluding the 'end' signal, how many reward-yielding actions are executed?",
)

TASK_GOAL_TEMPLATES: Sequence[str] = (
    "What is the stated task objective for this episode?",
    "Which instruction best matches the goal the agent is following?",
    "Identify the task description assigned to the agent.",
)

FINAL_STATE_BY_CONTAINER_TEMPLATES: Sequence[str] = (
    "By the end of the episode, which object is placed inside the {container}?",
    "At the conclusion, what item ends up within the {container}?",
    "Looking at the final state, which object rests in the {container}?",
)

FINAL_STATE_BY_ITEM_TEMPLATES: Sequence[str] = (
    "When the episode finishes, where is the {item} located?",
    "Which object holds the {item} at the end of the run?",
    "After the final step, the {item} is situated in which object?",
)

MAX_CHOICES = 4

@dataclass
class Episode:
    source_path: Path
    scene: str
    task_type: str
    task_name: str
    action_rows: List[Dict[str, Any]]
    action_labels: List[str]
    unique_labels: List[str]
    object_pool: List[str]
    trajectory: List[Tuple[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data5/zhuangyunhao/embodied_reasoner/data_engine/data_pickup_and_put"),
        help="Directory containing pickup-and-put episode folders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-questions-per-episode",
        type=int,
        default=6,
        help="Maximum number of QA pairs to generate per episode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data5/zhuangyunhao/outputs/pickup_put"),
        help="Directory where the aggregated QA JSON will be written.",
    )
    parser.add_argument(
        "--output-name",
        default="pickup_put_questions.json",
        help="Filename for the aggregated QA JSON.",
    )
    return parser.parse_args()


def normalize_object_name(raw: Optional[str]) -> str:
    if not raw:
        return ""
    name = raw.split("|")[0].strip()
    return name


def format_action_label(row: Dict[str, Any]) -> str:
    action = (row.get("action") or "").strip()
    target = (row.get("objectType") or "").strip()
    if not target:
        target = normalize_object_name(row.get("objectId"))
    if not target and row.get("relatedObject"):
        target = normalize_object_name(row["relatedObject"][0])
    if target:
        return f"{action} {target}".strip()
    return action or "unknown"


def extract_action_objects(row: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for key in ("objectType", "objectId"):
        candidate = normalize_object_name(row.get(key))
        if candidate:
            names.append(candidate)
    for related in row.get("relatedObject", []) or []:
        candidate = normalize_object_name(related)
        if candidate:
            names.append(candidate)
    return names


def clean_text_block(block: str) -> Tuple[str, str]:
    tag_match = re.match(r"\s*<([^>]+)>", block)
    tag = tag_match.group(1).strip() if tag_match else "Unknown"
    content = re.sub(r"^\s*<[^>]+>\s*", "", block)
    content = re.sub(r"</[^>]+>\s*$", "", content).strip()
    return tag, content


def parse_trajectory(raw_entries: Sequence[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for entry in raw_entries:
        tag, content = clean_text_block(entry)
        parsed.append((tag, content))
    return parsed


def load_episode_from_file(json_path: Path) -> List[Episode]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    episodes: List[Episode] = []
    for record in data:
        actions = record.get("task_metadata", {}).get("actions", [])
        action_labels = [format_action_label(row) for row in actions]
        dedup_labels: List[str] = []
        seen: set[str] = set()
        for label in action_labels:
            if label and label not in seen and label.lower() != "end":
                dedup_labels.append(label)
                seen.add(label)
        all_objects: List[str] = []
        for row in actions:
            all_objects.extend(extract_action_objects(row))
        episode = Episode(
            source_path=json_path,
            scene=record.get("scene", ""),
            task_type=record.get("tasktype", ""),
            task_name=record.get("taskname", ""),
            action_rows=actions,
            action_labels=action_labels,
            unique_labels=dedup_labels,
            object_pool=sorted(set(all_objects)),
            trajectory=parse_trajectory(record.get("trajectory", [])),
        )
        episodes.append(episode)
    return episodes


def discover_episode_files(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")
    json_files: List[Path] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        candidate = folder / f"{folder.name}.json"
        if candidate.exists():
            json_files.append(candidate)
    if not json_files:
        raise FileNotFoundError(f"No episode files discovered under {root}")
    return json_files


def assemble_choices(
    rng: random.Random,
    correct: str,
    pool: Sequence[str],
    max_options: int = MAX_CHOICES,
) -> Optional[List[str]]:
    distractors = [item for item in pool if item != correct]
    rng.shuffle(distractors)
    options = [correct]
    for item in distractors:
        if item not in options:
            options.append(item)
        if len(options) >= max_options:
            break
    if len(options) < 2:
        return None
    rng.shuffle(options)
    return options


def question_metadata(base: Dict[str, Any], episode: Episode) -> Dict[str, Any]:
    meta = dict(base)
    video_path = episode.source_path.parent / "step_frames.mp4"
    meta.update(
        {
            "scene": episode.scene,
            "task_type": episode.task_type,
            "task_name": episode.task_name,
            "source_file": str(episode.source_path),
            "video_path": str(video_path),
        }
    )
    return meta


def make_first_action_question(
    rng: random.Random,
    episode: Episode,
    global_pool: Sequence[str],
) -> Optional[Question]:
    labels = [label for label in episode.action_labels if label]
    if len(labels) < 2:
        return None
    correct = labels[0]
    options = assemble_choices(rng, correct, global_pool)
    if not options:
        return None
    prompt = rng.choice(FIRST_ACTION_TEMPLATES)
    metadata = question_metadata({"question_type": "first_action", "correct_action": correct}, episode)
    return {
        "question": prompt,
        "choices": options,
        "answer_index": options.index(correct),
        "answer": correct,
        "metadata": metadata,
    }


def make_pickup_object_question(
    rng: random.Random,
    episode: Episode,
    object_pool: Sequence[str],
) -> Optional[Question]:
    pickup_row = next((row for row in episode.action_rows if (row.get("action") or "").strip().lower() == "pickup"), None)
    if not pickup_row:
        return None
    answer = normalize_object_name(pickup_row.get("objectType")) or normalize_object_name(pickup_row.get("objectId"))
    if not answer:
        return None
    options = assemble_choices(rng, answer, object_pool)
    if not options:
        return None
    prompt = rng.choice(PICKUP_OBJECT_TEMPLATES)
    metadata = question_metadata({"question_type": "pickup_object", "pickup_target": answer}, episode)
    return {
        "question": prompt,
        "choices": options,
        "answer_index": options.index(answer),
        "answer": answer,
        "metadata": metadata,
    }


def make_put_target_question(
    rng: random.Random,
    episode: Episode,
    object_pool: Sequence[str],
) -> Optional[Question]:
    put_row = next((row for row in episode.action_rows if (row.get("action") or "").strip().lower() == "put"), None)
    if not put_row:
        return None
    answer = normalize_object_name(put_row.get("objectType")) or normalize_object_name(put_row.get("objectId"))
    if not answer:
        return None
    options = assemble_choices(rng, answer, object_pool)
    if not options:
        return None
    prompt = rng.choice(PUT_TARGET_TEMPLATES)
    metadata = question_metadata({"question_type": "put_target", "destination": answer}, episode)
    return {
        "question": prompt,
        "choices": options,
        "answer_index": options.index(answer),
        "answer": answer,
        "metadata": metadata,
    }


def make_action_count_question(rng: random.Random, episode: Episode) -> Optional[Question]:
    rewarded = [row for row in episode.action_rows if (row.get("action") or "").strip().lower() != "end"]
    if not rewarded:
        return None
    answer_value = len(rewarded)
    options_raw = {answer_value}
    if answer_value > 1:
        options_raw.add(answer_value - 1)
    options_raw.add(answer_value + 1)
    while len(options_raw) < MAX_CHOICES:
        options_raw.add(answer_value + rng.randint(2, 4))
    options = sorted(options_raw)[:MAX_CHOICES]
    rng.shuffle(options)
    labels = [str(value) for value in options]
    answer = str(answer_value)
    prompt = rng.choice(ACTION_COUNT_TEMPLATES)
    metadata = question_metadata({"question_type": "action_count", "rewarded_steps": answer_value}, episode)
    return {
        "question": prompt,
        "choices": labels,
        "answer_index": labels.index(answer),
        "answer": answer,
        "metadata": metadata,
    }


def build_order_options(rng: random.Random, sequence: Sequence[str]) -> Optional[List[str]]:
    if len(sequence) < 3:
        return None
    seen = set()
    perms: List[Tuple[str, ...]] = []
    for perm in permutations(sequence):
        if perm not in seen:
            seen.add(perm)
            perms.append(perm)
    rng.shuffle(perms)
    options: List[Tuple[str, ...]] = []
    correct = tuple(sequence)
    options.append(correct)
    for perm in perms:
        if perm == correct:
            continue
        options.append(perm)
        if len(options) >= MAX_CHOICES:
            break
    if len(options) < 2:
        return None
    formatted = [" -> ".join(option) for option in options]
    rng.shuffle(formatted)
    return formatted


def make_final_state_question(
    rng: random.Random,
    episode: Episode,
    object_pool: Sequence[str],
) -> Optional[Question]:
    pickup_row = next((row for row in episode.action_rows if (row.get("action") or "").strip().lower() == "pickup"), None)
    put_row = next((row for row in episode.action_rows if (row.get("action") or "").strip().lower() == "put"), None)
    if not pickup_row or not put_row:
        return None

    item = normalize_object_name(pickup_row.get("objectType")) or normalize_object_name(pickup_row.get("objectId"))
    container = normalize_object_name(put_row.get("objectType")) or normalize_object_name(put_row.get("objectId"))
    if not item or not container:
        return None

    ask_for_item = rng.random() < 0.5
    if ask_for_item:
        prompt = rng.choice(FINAL_STATE_BY_CONTAINER_TEMPLATES).format(container=container)
        answer = item
    else:
        prompt = rng.choice(FINAL_STATE_BY_ITEM_TEMPLATES).format(item=item)
        answer = container

    options = assemble_choices(rng, answer, object_pool)
    if not options:
        return None

    metadata = question_metadata(
        {
            "question_type": "final_state",
            "item": item,
            "container": container,
            "variant": "ask_item" if ask_for_item else "ask_container",
        },
        episode,
    )

    return {
        "question": prompt,
        "choices": options,
        "answer_index": options.index(answer),
        "answer": answer,
        "metadata": metadata,
    }


def make_action_order_question(rng: random.Random, episode: Episode) -> Optional[Question]:
    sequence = episode.unique_labels
    if len(sequence) < 3:
        return None
    focus = sequence[:4]
    options = build_order_options(rng, focus)
    if not options:
        return None
    answer = " -> ".join(focus)
    if answer not in options:
        options[0] = answer
    rng.shuffle(options)
    question_actions = ", ".join(focus)
    prompt = rng.choice(ACTION_ORDER_TEMPLATES).format(actions=question_actions)
    metadata = question_metadata({"question_type": "action_order", "ordered_actions": focus}, episode)
    return {
        "question": prompt,
        "choices": options,
        "answer_index": options.index(answer),
        "answer": answer,
        "metadata": metadata,
    }


def make_next_action_questions(
    rng: random.Random,
    episode: Episode,
    global_pool: Sequence[str],
) -> List[Question]:
    labels = [label for label in episode.action_labels if label]
    results: List[Question] = []
    for idx in range(len(labels) - 1):
        current_label = labels[idx]
        next_label = labels[idx + 1]
        if current_label == next_label:
            continue
        options = assemble_choices(rng, next_label, global_pool)
        if not options:
            continue
        prompt = rng.choice(NEXT_ACTION_TEMPLATES).format(current=current_label)
        metadata = question_metadata(
            {
                "question_type": "next_action",
                "current_action": current_label,
                "next_action": next_label,
                "step_index": idx,
            },
            episode,
        )
        question = {
            "question": prompt,
            "choices": options,
            "answer_index": options.index(next_label),
            "answer": next_label,
            "metadata": metadata,
        }
        results.append(question)
    return results


def make_task_goal_question(
    rng: random.Random,
    episode: Episode,
    task_pool: Sequence[str],
) -> Optional[Question]:
    answer = episode.task_name.strip()
    if not answer:
        return None
    options = assemble_choices(rng, answer, task_pool)
    if not options:
        return None
    prompt = rng.choice(TASK_GOAL_TEMPLATES)
    metadata = question_metadata({"question_type": "task_goal"}, episode)
    return {
        "question": prompt,
        "choices": options,
        "answer_index": options.index(answer),
        "answer": answer,
        "metadata": metadata,
    }


def gather_global_pools(episodes: Sequence[Episode]) -> Tuple[List[str], List[str], List[str]]:
    action_labels = sorted({label for ep in episodes for label in ep.action_labels if label})
    object_names = sorted({name for ep in episodes for name in ep.object_pool if name})
    tasks = sorted({ep.task_name.strip() for ep in episodes if ep.task_name.strip()})
    return action_labels, object_names, tasks


def generate_questions_for_episode(
    rng: random.Random,
    episode: Episode,
    action_pool: Sequence[str],
    object_pool: Sequence[str],
    task_pool: Sequence[str],
    max_count: int,
) -> List[Question]:
    candidates: List[Question] = []
    qa_first_action = make_first_action_question(rng, episode, action_pool)
    if qa_first_action:
        candidates.append(qa_first_action)

    qa_pickup = make_pickup_object_question(rng, episode, object_pool)
    if qa_pickup:
        candidates.append(qa_pickup)

    qa_put = make_put_target_question(rng, episode, object_pool)
    if qa_put:
        candidates.append(qa_put)

    qa_final = make_final_state_question(rng, episode, object_pool)
    if qa_final:
        candidates.append(qa_final)

    qa_count = make_action_count_question(rng, episode)
    if qa_count:
        candidates.append(qa_count)

    qa_order = make_action_order_question(rng, episode)
    if qa_order:
        candidates.append(qa_order)

    qa_goal = make_task_goal_question(rng, episode, task_pool)
    if qa_goal:
        candidates.append(qa_goal)

    next_action_questions = make_next_action_questions(rng, episode, action_pool)
    if next_action_questions:
        candidates.extend(next_action_questions)

    rng.shuffle(candidates)
    return candidates[:max_count]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    episode_files = discover_episode_files(args.data_root)
    episodes: List[Episode] = []
    for path in episode_files:
        episodes.extend(load_episode_from_file(path))

    if not episodes:
        raise RuntimeError("No episodes parsed from the provided data root.")

    action_pool, object_pool, task_pool = gather_global_pools(episodes)
    if not action_pool or not object_pool:
        raise RuntimeError("Insufficient action or object diversity to build questions.")

    all_questions: List[Question] = []
    for episode in episodes:
        questions = generate_questions_for_episode(
            rng,
            episode,
            action_pool,
            object_pool,
            task_pool,
            args.max_questions_per_episode,
        )
        all_questions.extend(questions)

    if not all_questions:
        raise RuntimeError("No questions could be generated from the episodes.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    output_path.write_text(json.dumps(all_questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(all_questions)} questions to {output_path}")


if __name__ == "__main__":
    main()
