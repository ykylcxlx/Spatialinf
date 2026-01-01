"""Generate memory-based ordering, co-occurrence, and other recall QA pairs from recorded ProcTHOR agent videos."""

from __future__ import annotations

import argparse
import json
import random
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from executable_plan import run_episode

# Templates phrased with light stylistic variety to avoid repetition.
THREE_OBJECT_TEMPLATES: Sequence[str] = (
    "Recall the clip: what is the order in which {objects} first appear on screen?",
    "Thinking back to the video, which option correctly lists when {objects} are first spotted?",
    "From earliest sighting to latest, how do {objects} show up in the footage?",
    "Which sequence reflects the first appearance timeline for {objects}?",
    "When replaying the scene mentally, how should {objects} be arranged by debut time?",
    "Which choice orders {objects} from the first object spotted to the last?",
)

FOUR_OBJECT_TEMPLATES: Sequence[str] = (
    "Choose the correct order of first appearance for {objects} in the recording.",
    "Review the video mentally: how should {objects} be arranged by their initial sightings?",
    "Memory check: which option captures the first-on-screen sequence of {objects}?",
    "Within the clip, what is the chronological debut order for {objects}?",
    "Which answer lists {objects} in the sequence they first come into view?",
    "Recalling the footage, how do {objects} line up from earliest glimpse to latest?",
)

CO_OCCURRENCE_TEMPLATES: Sequence[str] = (
    "Frames {frame_a} and {frame_b} both contain which group of objects?",
    "Looking at frame {frame_a} alongside frame {frame_b}, which option lists the shared objects?",
    "In the shots captured at frames {frame_a} and {frame_b}, which objects appear in both?",
    "Which selection names every object visible in both frame {frame_a} and frame {frame_b}?",
    "Comparing frame {frame_a} with frame {frame_b}, which set of objects overlaps between them?",
    "Which option lists only the objects present in both frame {frame_a} and frame {frame_b}?",
)

EARLIEST_OBJECT_TEMPLATES: Sequence[str] = (
    "Which object among {objects} shows up first in the video?",
    "Thinking back, which of {objects} is seen earliest?",
    "Within {objects}, which item appears before the others?",
    "Out of {objects}, which object leads the sequence of first sightings?",
)

FIRST_FRAME_TEMPLATES: Sequence[str] = (
    "At which frame index does {object} first appear?",
    "When do we first see {object} during the clip (frame number)?",
    "What is the frame ID of {object}'s earliest sighting?",
    "Identify the first frame that contains {object}.",
)

Option = List[str]
QuestionEntry = Dict[str, object]


def attach_episode_metadata(metadata: Dict[str, Any], episode_dir: Path) -> Dict[str, Any]:
    metadata = dict(metadata)
    metadata["episode_dir"] = str(episode_dir)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episode_dir",
        type=Path,
        nargs="?",
        help=(
            "Path to the directory that holds the per-agent frame dumps, e.g. "
            "outputs/video/train_15_Fridge_20251210_122305. If omitted or --auto-run is "
            "specified, a new episode will be recorded first."
        ),
    )
    parser.add_argument(
        "--auto-run",
        action="store_true",
        help="Execute a new ProcTHOR episode before generating questions.",
    )
    parser.add_argument(
        "--plan-output-base",
        type=Path,
        default=None,
        help="Directory where ProcTHOR frame dumps should be stored (default: executable_plan default).",
    )
    parser.add_argument(
        "--agent-prefix",
        default="agent_1",
        help="Prefix used for agent subdirectories (default: agent_1).",
    )
    parser.add_argument(
        "--min-objects",
        type=int,
        default=3,
        help="Minimum number of distinct objects to consider for ordering questions.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=4,
        help="Maximum number of distinct objects to consider for ordering questions.",
    )
    parser.add_argument(
        "--questions-per-size",
        type=int,
        default=8,
        help="Number of ordering questions to sample for each object set size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--cooccur-questions",
        type=int,
        default=8,
        help="Number of co-occurrence questions to sample from pairs of frames.",
    )
    parser.add_argument(
        "--cooccur-min-overlap",
        type=int,
        default=2,
        help="Minimum shared-object count required to form a co-occurrence question.",
    )
    parser.add_argument(
        "--cooccur-max-options",
        type=int,
        default=4,
        help="Maximum number of answer options for each co-occurrence question.",
    )
    parser.add_argument(
        "--cooccur-frame-gap",
        type=int,
        default=5,
        help="Minimum frame index gap between the two frames used in co-occurrence questions.",
    )
    parser.add_argument(
        "--earliest-questions",
        type=int,
        default=8,
        help="Number of earliest-object questions to sample for each object set size.",
    )
    parser.add_argument(
        "--first-frame-questions",
        type=int,
        default=12,
        help="Number of single-object first-frame questions to generate in total.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "memory",
        help="Directory where the generated QA JSON file will be stored.",
    )
    parser.add_argument(
        "--output-name",
        default="memory_order_questions.json",
        help="Filename for the aggregated QA file.",
    )
    parser.add_argument(
        "--episode-id",
        help="Unique identifier for this episode (stored in metadata and used for naming).",
    )
    parser.add_argument(
        "--conversation-prefix",
        help="Prefix for conversation IDs (defaults to --episode-id or the recorded run suffix).",
    )
    parser.add_argument(
        "--video-id",
        help="Basename to use for rendered video files (defaults to the conversation prefix).",
    )
    parser.add_argument(
        "--video-output-dir",
        type=Path,
        default=Path("outputs") / "memory" / "videos",
        help="Directory where rendered videos should be saved (agent/ and top_view/ subdirectories).",
    )
    parser.add_argument(
        "--plan-frame-rate",
        type=int,
        default=5,
        help="Frame rate used when rendering the episodic videos (passed to executable_plan).",
    )
    parser.add_argument(
        "--plan-done-repetitions",
        type=int,
        default=25,
        help="Number of Done action triplets to append before terminating the ProcTHOR episode.",
    )
    parser.add_argument(
        "--plan-nav-exclude",
        nargs="*",
        default=["Mug"],
        help="Object basenames to exclude when selecting the navigation target.",
    )
    parser.add_argument(
        "--plan-seed",
        type=int,
        help="Random seed forwarded to the ProcTHOR episode runner.",
    )
    parser.add_argument(
        "--plan-min-nav-distance",
        type=float,
        default=4.5,
        help="Minimum distance between the agent spawn locations and the navigation target (meters).",
    )
    parser.add_argument(
        "--plan-max-nav-iterations",
        type=int,
        default=400,
        help="Maximum navigation loop iterations before aborting (default: 400).",
    )
    return parser.parse_args()


def list_agent_json_files(agent_dir: Path) -> List[Path]:
    files = sorted(agent_dir.glob("img_*.json"))
    if not files:
        raise FileNotFoundError(f"No frame JSON files found under {agent_dir}")
    return files


def extract_items(itemlist: Iterable[str]) -> List[str]:
    names: List[str] = []
    for raw_entry in itemlist:
        if not raw_entry:
            continue
        name = raw_entry.split("|")[0].strip()
        if not name or name.lower() == "wall" or name.lower() == "room":
            continue
        names.append(name)
    return names


def load_frame_data(agent_dir: Path) -> List[Tuple[int, List[str]]]:
    frame_records: List[Tuple[int, List[str]]] = []
    for json_path in list_agent_json_files(agent_dir):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        frame_index = int(data.get("frame_index", 0))
        items = extract_items(data.get("itemlist", []))
        frame_records.append((frame_index, items))
    frame_records.sort(key=lambda entry: entry[0])
    return frame_records


def compute_first_appearance(frame_data: Sequence[Tuple[int, List[str]]]) -> Dict[str, int]:
    first_seen: Dict[str, int] = {}
    for frame_index, items in frame_data:
        for item in items:
            if item not in first_seen:
                first_seen[item] = frame_index
    return first_seen


def ensure_agent_dir(episode_dir: Path, agent_prefix: str) -> Path:
    candidates = sorted(episode_dir.glob(f"{agent_prefix}*"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate a directory starting with '{agent_prefix}' under {episode_dir}"
        )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No directory starting with '{agent_prefix}' was found under {episode_dir}"
    )


def format_object_list(names: Sequence[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def build_option_strings(orderings: Sequence[Sequence[str]]) -> Option:
    return [" -> ".join(order) for order in orderings]


def sample_distractors(
    combo: Sequence[str],
    correct_order: Sequence[str],
    rng: random.Random,
    max_options: int = 4,
) -> Option:
    all_perms = list(permutations(combo))
    correct_tuple = tuple(correct_order)
    rng.shuffle(all_perms)
    options: List[Tuple[str, ...]] = [correct_tuple]
    for perm in all_perms:
        if perm == correct_tuple:
            continue
        options.append(perm)
        if len(options) >= max_options:
            break
    rng.shuffle(options)
    return build_option_strings(options)


def choose_template(rng: random.Random, size: int) -> str:
    if size == 3:
        return rng.choice(THREE_OBJECT_TEMPLATES)
    if size == 4:
        return rng.choice(FOUR_OBJECT_TEMPLATES)
    raise ValueError(f"Unsupported object count for templates: {size}")


def sample_set_distractors(
    correct: Sequence[str],
    union_pool: Sequence[str],
    rng: random.Random,
    max_options: int = 4,
) -> Option:
    unique_pool = sorted(set(union_pool))
    if len(unique_pool) < len(correct):
        return []
    correct_tuple = tuple(sorted(correct))
    options: List[Tuple[str, ...]] = [correct_tuple]
    seen = {correct_tuple}
    attempts = 0
    while len(options) < max_options and attempts < 50:
        candidate = tuple(sorted(rng.sample(unique_pool, len(correct_tuple))))
        if candidate in seen:
            attempts += 1
            continue
        seen.add(candidate)
        options.append(candidate)
    if len(options) < 2:
        return []
    rng.shuffle(options)
    return [", ".join(option) for option in options]


def generate_ordering_questions(
    first_seen: Dict[str, int],
    size: int,
    rng: random.Random,
    limit: int,
    episode_dir: Path,
) -> List[QuestionEntry]:
    if size < 3:
        return []
    available_objects = [obj for obj, frame in first_seen.items() if frame is not None]
    combos = list(combinations(available_objects, size))
    rng.shuffle(combos)
    questions: List[QuestionEntry] = []

    for combo in combos:
        frame_pairs = sorted((obj, first_seen[obj]) for obj in combo)
        frames = [frame for _, frame in frame_pairs]
        if len(set(frames)) != size:
            continue
        correct_order = [obj for obj, _ in sorted(frame_pairs, key=lambda entry: entry[1])]
        option_strings = sample_distractors(combo, correct_order, rng)
        if len(option_strings) < 2:
            continue
        presented_names = list(combo)
        rng.shuffle(presented_names)
        template = choose_template(rng, size)
        question_text = template.format(objects=format_object_list(presented_names))
        answer = " -> ".join(correct_order)
        try:
            answer_index = option_strings.index(answer)
        except ValueError:
            # Ensure the correct ordering is always present.
            option_strings[0] = answer
            answer_index = 0
        metadata = attach_episode_metadata(
            {
            "question_type": "ordering",
            "object_first_frames": {obj: first_seen[obj] for obj in combo},
            "question_size": size,
            },
            episode_dir,
        )
        questions.append(
            {
                "question": question_text,
                "choices": option_strings,
                "answer_index": answer_index,
                "answer": answer,
                "metadata": metadata,
            }
        )
        if len(questions) >= limit:
            break
    return questions


def generate_cooccurrence_questions(
    frame_data: Sequence[Tuple[int, Sequence[str]]],
    rng: random.Random,
    limit: int,
    all_objects: Sequence[str],
    episode_dir: Path,
    min_overlap: int = 2,
    max_options: int = 4,
    min_frame_gap: int = 5,
) -> List[QuestionEntry]:
    if limit <= 0:
        return []
    frame_pairs = list(combinations(frame_data, 2))
    rng.shuffle(frame_pairs)
    questions: List[QuestionEntry] = []

    for (frame_a, items_a), (frame_b, items_b) in frame_pairs:
        if abs(frame_a - frame_b) < min_frame_gap:
            continue
        shared_objects = sorted(set(items_a) & set(items_b))
        if len(shared_objects) < min_overlap:
            continue
        union_pool = list(set(items_a) | set(items_b) | set(all_objects))
        option_strings = sample_set_distractors(shared_objects, union_pool, rng, max_options)
        if len(option_strings) < 2:
            continue
        template = rng.choice(CO_OCCURRENCE_TEMPLATES)
        question_text = template.format(frame_a=frame_a, frame_b=frame_b)
        answer = ", ".join(shared_objects)
        try:
            answer_index = option_strings.index(answer)
        except ValueError:
            option_strings[0] = answer
            answer_index = 0
        metadata = attach_episode_metadata(
            {
                "question_type": "cooccurrence",
                "frame_pair": [frame_a, frame_b],
                "shared_objects": shared_objects,
            },
            episode_dir,
        )
        questions.append(
            {
                "question": question_text,
                "choices": option_strings,
                "answer_index": answer_index,
                "answer": answer,
                "metadata": metadata,
            }
        )
        if len(questions) >= limit:
            break
    return questions


def generate_earliest_object_questions(
    first_seen: Dict[str, int],
    size: int,
    rng: random.Random,
    limit: int,
    episode_dir: Path,
) -> List[QuestionEntry]:
    if limit <= 0 or size < 2:
        return []
    available_objects = [obj for obj in first_seen if first_seen[obj] is not None]
    combos = list(combinations(available_objects, size))
    rng.shuffle(combos)
    questions: List[QuestionEntry] = []

    for combo in combos:
        frame_pairs = [(obj, first_seen[obj]) for obj in combo]
        min_frame = min(frame for _, frame in frame_pairs)
        earliest_candidates = [obj for obj, frame in frame_pairs if frame == min_frame]
        if len(earliest_candidates) != 1:
            continue
        earliest_obj = earliest_candidates[0]
        option_names = list(combo)
        rng.shuffle(option_names)
        template = rng.choice(EARLIEST_OBJECT_TEMPLATES)
        question_text = template.format(objects=format_object_list(option_names))
        answer = earliest_obj
        try:
            answer_index = option_names.index(answer)
        except ValueError:
            continue
        metadata = attach_episode_metadata(
            {
                "question_type": "earliest_object",
                "object_first_frames": {obj: first_seen[obj] for obj in combo},
                "question_size": size,
                "earliest_object": earliest_obj,
            },
            episode_dir,
        )
        questions.append(
            {
                "question": question_text,
                "choices": option_names,
                "answer_index": answer_index,
                "answer": answer,
                "metadata": metadata,
            }
        )
        if len(questions) >= limit:
            break
    return questions


def sample_frame_options(
    correct_frame: int,
    all_frames: Sequence[int],
    rng: random.Random,
    max_options: int = 4,
    min_gap: int = 3,
) -> Option:
    def is_valid(candidate: int, existing: Sequence[int]) -> bool:
        return all(abs(candidate - value) >= min_gap for value in existing)

    options: List[int] = [correct_frame]
    pool = [frame for frame in sorted(set(all_frames)) if frame != correct_frame]
    rng.shuffle(pool)
    while pool and len(options) < max_options:
        candidate = pool.pop()
        if candidate not in options and is_valid(candidate, options):
            options.append(candidate)

    offset = 1
    attempts = 0
    while len(options) < max_options and attempts < 20:
        direction = -1 if rng.random() < 0.5 else 1
        candidate = correct_frame + direction * offset
        offset += 1
        attempts += 1
        if candidate < 0 or candidate in options or not is_valid(candidate, options):
            continue
        options.append(candidate)

    rng.shuffle(options)
    return [str(value) for value in options]


def generate_first_frame_questions(
    first_seen: Dict[str, int],
    rng: random.Random,
    limit: int,
    episode_dir: Path,
    max_options: int = 4,
) -> List[QuestionEntry]:
    if limit <= 0:
        return []
    objects = list(first_seen.items())
    rng.shuffle(objects)
    all_frames = list(first_seen.values())
    questions: List[QuestionEntry] = []

    for obj_name, frame_index in objects:
        if frame_index is None:
            continue
        option_strings = sample_frame_options(frame_index, all_frames, rng, max_options)
        if len(option_strings) < 2:
            continue
        answer = str(frame_index)
        try:
            answer_index = option_strings.index(answer)
        except ValueError:
            continue
        template = rng.choice(FIRST_FRAME_TEMPLATES)
        question_text = template.format(object=obj_name)
        metadata = attach_episode_metadata(
            {
                "question_type": "first_frame",
                "object": obj_name,
                "answer_frame": frame_index,
            },
            episode_dir,
        )
        questions.append(
            {
                "question": question_text,
                "choices": option_strings,
                "answer_index": answer_index,
                "answer": answer,
                "metadata": metadata,
            }
        )
        if len(questions) >= limit:
            break
    return questions


def collect_video_paths(video_map: Dict[str, str]) -> List[Path]:
    ordered_keys = ["agent", "top_view"]
    paths: List[Path] = []
    seen: set[Path] = set()
    for key in ordered_keys:
        value = video_map.get(key)
        if value:
            candidate = Path(value)
            if candidate not in seen:
                seen.add(candidate)
                paths.append(candidate)
    for key, value in video_map.items():
        if key in ordered_keys:
            continue
        candidate = Path(value)
        if candidate not in seen:
            seen.add(candidate)
            paths.append(candidate)
    return paths


def infer_existing_videos(episode_dir: Path) -> List[Path]:
    run_root = episode_dir
    patterns = ("video_agent*.mp4", "video_top_view*.mp4", "*.mp4")
    discovered: List[Path] = []
    for pattern in patterns:
        discovered.extend(sorted(run_root.glob(pattern)))
    unique: List[Path] = []
    seen = set()
    for path in discovered:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def relative_video_paths(video_paths: Sequence[Path], base_dir: Path) -> List[str]:
    relative: List[str] = []
    for path in video_paths:
        try:
            rel = path.relative_to(base_dir)
            relative.append(rel.as_posix())
        except ValueError:
            relative.append(path.as_posix())
    return relative


def build_conversation_entry(
    question: QuestionEntry,
    video_refs: Sequence[str],
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    raw_choices: Sequence[str] = question["choices"]  # type: ignore[index]
    answer_index: int = question["answer_index"]  # type: ignore[index]
    labelled_choices = [
        f"{chr(ord('A') + idx)}.{choice}"
        for idx, choice in enumerate(raw_choices)
    ]
    question_text = str(question["question"]).strip()
    human_value = f"<video><video>{question_text} " + " ".join(labelled_choices)

    if not (0 <= answer_index < len(raw_choices)):
        raise ValueError("Answer index is out of range for the provided choices.")

    answer_label = chr(ord('A') + answer_index)
    gpt_value = f"{answer_label}.{raw_choices[answer_index]}"

    entry: Dict[str, Any] = {
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": gpt_value},
        ],
        "videos": list(video_refs),
    }

    metadata = question.get("metadata")
    if isinstance(metadata, dict) and metadata:
        entry["metadata"] = dict(metadata)
    if conversation_id:
        entry["id"] = conversation_id

    return entry


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    requested_episode_id = args.episode_id.strip() if args.episode_id else None
    requested_conversation_prefix = args.conversation_prefix.strip() if args.conversation_prefix else None
    requested_video_id = args.video_id.strip() if args.video_id else None

    output_dir = args.output_dir.resolve()
    video_output_dir = args.video_output_dir.resolve()
    plan_output_base = args.plan_output_base.resolve() if args.plan_output_base else None

    episode_dir: Optional[Path] = args.episode_dir.resolve() if args.episode_dir else None
    video_paths: List[Path] = []
    run_result: Optional[Dict[str, Any]] = None

    if args.auto_run or episode_dir is None:
        run_kwargs: Dict[str, Any] = {
            "video_output_dir": video_output_dir,
            "agent_prefix": args.agent_prefix,
            "frame_rate": args.plan_frame_rate,
            "done_repetitions": args.plan_done_repetitions,
            "nav_exclude": args.plan_nav_exclude,
            "random_seed": args.plan_seed,
            "min_nav_distance": args.plan_min_nav_distance,
            "max_nav_iterations": args.plan_max_nav_iterations,
        }
        if plan_output_base is not None:
            run_kwargs["output_base"] = plan_output_base
        video_filename_stem = requested_video_id or requested_conversation_prefix or requested_episode_id
        if video_filename_stem:
            run_kwargs["video_filename_stem"] = video_filename_stem
        run_result = run_episode(**run_kwargs)
        episode_dir = Path(run_result["run_output_root"]).resolve()
        video_paths = collect_video_paths(run_result.get("video_paths", {}))
        if not video_paths:
            raise RuntimeError("No videos were produced by the ProcTHOR episode run.")
    else:
        episode_dir = episode_dir.resolve()
        video_paths = infer_existing_videos(episode_dir)
        if not video_paths:
            raise FileNotFoundError(
                "Unable to locate MP4 videos for the provided episode directory. "
                "Run with --auto-run or supply an episode that already has videos rendered."
            )

    if episode_dir is None:
        raise RuntimeError("Episode directory was not resolved.")

    agent_dir = ensure_agent_dir(episode_dir, args.agent_prefix)
    frame_data = load_frame_data(agent_dir)
    first_seen = compute_first_appearance(frame_data)

    if len(first_seen) < args.min_objects:
        raise RuntimeError(
            f"Only {len(first_seen)} usable objects detected; need at least {args.min_objects}."
        )

    run_suffix = run_result.get("run_suffix") if run_result else None
    episode_identifier = requested_episode_id or run_suffix
    if episode_identifier is None:
        episode_identifier = episode_dir.name
    conversation_prefix = requested_conversation_prefix or episode_identifier or run_suffix
    video_identifier = requested_video_id
    if video_identifier is None and run_result:
        video_identifier = run_result.get("video_filename_stem")
    if video_identifier is None:
        video_identifier = conversation_prefix

    all_questions: List[QuestionEntry] = []
    for size in range(args.min_objects, args.max_objects + 1):
        questions = generate_ordering_questions(first_seen, size, rng, args.questions_per_size, episode_dir)
        if questions:
            all_questions.extend(questions)

        earliest_questions = generate_earliest_object_questions(first_seen, size, rng, args.earliest_questions, episode_dir)
        if earliest_questions:
            all_questions.extend(earliest_questions)

    if args.cooccur_questions > 0:
        cooccur = generate_cooccurrence_questions(
            frame_data,
            rng,
            args.cooccur_questions,
            list(first_seen.keys()),
            episode_dir,
            min_overlap=args.cooccur_min_overlap,
            max_options=args.cooccur_max_options,
            min_frame_gap=args.cooccur_frame_gap,
        )
        if not cooccur and args.cooccur_min_overlap > 1:
            cooccur = generate_cooccurrence_questions(
                frame_data,
                rng,
                args.cooccur_questions,
                list(first_seen.keys()),
                episode_dir,
                min_overlap=max(1, args.cooccur_min_overlap - 1),
                max_options=args.cooccur_max_options,
                min_frame_gap=args.cooccur_frame_gap,
            )
        if cooccur:
            all_questions.extend(cooccur)

    if args.first_frame_questions > 0:
        first_frame = generate_first_frame_questions(first_seen, rng, args.first_frame_questions, episode_dir)
        if first_frame:
            all_questions.extend(first_frame)

    if not all_questions:
        raise RuntimeError("No questions could be generated with the current configuration.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    video_refs = relative_video_paths([path.resolve() for path in video_paths], output_dir)
    common_metadata: Dict[str, Any] = {
        "episode_dir": str(episode_dir),
        "video_refs": list(video_refs),
    }
    if episode_identifier:
        common_metadata["episode_id"] = episode_identifier
    if video_identifier:
        common_metadata["video_id"] = video_identifier
    if run_result:
        run_info = run_result.get("run_info") or {}
        if run_info:
            common_metadata["run_info"] = run_info
        nav_target = run_result.get("random_nav_target")
        if nav_target:
            common_metadata["nav_target"] = nav_target
        video_stem = run_result.get("video_filename_stem")
        if video_stem:
            common_metadata["video_filename_stem"] = video_stem
        index_value = run_result.get("video_index")
        if index_value is not None:
            common_metadata["video_index"] = index_value

    for question in all_questions:
        metadata = dict(question.get("metadata", {}))
        metadata.update(common_metadata)
        question["metadata"] = metadata

    conversations: List[Dict[str, Any]] = []
    for idx, question in enumerate(all_questions, start=1):
        conversation_id = None
        if conversation_prefix:
            conversation_id = f"{conversation_prefix}_q{idx:04d}"
        conversations.append(build_conversation_entry(question, video_refs, conversation_id))
    output_path.write_text(json.dumps(conversations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(conversations)} conversation entries to {output_path}")


if __name__ == "__main__":
    main()
