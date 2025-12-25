
import json
import pathlib

import prior

dataset = prior.load_dataset("procthor-10k")

root = pathlib.Path("houses")
root.mkdir(parents=True, exist_ok=True)

print("Dataset type:", type(dataset))
available_attrs = [attr for attr in dir(dataset) if not attr.startswith("__")]
print("Available attributes:", available_attrs)

split_names: list[str] = []
if hasattr(dataset, "split_names") and dataset.split_names:
	split_names = list(dataset.split_names)
else:
	for candidate in ("train", "val", "validation", "test", "dev"):
		if hasattr(dataset, candidate):
			split_names.append(candidate)

if not split_names:
	raise RuntimeError("Could not determine dataset splits from available attributes.")

print("Detected splits:", split_names)
saved = 0
for split_name in split_names:
	scenes = getattr(dataset, split_name)
	split_dir = root / split_name
	split_dir.mkdir(parents=True, exist_ok=True)
	for index, house in enumerate(scenes):
		scene_id = house.get("scene") or house.get("scene_id") or f"{split_name}_{index:05d}"
		file_path = split_dir / f"{scene_id}.json"
		file_path.write_text(json.dumps(house), encoding="utf-8")
		saved += 1

print(f"Saved {saved} scenes under {root.resolve()}")