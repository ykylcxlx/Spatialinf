python - <<'PY'
import json, pathlib, prior
dataset = prior.load_dataset("procthor-10k")
house = dataset["train"][0]                    # 任选一个场景
path = pathlib.Path("procthordata/houses/train_000.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(house), encoding="utf-8")
print(path.resolve())
PY