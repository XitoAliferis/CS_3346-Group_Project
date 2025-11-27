import json
from pathlib import Path

def save_jsonl(data, filename):
    if not filename:
        raise ValueError("Filename cannot be empty.")
    data_dir = Path(__file__).resolve().parent.parent.parent / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / f"{filename}.jsonl"

    with open(file_path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(data)} rows to {file_path}")

def load_jsonl(path):
    with open(path+'.jsonl', "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]