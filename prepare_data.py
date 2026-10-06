import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATASET_ID = "airesearch/wangchanx-seed-free-synthetic-instruct-thai-120k"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
SEED = 42
MIN_RATING = 7.0
MIN_LENGTH_RATIO = 0.65
MAX_LENGTH_RATIO = 1.25
TEST_FRACTION = 0.10
VALIDATION_FRACTION = 0.10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    shuffled = records.copy()
    random.Random(SEED).shuffle(shuffled)
    test_count = round(len(shuffled) * TEST_FRACTION)
    validation_count = round(len(shuffled) * VALIDATION_FRACTION)
    return {
        "test": shuffled[:test_count],
        "validation": shuffled[test_count : test_count + validation_count],
        "train": shuffled[test_count + validation_count :],
    }


def main() -> None:
    dataset = load_dataset(DATASET_ID, split="train")
    accepted: list[dict[str, Any]] = []
    rejected = Counter()
    seen_sources: set[str] = set()
    ratios: list[float] = []

    for source_index, row in enumerate(dataset):
        if row.get("type") != "summarization":
            rejected["not_summarization"] += 1
            continue

        source = (row.get("context") or "").strip()
        target = (row.get("output") or "").strip()
        if not source or not target:
            rejected["missing_source_or_target"] += 1
            continue

        rating = float(row.get("rating") or 0)
        if rating < MIN_RATING:
            rejected["rating_below_threshold"] += 1
            continue

        ratio = len(target) / len(source)
        if not MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
            rejected["length_ratio_out_of_range"] += 1
            continue

        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if source_hash in seen_sources:
            rejected["duplicate_source"] += 1
            continue
        seen_sources.add(source_hash)
        ratios.append(ratio)
        accepted.append(
            {
                "source_text": source,
                "target_text": target,
                "instruction": row.get("instruction") or "",
                "rating": rating,
                "source_index": source_index,
            }
        )

    if len(accepted) < 20:
        raise ValueError(f"Only {len(accepted)} examples passed the filters; refusing to create splits")

    DATA_DIR.mkdir(exist_ok=True)
    splits = split_records(accepted)
    output_files: dict[str, dict[str, Any]] = {}
    for split_name, split_rows in splits.items():
        path = DATA_DIR / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as output:
            for record in split_rows:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_files[path.name] = {
            "rows": len(split_rows),
            "sha256": sha256_file(path),
        }

    manifest = {
        "project": "Thai text rewriting prototype",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "id": DATASET_ID,
            "url": DATASET_URL,
            "license": "MIT (as stated on the Hugging Face dataset card)",
            "source_split": "train",
            "source_rows": len(dataset),
            "dataset_fingerprint": dataset._fingerprint,
            "citation": "Pengpun et al., ACL SRW 2024, https://aclanthology.org/2024.acl-srw.38",
        },
        "selection": {
            "task_type": "summarization",
            "minimum_rating": MIN_RATING,
            "target_to_source_character_ratio": [MIN_LENGTH_RATIO, MAX_LENGTH_RATIO],
            "remove_duplicate_sources": True,
            "duplicate_key": "SHA-256 of trimmed source context",
            "source_field": "context",
            "target_field": "output",
            "note": "Weakly supervised rewriting pairs; the dataset labels these examples as summarization, not paraphrase or style transfer.",
            "accepted_rows": len(accepted),
            "rejected_rows_by_reason": dict(sorted(rejected.items())),
            "median_target_to_source_character_ratio": median(ratios),
        },
        "split": {
            "seed": SEED,
            "method": "deterministic shuffle, then 80/10/10 train/validation/test",
            "files": output_files,
        },
        "known_limitations": [
            "Targets may summarize, omit details, or introduce unsupported facts; length filtering does not guarantee semantic equivalence.",
            "The source dataset has no labels for formal, casual, or other Thai style levels.",
            "The source dataset is synthetic and may contain factual errors or model-generated biases.",
        ],
    }
    with (DATA_DIR / "manifest.json").open("w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")

    print(f"Source rows: {len(dataset)}")
    print(f"Selected weakly supervised pairs: {len(accepted)}")
    print("Splits: " + ", ".join(f"{name}={len(rows)}" for name, rows in splits.items()))
    print("Data and provenance manifest: data/")


if __name__ == "__main__":
    main()