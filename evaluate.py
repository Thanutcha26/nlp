import argparse
import json
from pathlib import Path
from typing import Any

import torch
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
TEST_FILE = ROOT / "data" / "test.jsonl"
MODEL_DIR = ROOT / "artifacts" / "model"
ARTIFACT_DIR = ROOT / "artifacts"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned model on held-out examples")
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates the full test split")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if not TEST_FILE.exists():
        raise FileNotFoundError("Missing data/test.jsonl; run prepare_data.py first")
    if not MODEL_DIR.exists():
        raise FileNotFoundError("Missing artifacts/model; run train.py first")

    rows = read_jsonl(TEST_FILE)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions: list[str] = []

    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        prompts = [
            f"instruction: {row['instruction']}\ncontext: {row['source_text']}"
            for row in batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=4,
                do_sample=False,
            )
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    references = [row["target_text"] for row in rows]
    chrf = CHRF().corpus_score(predictions, [references]).score
    rouge = RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l_f1 = sum(
        rouge.score(reference, prediction)["rougeL"].fmeasure
        for reference, prediction in zip(references, predictions)
    ) / max(1, len(references))

    ARTIFACT_DIR.mkdir(exist_ok=True)
    prediction_path = ARTIFACT_DIR / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as output:
        for row, prediction in zip(rows, predictions):
            output.write(
                json.dumps(
                    {
                        "source_text": row["source_text"],
                        "reference_text": row["target_text"],
                        "model_output": prediction,
                        "source_index": row["source_index"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = {
        "model_path": str(MODEL_DIR.relative_to(ROOT)),
        "test_examples": len(rows),
        "metrics": {
            "chrf": chrf,
            "rougeL_f1": rouge_l_f1,
        },
        "metric_interpretation": "Lexical overlap with synthetic summarization targets only; these metrics do not measure factual faithfulness, meaning preservation, or style control.",
        "predictions_file": str(prediction_path.relative_to(ROOT)),
    }
    with (ARTIFACT_DIR / "evaluation_report.json").open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()