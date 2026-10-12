import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch_directml
from peft import PeftModel
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import CHRF
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
TEST_FILE = ROOT / "data" / "test.jsonl"
DEFAULT_MODEL = "typhoon-ai/llama3.2-typhoon2-1b-instruct"
DEFAULT_ADAPTER_DIR = ROOT / "artifacts_llama" / "typhoon2-1b-lora"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Typhoon LoRA adapter on the Thai test split")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--use-cache", action="store_true", help="Use the KV cache to speed up generation")
    args = parser.parse_args()

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Missing test split: {TEST_FILE}")
    if not (args.adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing LoRA adapter in {args.adapter_dir}; run train_llama.py first")

    rows = read_jsonl(TEST_FILE)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    device = torch_directml.device()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.to(device)
    model.eval()

    predictions: list[str] = []
    for row in tqdm(rows, desc="Evaluating Typhoon LoRA"):
        prompt = (
            f"Instruction: {row['instruction']}\n"
            f"Context: {row['source_text']}\n"
            "Answer:"
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=args.use_cache,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[1]
        predictions.append(
            tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
        )

    references = [row["target_text"] for row in rows]
    chrf = CHRF().corpus_score(predictions, [references]).score
    rouge = RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l_f1 = sum(
        rouge.score(reference, prediction)["rougeL"].fmeasure
        for reference, prediction in zip(references, predictions)
    ) / max(1, len(references))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as output:
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
        "base_model": args.model_name,
        "adapter_path": str(args.adapter_dir),
        "test_examples": len(rows),
        "test_sha256": sha256_file(TEST_FILE),
        "metrics": {"chrf": chrf, "rougeL_f1": rouge_l_f1},
        "metric_interpretation": "Lexical overlap with synthetic summarization targets only; these metrics do not measure factual faithfulness, meaning preservation, or style control.",
        "predictions_file": str(predictions_path),
    }
    report_path = args.output_dir / "evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()