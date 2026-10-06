import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
DEFAULT_MODEL = "google/mt5-small"


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
    parser = argparse.ArgumentParser(description="Fine-tune mT5 on prepared Thai rewriting pairs")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 uses the full training split")
    parser.add_argument("--source-max-length", type=int, default=512)
    parser.add_argument("--target-max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = DATA_DIR / "train.jsonl"
    validation_path = DATA_DIR / "validation.jsonl"
    for path in (train_path, validation_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run prepare_data.py first")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        prompts = [
            f"instruction: {instruction}\ncontext: {source}"
            for instruction, source in zip(batch["instruction"], batch["source_text"])
        ]
        encoded = tokenizer(
            prompts,
            max_length=args.source_max_length,
            truncation=True,
        )
        encoded["labels"] = tokenizer(
            text_target=batch["target_text"],
            max_length=args.target_max_length,
            truncation=True,
        )["input_ids"]
        return encoded

    train_dataset = Dataset.from_list(train_rows).map(
        tokenize, batched=True, remove_columns=Dataset.from_list(train_rows).column_names
    )
    validation_dataset = Dataset.from_list(validation_rows).map(
        tokenize,
        batched=True,
        remove_columns=Dataset.from_list(validation_rows).column_names,
    )

    ARTIFACT_DIR.mkdir(exist_ok=True)
    checkpoint_dir = ARTIFACT_DIR / "checkpoints"
    model_dir = ARTIFACT_DIR / "model"
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_dir),
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        predict_with_generate=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        logging_steps=25,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        processing_class=tokenizer,
    )

    result = trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)
    trainer.save_state()

    run_report = {
        "model": args.model_name,
        "fine_tuned_model_path": str(model_dir.relative_to(ROOT)),
        "training_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "epochs": args.epochs,
        "source_max_length": args.source_max_length,
        "target_max_length": args.target_max_length,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "dataset_sha256": {
            "train.jsonl": sha256_file(train_path),
            "validation.jsonl": sha256_file(validation_path),
        },
        "train_metrics": result.metrics,
        "validation_metrics": trainer.evaluate(),
        "training_log": trainer.state.log_history,
        "warning": "This training set uses summarization outputs as weak rewriting targets; it does not supervise controlled formality or guarantee meaning preservation.",
    }
    with (ARTIFACT_DIR / "training_report.json").open("w", encoding="utf-8") as output:
        json.dump(run_report, output, ensure_ascii=False, indent=2, default=str)
        output.write("\n")

    print(f"Model saved to: {model_dir}")
    print(f"Training evidence: {ARTIFACT_DIR / 'training_report.json'}")


if __name__ == "__main__":
    main()