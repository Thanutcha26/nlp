import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts_llama"
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B" 

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
    parser = argparse.ArgumentParser(description="Fine-tune Llama on prepared Thai rewriting pairs")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=1024) 
    parser.add_argument("--batch-size", type=int, default=2)
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # โหลดโมเดลลง GPU อัตโนมัติด้วย device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    
    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]

    def prepare_dataset(batch):
        texts = []
        for inst, src, tgt in zip(batch['instruction'], batch['source_text'], batch['target_text']):
            texts.append(f"Instruction: {inst}\nContext: {src}\nAnswer: {tgt}{tokenizer.eos_token}")
        
        tokenized = tokenizer(texts, truncation=True, max_length=args.max_seq_length)
        return tokenized

    train_dataset = Dataset.from_list(train_rows).map(
        prepare_dataset, batched=True, remove_columns=Dataset.from_list(train_rows).column_names
    )
    validation_dataset = Dataset.from_list(validation_rows).map(
        prepare_dataset, batched=True, remove_columns=Dataset.from_list(validation_rows).column_names
    )

    ARTIFACT_DIR.mkdir(exist_ok=True)
    checkpoint_dir = ARTIFACT_DIR / "checkpoints"
    model_dir = ARTIFACT_DIR / "model"
    
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=1,
        logging_steps=25,
        report_to="none",
        seed=args.seed,
        
        # --- ตั้งค่าสำหรับ GPU ---
        fp16=True,                    # เปิดใช้ 16-bit precision
        gradient_checkpointing=True,  # ประหยัด VRAM ป้องกันการ์ดจอแรมเต็ม
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("Starting training on GPU...")
    result = trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)

    run_report = {
        "model": args.model_name,
        "fine_tuned_model_path": str(model_dir.relative_to(ROOT)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_metrics": result.metrics,
        "dataset_sha256": {
            "train.jsonl": sha256_file(train_path)
        }
    }
    with (ARTIFACT_DIR / "training_report.json").open("w", encoding="utf-8") as output:
        json.dump(run_report, output, ensure_ascii=False, indent=2, default=str)

    print(f"Model saved to: {model_dir}")

if __name__ == "__main__":
    main()