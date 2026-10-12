import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch_directml
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts_llama"
DEFAULT_MODEL = "typhoon-ai/llama3.2-typhoon2-1b-instruct"

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def collate_batch(rows: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_length = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full((len(rows), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
    labels = torch.full((len(rows), max_length), -100, dtype=torch.long)
    for index, row in enumerate(rows):
        row_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        row_labels = torch.tensor(row["labels"], dtype=torch.long)
        input_ids[index, : len(row_ids)] = row_ids
        attention_mask[index, : len(row_ids)] = 1
        labels[index, : len(row_labels)] = row_labels
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def evaluate(model: torch.nn.Module, batches: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_total = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch in tqdm(batches, desc="Validation"):
            inputs = {key: value.to(device) for key, value in batch.items()}
            loss_total += float(model(**inputs).loss.cpu())
            batch_count += 1
    model.train()
    return loss_total / max(1, batch_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Typhoon Instruct with LoRA on an AMD DirectML GPU")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR / "typhoon2-1b-lora")
    args = parser.parse_args()

    train_path = DATA_DIR / "train.jsonl"
    validation_path = DATA_DIR / "validation.jsonl"
    for path in (train_path, validation_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run prepare_data.py first")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch_directml.device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.to(device)
    model.print_trainable_parameters()

    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_validation_samples > 0:
        validation_rows = validation_rows[: args.max_validation_samples]

    def tokenize(rows: list[dict[str, Any]]) -> list[dict[str, list[int]]]:
        encoded = []
        for row in rows:
            prompt = (
                f"Instruction: {row['instruction']}\n"
                f"Context: {row['source_text']}\nAnswer: "
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            target_ids = tokenizer(
                row["target_text"] + tokenizer.eos_token,
                add_special_tokens=False,
            )["input_ids"]
            target_ids = target_ids[: max(1, args.max_seq_length // 2)]
            prompt_ids = prompt_ids[: args.max_seq_length - len(target_ids)]
            input_ids = prompt_ids + target_ids
            encoded.append(
                {
                    "input_ids": input_ids,
                    "labels": [-100] * len(prompt_ids) + target_ids,
                }
            )
        return encoded

    pad_token_id = tokenizer.pad_token_id
    train_batches = DataLoader(
        tokenize(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate_batch(rows, pad_token_id),
    )
    validation_batches = DataLoader(
        tokenize(validation_rows),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda rows: collate_batch(rows, pad_token_id),
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)
    training_losses = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_batches, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            inputs = {key: value.to(device) for key, value in batch.items()}
            loss = model(**inputs).loss
            (loss / args.gradient_accumulation_steps).backward()
            epoch_loss += float(loss.detach().cpu())
            is_update_step = (step + 1) % args.gradient_accumulation_steps == 0
            is_last_step = step + 1 == len(train_batches)
            if is_update_step or is_last_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        training_losses.append(epoch_loss / max(1, len(train_batches)))

    validation_loss = evaluate(model, validation_batches, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_state = {
        key: value.detach().cpu().contiguous()
        for key, value in get_peft_model_state_dict(model).items()
    }
    model.save_pretrained(args.output_dir, state_dict=adapter_state)
    tokenizer.save_pretrained(args.output_dir)

    report = {
        "base_model": args.model_name,
        "adapter_path": str(args.output_dir.relative_to(ROOT))
        if args.output_dir.is_relative_to(ROOT)
        else str(args.output_dir),
        "device": str(device),
        "training_method": "LoRA",
        "training_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "epochs": args.epochs,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "dataset_sha256": {
            "train.jsonl": sha256_file(train_path),
            "validation.jsonl": sha256_file(validation_path),
        },
        "train_loss_by_epoch": training_losses,
        "validation_loss": validation_loss,
        "loss_scope": "Summary target tokens only; instruction and context tokens are masked.",
        "note": "Only the LoRA adapter is saved; load it together with the base model.",
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    report_path = args.output_dir / "training_report.json"
    with report_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")

    print(f"LoRA adapter saved to: {args.output_dir}")
    print(f"Training report: {report_path}")


if __name__ == "__main__":
    main()