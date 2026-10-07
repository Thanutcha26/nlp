import argparse
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
MODEL_ID = "typhoon-ai/llama3.2-typhoon2-3b-instruct"
MODES = {
    "rewrite": "โปรดเรียบเรียงข้อความต่อไปนี้ใหม่เป็นภาษาไทย โดยคงสาระสำคัญและข้อเท็จจริงเดิมให้ครบถ้วน ใช้สำนวนที่เป็นธรรมชาติ อ่านลื่นไหล และอย่าสรุปหรือตัดรายละเอียด",
    "formal": "โปรดเรียบเรียงข้อความต่อไปนี้ใหม่เป็นภาษาไทยระดับทางการ สุภาพและเป็นธรรมชาติ โดยคงสาระสำคัญและข้อเท็จจริงเดิมให้ครบถ้วน อย่าสรุปหรือตัดรายละเอียด",
    "semiformal": "โปรดเรียบเรียงข้อความต่อไปนี้ใหม่เป็นภาษาไทยกึ่งทางการที่สุภาพ เป็นธรรมชาติ และอ่านเข้าใจง่าย โดยคงสาระสำคัญและข้อเท็จจริงเดิมให้ครบถ้วน",
    "concise": "โปรดเรียบเรียงข้อความต่อไปนี้ใหม่ให้กระชับ อ่านลื่นไหล โดยรักษาข้อเท็จจริงและสาระสำคัญไว้ให้ครบ",
}
MODE_LABELS = {
    "rewrite": "ใหม่",
    "formal": "เป็นภาษาทางการ",
    "semiformal": "เป็นภาษากึ่งทางการ",
    "concise": "ให้กระชับ",
}
PROTECTED_TERMS = (
    "พรุ่งนี้", "วันนี้", "เมื่อวาน", "มะรืน", "เที่ยงคืน", "บาท", "เปอร์เซ็นต์",
)


def validate_output(source: str, output: str) -> str | None:
    candidate = output.strip()
    if not candidate:
        return "โมเดลไม่ได้สร้างข้อความ"
    if any(marker in candidate for marker in ("<extra_id_", "instruction:", "context:", "โปรดเรียบเรียง")):
        return "ผลลัพธ์มี token พิเศษหรือข้อความคำสั่งปะปน"
    if len(candidate) > max(len(source) * 2, len(source) + 120):
        return "ผลลัพธ์ยาวผิดปกติและอาจวนซ้ำ"

    number_pattern = r"\d+(?:[.,]\d+)*|[๐-๙]+"

    def normalize_number(value: str) -> str:
        thai_digits = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
        normalized = value.translate(thai_digits).replace(",", "")
        if normalized.endswith(".00"):
            normalized = normalized[:-3]
        return normalized

    source_numbers = [normalize_number(value) for value in re.findall(number_pattern, source)]
    output_numbers = [normalize_number(value) for value in re.findall(number_pattern, candidate)]
    if "เที่ยง" in source and "เที่ยงคืน" not in source:
        source_numbers.append("12")
    if source_numbers != output_numbers:
        return "ตัวเลขหรือลำดับตัวเลขในผลลัพธ์ไม่ตรงกับต้นฉบับ"

    source_terms = {term for term in PROTECTED_TERMS if term in source}
    output_terms = {term for term in PROTECTED_TERMS if term in candidate}
    if source_terms != output_terms:
        return "คำบอกเวลา/วันที่หรือหน่วยสำคัญเปลี่ยนไปจากต้นฉบับ"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="เรียบเรียงข้อความภาษาไทยด้วยโมเดลที่ fine-tune แล้ว")
    parser.add_argument("--mode", choices=MODES, default="rewrite")
    parser.add_argument("--text", help="ข้อความต้นฉบับ (ถ้าไม่ระบุจะถามใน terminal)")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    text = (args.text or input("ข้อความต้นฉบับ: ")).strip()
    if not text:
        parser.error("กรุณาระบุข้อความด้วย --text หรือพิมพ์ข้อความเมื่อโปรแกรมถาม")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    messages = [
        {
            "role": "system",
            "content": "คุณเป็นผู้ช่วยเรียบเรียงภาษาไทย ให้เขียนใหม่โดยรักษาข้อมูลเดิมทุกประการ ห้ามเพิ่มเวลา ตัวเลข หรือข้อเท็จจริง และตอบเฉพาะข้อความที่เรียบเรียงแล้ว",
        },
        {"role": "user", "content": f"ช่วยเรียบเรียง{MODE_LABELS[args.mode]}: {text}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=4,
        )

    result = tokenizer.decode(
        generated[0, inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip().split("\n\n", 1)[0]
    rejection_reason = validate_output(text, result)
    print("\nผลลัพธ์:")
    if rejection_reason:
        print(f"[ป้องกันการแสดงผลที่อาจผิดความหมาย: {rejection_reason}]")
        print("ใช้ข้อความต้นฉบับแทน:")
        print(text)
    else:
        print(result)


if __name__ == "__main__":
    main()