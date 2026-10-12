# ต้นแบบสรุปข้อความภาษาไทย

โครงงานนี้เตรียมข้อมูลจาก [WangchanX Seed-Free Synthetic Instruct Thai 120k](https://huggingface.co/datasets/airesearch/wangchanx-seed-free-synthetic-instruct-thai-120k) สำหรับงานสรุปแบบ weak supervision มี baseline `google/mt5-small` และการทดลอง fine-tune Typhoon2 1B ด้วย LoRA ปัจจุบัน CLI ใช้การคัดประโยคภาษาไทยจากต้นฉบับเพื่อไม่สร้างข้อเท็จจริงใหม่; adapter ยังเป็นงานทดลองและไม่ถูกใช้ในแอป

> ข้อจำกัดสำคัญ: ใช้คู่ `context → output` จากตัวอย่าง `summarization` แบบ weak supervision เท่านั้น ตัวกรองไม่ได้รับประกันความหมายตรงกัน และผลสรุปจากโมเดลเป็นผลทดลอง ควรตรวจข้อเท็จจริงก่อนนำไปใช้

## เริ่มใช้งาน

แนะนำ Python 3.11 หรือ 3.12 และเครื่องที่มี NVIDIA GPU สำหรับการฝึกเต็มชุด (ค่าเริ่มต้นใช้ CPU ได้ แต่อาจใช้เวลานานและหน่วยความจำสูง) บน Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
python prepare_data.py
python train.py
python evaluate.py
```

ติดตั้งและรันแอปสรุปบน Windows PowerShell ด้วย Python 3.12 environment ที่มี PyThaiNLP/CRF tokenizer:

```powershell
py -3.12 -m venv "$env:LOCALAPPDATA\nlp-directml-venv"
$py = Join-Path $env:LOCALAPPDATA 'nlp-directml-venv\Scripts\python.exe'
& $py -m pip install -r requirements-directml.txt
& $py app.py --text "วางเนื้อหาที่ต้องการสรุปตรงนี้"
```

ถ้าไม่ระบุ `--text` แอปจะถามเนื้อหาใน terminal บทความทั่วไปจะย่อโดยเลือกประโยคสำคัญจากต้นฉบับ ส่วนเนื้อหาแบบรายชื่อที่มีช่วงวันที่จะสรุปเป็นภาพรวม

`train_llama.py` และ `evaluate_llama.py` เป็น workflow ทดลอง LoRA แยกจากแอป CLI

### ทดลอง pipeline ขนาดเล็ก

ใช้ตรวจว่าติดตั้งและ fine-tune ได้ก่อนเริ่ม train เต็มชุด:

```powershell
python train.py --max-train-samples 32 --epochs 1
python evaluate.py --max-samples 32
```

ผลจากข้อมูลเพียง 32 ตัวอย่างเป็น smoke test เท่านั้น ห้ามนำไปกล่าวอ้างว่าเป็นผลการทดลองสุดท้าย

## การเตรียมข้อมูล

`prepare_data.py` ทำงานกับ split `train` ทั้งหมดและเลือกเฉพาะตัวอย่างที่:

- `type == "summarization"`
- `rating >= 7`
- ทั้ง `context` และ `output` ไม่ว่าง
- ความยาว `output/context` อยู่ระหว่าง 0.65 ถึง 1.25 (นับตัวอักษร)
- `context` ไม่ซ้ำกันหลัง trim
- `source_text`, `target_text` และ `instruction` ต้องไม่มีอักษร/เครื่องหมายประกอบจาก Unicode script อื่นที่ไม่ใช่ไทย (อนุญาตตัวเลขและเครื่องหมายทั่วไป)

ค่า seed 42 ใช้สุ่มแบ่งข้อมูลแบบ 80/10/10 โดย source ที่ซ้ำถูกตัดก่อน split `data/manifest.json` บันทึก dataset fingerprint, เงื่อนไข, จำนวนที่คัด/ตัด, สัดส่วนข้อมูล, citation และ SHA-256 ของไฟล์ split ทุกไฟล์

## หลักฐานและการประเมิน

- `data/manifest.json`: ที่มา license, fingerprint, เกณฑ์คัด, จำนวน และ hash ของข้อมูล
- `data/train.jsonl`, `data/validation.jsonl`, `data/test.jsonl`: ข้อมูลที่ใช้ฝึกและ holdout; ห้ามนำ test ไป train
- `artifacts/training_report.json`: พารามิเตอร์, จำนวนตัวอย่าง, hash ของ train/validation และ training/evaluation loss เมื่อ train สำเร็จ
- `artifacts/evaluation_report.json`: chrF และ ROUGE-L เทียบกับ output อ้างอิงบน test split
- `artifacts/predictions.jsonl`: source/reference/model output สำหรับตรวจตัวอย่าง

chrF/ROUGE-L วัดการทับซ้อนของข้อความกับ reference งานสรุปเท่านั้น ไม่วัดความเที่ยงตรงของข้อเท็จจริง การรักษาความหมาย หรือความสำเร็จในการเปลี่ยนระดับภาษา จึงควรให้ผู้ประเมินอย่างน้อย 2 คนให้คะแนนตัวอย่าง test แยกกัน เช่น ความลื่นไหล (1–5), ความหมายคงเดิม (1–5), ระดับภาษาตรงโจทย์ (1–5), และข้อเท็จจริงผิดเพี้ยน (ใช่/ไม่ใช่) พร้อมสรุปความเห็นไม่ตรงกัน

ก่อน train ให้ทบทวนตัวอย่างสุ่มจากแต่ละ split และบันทึกข้อผิดพลาด/การคัดออกไว้ อย่าเติมข้อมูลส่วนตัวหรือใช้ข้อความที่ไม่มีสิทธิ์เผยแพร่ลงในชุดทดสอบ

## ผลรอบเตรียมข้อมูล

ข้อมูลปัจจุบันผ่านตัวกรองอักษรไทยเพิ่มเติมแล้ว: train 2,527, validation 315, test 323 ตัวอย่าง โดย manifest บันทึกจำนวนที่ตัดและ hash ปัจจุบันไว้ โมเดลใน `artifacts/model` และรายงานฝึกเดิมยังอ้างอิงข้อมูลก่อนคลีน; ต้องรัน `python train.py` ใหม่เพื่อฝึกด้วยชุดข้อมูลปัจจุบัน ซึ่งจะเขียนทับ `artifacts/model` และ checkpoint เดิม

รอบเตรียมข้อมูลต้นฉบับโหลดได้ 118,898 แถว และคัดได้ 4,865 คู่ก่อนใช้ตัวกรองอักษรไทยเพิ่มเติม ค่า fingerprint และที่มาของข้อมูลอยู่ใน manifest

ยังไม่มีการบันทึกผล fine-tuning ใน environment ที่ใช้เตรียม repository นี้ เพราะไม่มี PyTorch/Transformers และไม่พบ GPU ตอนเริ่มงาน อย่าอ้างว่ามีโมเดลที่ฝึกเสร็จหรือมีคะแนนประเมินจนกว่าจะรันคำสั่งข้างต้นและสร้างรายงานจริง

## Citation

Pengpun, Parinthapat, Can Udomcharoenchaikit, Weerayut Buaphet, and Peerat Limkonchotiwat. 2024. “Seed-Free Synthetic Data Generation Framework for Instruction-Tuning LLMs: A Case Study in Thai.” ACL Student Research Workshop. <https://aclanthology.org/2024.acl-srw.38>.

Dataset card ระบุ MIT License; โปรดตรวจเงื่อนไขและการอ้างอิงต้นฉบับก่อนเผยแพร่ผลงาน