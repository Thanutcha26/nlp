# ต้นแบบเรียบเรียงข้อความภาษาไทย

โครงงานนี้เตรียมข้อมูลภาษาไทยและ fine-tune `google/mt5-small` สำหรับทดลองเรียบเรียงข้อความ โดยยึดชุดข้อมูล [WangchanX Seed-Free Synthetic Instruct Thai 120k](https://huggingface.co/datasets/airesearch/wangchanx-seed-free-synthetic-instruct-thai-120k) เป็นแหล่งข้อมูลหลัก

> ข้อจำกัดสำคัญ: dataset มีตัวอย่าง 5 ประเภท แต่ไม่มีป้ายกำกับ paraphrase หรือระดับภาษาโดยตรง โครงการนี้ใช้เฉพาะ `summarization` และจับคู่ `context → output` เป็น weak supervision เท่านั้น ตัวกรองความยาวช่วยลดตัวอย่างที่สรุปสั้นมาก แต่ไม่รับประกันความหมายตรงกัน และโมเดลไม่ได้ถูกฝึกแบบมี label สำหรับภาษาทางการ/กึ่งทางการ/กันเอง โหมดในหน้า demo จึงเป็นการทดลอง ไม่ใช่ผลที่ dataset รับรอง

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
streamlit run app.py
```

ขั้นแรก download dataset จาก Hugging Face ประมาณ 239 MB โดยไม่ต้องมี API token; การตั้ง `HF_TOKEN` ช่วยเพิ่ม quota ได้

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

รอบที่บันทึกใน repository นี้โหลดได้ 118,898 แถว จากนั้นคัดได้ 4,865 คู่ weakly supervised: train 3,893, validation 486, test 486 (seed 42) ค่าเหล่านี้อยู่ใน manifest พร้อม fingerprint/hash เพื่อให้ตรวจและสร้างซ้ำได้

ยังไม่มีการบันทึกผล fine-tuning ใน environment ที่ใช้เตรียม repository นี้ เพราะไม่มี PyTorch/Transformers และไม่พบ GPU ตอนเริ่มงาน อย่าอ้างว่ามีโมเดลที่ฝึกเสร็จหรือมีคะแนนประเมินจนกว่าจะรันคำสั่งข้างต้นและสร้างรายงานจริง

## Citation

Pengpun, Parinthapat, Can Udomcharoenchaikit, Weerayut Buaphet, and Peerat Limkonchotiwat. 2024. “Seed-Free Synthetic Data Generation Framework for Instruction-Tuning LLMs: A Case Study in Thai.” ACL Student Research Workshop. <https://aclanthology.org/2024.acl-srw.38>.

Dataset card ระบุ MIT License; โปรดตรวจเงื่อนไขและการอ้างอิงต้นฉบับก่อนเผยแพร่ผลงาน