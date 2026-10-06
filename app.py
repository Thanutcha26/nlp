from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "artifacts" / "model"
MODEL_NAME = "google/mt5-small"
MODES = {
    "เรียบเรียงใหม่": "โปรดเรียบเรียงข้อความต่อไปนี้ใหม่เป็นภาษาไทย โดยคงสาระสำคัญและข้อเท็จจริงเดิมให้ครบถ้วน ใช้สำนวนที่เป็นธรรมชาติ อ่านลื่นไหล และอย่าสรุปหรือตัดรายละเอียด",
    "ภาษาทางการ": "โปรดเรียบเรียงข้อความต่อไปนี้ใหม่เป็นภาษาไทยระดับทางการ สุภาพและเป็นธรรมชาติ โดยคงสาระสำคัญและข้อเท็จจริงเดิมให้ครบถ้วน อย่าสรุปหรือตัดรายละเอียด",
    "ภาษากึ่งทางการ": "โปรดเรียบเรียงข้อความต่อไปนี้ใหม่เป็นภาษาไทยกึ่งทางการที่สุภาพ เป็นธรรมชาติ และอ่านเข้าใจง่าย โดยคงสาระสำคัญและข้อเท็จจริงเดิมให้ครบถ้วน",
    "กระชับ": "โปรดเรียบเรียงข้อความต่อไปนี้ใหม่ให้กระชับ อ่านลื่นไหล โดยรักษาข้อเท็จจริงและสาระสำคัญไว้ให้ครบ",
}


@st.cache_resource
def load_model():
    model_path = MODEL_DIR if MODEL_DIR.exists() else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


st.set_page_config(page_title="Thai Text Rewriter", page_icon="✍️", layout="centered")
st.title("Thai Text Rewriter")
st.caption("ต้นแบบเรียบเรียงข้อความภาษาไทย")
st.warning(
    "โมเดลฝึกจากตัวอย่างงานสรุป ไม่ได้ฝึกด้วยป้ายกำกับระดับภาษาโดยตรง "
    "ผลลัพธ์ทุกโหมดเป็นการทดลองและควรตรวจความหมายก่อนนำไปใช้"
)

mode = st.selectbox("รูปแบบภาษา", list(MODES))
text = st.text_area("ข้อความต้นฉบับ", height=220, max_chars=4000)
run = st.button("เรียบเรียง", type="primary", disabled=not text.strip())

if run:
    if not MODEL_DIR.exists():
        st.error("ยังไม่พบโมเดลที่ผ่านการ fine-tune ใน artifacts/model. อ่าน README.md เพื่อเริ่ม train")
    else:
        with st.spinner("กำลังเรียบเรียงข้อความ..."):
            tokenizer, model = load_model()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            prompt = f"instruction: {MODES[mode]}\ncontext: {text.strip()}"
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=384,
                    num_beams=4,
                    do_sample=False,
                )
            result = tokenizer.decode(generated[0], skip_special_tokens=True)
        st.text_area("ผลลัพธ์", result, height=220)
        st.download_button(
            "ดาวน์โหลดผลลัพธ์",
            data=result,
            file_name="thai_rewrite.txt",
            mime="text/plain",
        )