import requests

# ==========================================
# 1. ตั้งค่า API และ ข้อความต้นแบบ (Exemplar)
# ==========================================
# 🔴 นำ API Key ของ Typhoon มาใส่ตรงนี้
API_KEY = "sk-mCs6zOZIVX6BfKj141jr9hz2Y64HxTeltNOiv9s2EvQKPifF"
URL = "https://api.opentyphoon.ai/v1/chat/completions" 

# 🔴 นำข้อความจาก Step 1 (Wangchan ThaiInstruct) มาวางระหว่าง """ ... """
EXEMPLAR_TEXT = """
แคมเปญ "The ONGR 1" เป็นการนำเสนอภารกิจสุดสนุก ผ่านตัวละครเอเจนท์
ตัวละครเอเจนท์พูดภาษาถิ่น สร้างความแปลกใหม่ น่าสนใจ ดึงดูดความสนใจลูกค้า
"""

# ==========================================
# 2. สร้าง System Prompt (ใช้ทฤษฎี Register Analysis)
# ==========================================
SYSTEM_PROMPT = f"""คุณคือผู้เชี่ยวชาญด้านการร่างเอกสารทางการและจดหมายธุรกิจ
หน้าที่ของคุณคือร่างเอกสารใหม่จากข้อมูลที่ผู้ใช้ระบุ โดยต้องปฏิบัติตามกฎต่อไปนี้อย่างเคร่งครัด:

1. การวิเคราะห์ระดับภาษา (Register Analysis): คุณต้องใช้คำศัพท์ โครงสร้างประโยค และระดับความทางการ (Formality) ให้เทียบเท่ากับ "ข้อความต้นแบบ" ด้านล่างนี้
2. ห้ามแต่งเติมเนื้อหา (No Hallucination): ใช้เฉพาะข้อมูลที่ผู้ใช้ให้มาเท่านั้น ห้ามคิดชื่อคน วันที่ หรือสถานที่ขึ้นมาเอง หากข้อมูลไม่พอให้เว้นช่องว่างไว้ เช่น [ระบุวันที่]
3. ความสละสลวย: เปลี่ยนภาษาพูดให้เป็นภาษาเขียนทางการทั้งหมด

[ข้อความต้นแบบ (Style Exemplar)]
{EXEMPLAR_TEXT}
"""

# ==========================================
# 3. ฟังก์ชันสำหรับเรียกใช้งาน AI
# ==========================================
def draft_document(user_input):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "typhoon-v2.5-30b-a3b-instruct",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"ร่างเอกสารจากข้อมูลนี้: {user_input}"}
        ],
        "temperature": 0.2, # ใช้ 0.2 เพื่อลดการแต่งเรื่องมั่ว (อิงจากเปเปอร์ Wangchan)
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(URL, headers=headers, json=data)
        
        # ถ้าระบบแจ้ง Error (เช่น 400) จะให้มันแสดงข้อความที่เซิร์ฟเวอร์ตอบกลับมาด้วย
        if response.status_code != 200:
            return f"❌ API Error ({response.status_code}): {response.text}"
            
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการเชื่อมต่อ: {e}"

# ==========================================
# 4. ส่วนการโต้ตอบกับผู้ใช้ (User Interface)
# ==========================================
def main():
    print("="*50)
    print("🤖 ยินดีต้อนรับสู่ระบบ AI Formal Document Drafter")
    print("="*50)
    print("พิมพ์คีย์เวิร์ดบ้านๆ เพื่อให้ AI ร่างเอกสารทางการ (พิมพ์ 'exit' เพื่อออก)")
    
    while True:
        user_input = input("\n📝 ใส่คีย์เวิร์ด/เรื่องที่ต้องการร่าง: ")
        if user_input.lower() == 'exit':
            print("ลาก่อนครับ!")
            break
            
        print("\n⏳ AI กำลังประมวลผลระดับภาษาและร่างเอกสาร...")
        drafted_doc = draft_document(user_input)
        
        print("-" * 50)
        print("📄 เอกสารที่ร่างเสร็จแล้ว:\n")
        print(drafted_doc)
        print("-" * 50)

if __name__ == "__main__":
    main()