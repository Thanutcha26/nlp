import sys
import requests

sys.stdout.reconfigure(encoding='utf-8')

SYSTEM_PROMPT = """
คุณคือ "ผู้เชี่ยวชาญด้านการสื่อสารองค์กร" ที่มีทักษะสูงในการปรับระดับภาษา (Register) จากภาษาพูดให้เป็นภาษาทางการ (Formal) 
Guidelines:
1. โครงสร้างต้องมี: คำขึ้นต้น, เนื้อหาหลักที่สมเหตุสมผล, และคำลงท้าย
2. การปรับระดับภาษา: ห้ามใช้คำแสลง เปลี่ยนคำพูดทั่วไปเป็นภาษาเขียน
3. ใช้เครื่องหมาย [...] สำหรับข้อมูลที่ผู้ใช้ต้องไปเติมเอง เช่น [ชื่อผู้รับ], [วันที่]
4. สร้างเฉพาะเนื้อหาเอกสาร ห้ามมีคำอธิบายอื่นๆ นอกเหนือจากนี้
"""

def main():
    print("="*50)
    print("🤖 AI Formal Document Drafter")
    print("="*50)
    
    # ⚠️ ใส่ API Key ของคุณที่นี่ ⚠️
    my_api_key = "YOUR_API_KEY" 
    
    # ใช้ชื่อโมเดลที่สแกนเจอ
    model_name = "typhoon-v2.5-30b-a3b-instruct" 
    
    keywords = input("\n📝 กรุณาพิมพ์คีย์เวิร์ด (เช่น ลาป่วย, ท้องเสีย, 2 วัน): ")
    
    if not keywords.strip():
        print("จบการทำงาน")
        return

    print("\n⏳ กำลังส่งข้อมูลไปที่ Typhoon API... โปรดรอสักครู่\n")
    
    url = "https://api.opentyphoon.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {my_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Input: {keywords}\nOutput:"}
        ],
        "temperature": 0.2,
        "max_tokens": 800
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            result = data['choices'][0]['message']['content'].strip()
            print("✨ --- ผลลัพธ์จดหมายทางการ --- ✨\n")
            print(result)
            print("\n" + "="*50)
        else:
            print(f"❌ Error Code: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")

if __name__ == "__main__":
    main()