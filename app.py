import sys
import os
import requests
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

# โหลด API Key จาก .env
load_dotenv()

SYSTEM_PROMPT = """
คุณคือ "ผู้เชี่ยวชาญด้านการสื่อสารองค์กร"
มีหน้าที่เปลี่ยนข้อมูลและคีย์เวิร์ดสั้น ๆ ของผู้ใช้
ให้เป็นเอกสารทางการภาษาไทยที่สุภาพและเหมาะสม

กฎ:
1. ต้องใช้ข้อมูลของผู้ใช้ให้ครบถ้วน
2. ต้องมีชื่อผู้รับและตำแหน่งผู้รับ
3. ต้องมีวันที่
4. ต้องมีชื่อผู้เขียนและตำแหน่งผู้เขียน
5. ปรับภาษาพูดหรือคีย์เวิร์ดให้เป็นภาษาทางการ
6. ห้ามสร้างข้อมูลส่วนตัวที่ผู้ใช้ไม่ได้ระบุขึ้นมาเอง
7. หากข้อมูลบางอย่างไม่มี ให้ใช้ [กรุณาระบุ...]
8. เอกสารต้องมี:
   - วันที่
   - คำขึ้นต้น
   - ชื่อผู้รับ
   - ตำแหน่งผู้รับ
   - เนื้อหาหลัก
   - คำลงท้าย
   - ชื่อผู้เขียน
   - ตำแหน่งผู้เขียน
9. สร้างเฉพาะเอกสาร ห้ามอธิบายเพิ่มเติม
"""

def main():

    print("=" * 60)
    print("🤖 AI Formal Document Drafter")
    print("=" * 60)

    # API Key
    api_key = os.getenv("TYPHOON_API_KEY")

    if not api_key:
        print("\n❌ ไม่พบ TYPHOON_API_KEY")
        print("กรุณาตรวจสอบไฟล์ .env")
        return

    print("\n📋 กรุณากรอกข้อมูลเอกสาร")
    print("-" * 60)

    # ข้อมูลผู้เขียน
    author_name = input("👤 ชื่อผู้เขียน: ")
    author_position = input("💼 ตำแหน่งผู้เขียน: ")

    # วันที่
    date = input("📅 วันที่: ")

    # ข้อมูลผู้รับ
    receiver_name = input("👨‍💼 ชื่อผู้รับ: ")
    receiver_position = input("🏢 ตำแหน่งผู้รับ: ")

    # คีย์เวิร์ด
    keywords = input(
        "📝 คีย์เวิร์ด เช่น ลาป่วย, ท้องเสีย, 2 วัน: "
    )

    if not keywords.strip():
        print("\n❌ กรุณาระบุคีย์เวิร์ด")
        return

    print("\n⏳ กำลังสร้างเอกสารทางการ...")
    print("โปรดรอสักครู่\n")

    user_prompt = f"""
ข้อมูลผู้เขียน:
ชื่อ: {author_name}
ตำแหน่ง: {author_position}

วันที่:
{date}

ข้อมูลผู้รับ:
ชื่อ: {receiver_name}
ตำแหน่ง: {receiver_position}

คีย์เวิร์ดของเอกสาร:
{keywords}

โปรดนำข้อมูลทั้งหมดไปสร้างเป็นเอกสารทางการภาษาไทย
"""

    url = "https://api.opentyphoon.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "typhoon-v2.5-30b-a3b-instruct",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 800
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload
        )

        if response.status_code == 200:

            data = response.json()

            result = data["choices"][0]["message"]["content"].strip()

            print("=" * 60)
            print("✨ เอกสารทางการ ✨")
            print("=" * 60)
            print()
            print(result)
            print()
            print("=" * 60)

        else:

            print(f"❌ Error Code: {response.status_code}")
            print(response.text)

    except Exception as e:

        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")


if __name__ == "__main__":
    main()