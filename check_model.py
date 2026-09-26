import requests

# ⚠️ อย่าลืมใส่ API Key ของจริงของคุณตรงนี้นะครับ
api_key = "ใส่_API_KEY_ของคุณที่นี่"
url = "https://api.opentyphoon.ai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}"
}

print("กำลังเชื่อมต่อเซิร์ฟเวอร์ Typhoon เพื่อดึงชื่อโมเดล...")
response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print("\n✅ รายชื่อโมเดลที่คุณสามารถใช้งานได้ (นำชื่อไปใส่ใน app.py):")
    
    # จัดการกรณีที่เซิร์ฟเวอร์ส่งมาเป็น List ตรงๆ หรือส่งมาใน Key 'data'
    models = data.get('data', data) if isinstance(data, dict) else data
    
    for m in models:
        # ดึงชื่อ id ของโมเดลออกมา
        model_name = m.get('id') if isinstance(m, dict) else m
        print(f" 👉 {model_name}")
else:
    print(f"❌ เกิดข้อผิดพลาดจากเซิร์ฟเวอร์: Code {response.status_code}")
    print(response.text)