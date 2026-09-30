import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("TYPHOON_API_KEY")

if not api_key:
    print("❌ ไม่พบ API Key")
    exit()

url = "https://api.opentyphoon.ai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}"
}

response = requests.get(url, headers=headers)

print("Status Code:", response.status_code)
print(response.text)