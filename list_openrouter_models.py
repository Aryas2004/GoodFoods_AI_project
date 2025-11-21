# list_openrouter_models.py
import os, requests, json

key = os.getenv("OPENROUTER_API_KEY")
if not key:
    raise SystemExit("Set OPENROUTER_API_KEY environment variable first.")

url = "https://openrouter.ai/api/v1/models"
headers = {"Authorization": f"Bearer {key}"}

r = requests.get(url, headers=headers, timeout=30)
print("Status:", r.status_code)
try:
    data = r.json()
    print(json.dumps(data, indent=2)[:4000])   # print first 4000 chars
except Exception:
    print("Non-json response:", r.text[:2000])
