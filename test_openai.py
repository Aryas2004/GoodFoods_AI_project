# test_openai.py
import os
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("Set OPENAI_API_KEY in your environment before testing.")

client = OpenAI(api_key=OPENAI_API_KEY)
SYSTEM_PROMPT = "You are a test assistant. Say hello in one sentence."

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":"Hello?"}],
    max_tokens=50
)
print("Result:", resp.choices[0].message["content"])
