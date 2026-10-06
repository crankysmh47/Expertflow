from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen


BASE_URL = os.environ.get("EXPERTFLOW_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")


def chat(prompt: str) -> str:
    payload = json.dumps({"model": "expertflow-q6", "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 96}).encode()
    request = Request(f"{BASE_URL}/chat/completions", data=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('EXPERTFLOW_API_KEY', 'local')}"})
    with urlopen(request, timeout=600) as response:
        result = json.load(response)
    message = result["choices"][0]["message"]
    return message.get("content") or message.get("reasoning_content") or ""


if __name__ == "__main__":
    print(chat("Give one practical refactoring suggestion for a Python CLI."))
