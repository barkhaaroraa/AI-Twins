import json
import re
import requests
from app.config import OLLAMA_URL, OLLAMA_MODEL


def generate_response(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["response"]


def generate_json(prompt: str) -> dict:
    raw = generate_response(prompt)
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    return json.loads(cleaned)
