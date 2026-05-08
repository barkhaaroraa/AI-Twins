import json
import re

import requests

from app.config import OLLAMA_URL, OLLAMA_MODEL


class OllamaUnavailable(RuntimeError):
    """Ollama is not running, unreachable, or timed out."""


def generate_response(prompt: str, timeout: int = 180) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 512, "temperature": 0.7},
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as e:
        raise OllamaUnavailable(
            f"Ollama not reachable at {OLLAMA_URL}. Is the container running?"
        ) from e
    except requests.exceptions.Timeout as e:
        raise OllamaUnavailable(
            f"Ollama timed out after {timeout}s while generating."
        ) from e

    response.raise_for_status()
    return response.json()["response"]


def generate_json(prompt: str, timeout: int = 90) -> dict:
    raw = generate_response(prompt, timeout=timeout)
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Small models often add prose around the JSON; salvage the first {...} block.
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise
