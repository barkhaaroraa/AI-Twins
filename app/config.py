import os
from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    """Fail fast at import time if a required env var is missing — better than a
    confusing TypeError deep inside requests.post(None, ...) on the first request."""
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Required env var {name} is unset. Check your .env file or environment."
        )
    return val


OLLAMA_URL = _required("OLLAMA_URL")
OLLAMA_MODEL = _required("OLLAMA_MODEL")
APP_NAME = os.getenv("APP_NAME", "ai-twin")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j_dev")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
