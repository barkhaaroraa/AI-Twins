import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
APP_NAME = os.getenv("APP_NAME")

# --- Graph Settings ---
GRAPH_SIMILARITY_THRESHOLD = 0.80
GRAPH_MAX_HOPS = 2
PAGERANK_ALPHA = 0.85

# --- Consolidation Settings ---
CONSOLIDATION_THRESHOLD = 20
CLUSTER_DISTANCE_THRESHOLD = 0.5
DEDUP_SIMILARITY_THRESHOLD = 0.92
NMF_TOPICS = 5

# --- Episodic Settings ---
SESSION_TIMEOUT_MINUTES = 30
DECAY_LAMBDA = 0.05

# --- Retrieval Weights ---
W_SEMANTIC = 0.40
W_GRAPH = 0.25
W_RECENCY = 0.20
W_IMPORTANCE = 0.15

# --- Cache ---
CACHE_MAX_SIZE = 100
