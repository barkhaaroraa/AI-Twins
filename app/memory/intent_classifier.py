"""
ML intent classifier for memory routing.

Pipeline: sentence-transformer embeddings -> logistic regression head.
No regex, no similarity-search-against-prototypes at inference. The LR
weights are learned once at startup from a small labelled set of natural-
language utterances. Adding more examples improves the classifier without
any code change.

Two independent binary decisions are produced for every query:

  needs_retrieval : should we look up old memories?
  should_store    : does this message contain something worth remembering?
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.memory.vector import embedder


# ---------------------------------------------------------------------------
# Labelled training data (positives + negatives, embedded once at startup).
# ---------------------------------------------------------------------------

NEEDS_RETRIEVAL_POS: List[str] = [
    "what did I tell you yesterday",
    "remember when I said I was working on the dashboard",
    "continue from where we left off",
    "what was my preference about dark mode",
    "did I mention my project last week",
    "as I told you earlier the deadline is friday",
    "recall my project details",
    "what do you know about me",
    "what tasks am I working on",
    "remind me what my goal was",
    "what is my favourite language",
    "have I told you about my team",
    "based on what you know about me suggest a plan",
    "given my background what should I do next",
    "use what I told you before to answer this",
    "summarize what we discussed",
    "what have I been building",
    "what are my open todos",
    "where did we leave off in the auth refactor",
    "look up my past conversation about kubernetes",
    "tell me what my stack looks like",
    "what's my current project status",
]

NEEDS_RETRIEVAL_NEG: List[str] = [
    "hello",
    "hi how are you",
    "good morning",
    "thanks",
    "ok cool",
    "what is the capital of france",
    "explain quantum mechanics in simple terms",
    "write a poem about the sea",
    "what is 2 plus 2",
    "tell me a joke",
    "translate hola to english",
    "what is python used for",
    "how does a transformer architecture work",
    "summarize this article for me",
    "give me a recipe for pasta",
    "what time is it",
    "code a fibonacci function in javascript",
    "what is the difference between sql and nosql",
    "draft an email to my landlord",
    "convert celsius to fahrenheit",
    "write a haiku about autumn",
    "who won the world cup in 2018",
    "explain async await in javascript",
    "show me a docker compose example",
]

SHOULD_STORE_POS: List[str] = [
    "I prefer dark mode in my editor",
    "I am working on a recommendation engine at work",
    "my name is Alex and I am a backend engineer",
    "I live in Berlin",
    "I love medium roast coffee",
    "I am building an AI startup focused on memory",
    "actually I changed my mind I prefer postgres over mongo",
    "I finished the migration task today",
    "my favourite language is Python",
    "I studied computer science at IIT",
    "I work at a fintech company",
    "my goal this quarter is to ship the new auth system",
    "I want to learn rust this year",
    "remember that I am vegetarian",
    "the project deadline is march 30",
    "I usually code in vim",
    "my team uses kubernetes for deployment",
    "I have two cats named Pixel and Bit",
    "I hate writing yaml",
    "I'm planning to move to Amsterdam next year",
    "my partner's name is Jamie",
    "I commute by bike",
]

SHOULD_STORE_NEG: List[str] = [
    "hello",
    "hi",
    "thanks a lot",
    "ok",
    "what is python",
    "explain how cosine similarity works",
    "write me a fibonacci function",
    "tell me a joke",
    "translate this sentence",
    "summarize this paragraph",
    "what is the weather like",
    "who is the president of france",
    "give me a regex for emails",
    "what does this error mean",
    "draft an email",
    "convert 100 km to miles",
    "is python better than rust",
    "what should I cook tonight",
    "good morning",
    "haha that's funny",
    "tell me about quantum mechanics",
    "what is the difference between a list and a tuple",
    "show me a sql join example",
    "explain dependency injection",
]


def _embed_batch(texts: List[str]) -> np.ndarray:
    return np.asarray(
        embedder.encode(texts, normalize_embeddings=True), dtype=np.float32
    )


def _train(pos: List[str], neg: List[str]) -> LogisticRegression:
    X = _embed_batch(pos + neg)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    # liblinear is robust on small datasets and fast at predict time.
    model = LogisticRegression(C=1.0, solver="liblinear", random_state=42)
    model.fit(X, y)
    return model


class IntentClassifier:
    """Logistic regression on sentence-transformer embeddings."""

    NEEDS_RETRIEVAL_THRESHOLD = 0.5
    SHOULD_STORE_THRESHOLD = 0.5

    def __init__(self) -> None:
        self.retrieval_lr = _train(NEEDS_RETRIEVAL_POS, NEEDS_RETRIEVAL_NEG)
        self.store_lr = _train(SHOULD_STORE_POS, SHOULD_STORE_NEG)

    def classify(self, query: str) -> Dict:
        if not query or not query.strip():
            return {
                "needs_retrieval": False,
                "should_store": False,
                "retrieval_score": 0.0,
                "store_score": 0.0,
                "method": "embedding+logreg",
            }

        x = _embed_batch([query])
        ret_prob = float(self.retrieval_lr.predict_proba(x)[0, 1])
        sto_prob = float(self.store_lr.predict_proba(x)[0, 1])

        return {
            "needs_retrieval": ret_prob >= self.NEEDS_RETRIEVAL_THRESHOLD,
            "should_store": sto_prob >= self.SHOULD_STORE_THRESHOLD,
            "retrieval_score": round(ret_prob, 3),
            "store_score": round(sto_prob, 3),
            "method": "embedding+logreg",
        }


_instance: IntentClassifier | None = None


def get_intent_classifier() -> IntentClassifier:
    global _instance
    if _instance is None:
        _instance = IntentClassifier()
    return _instance
