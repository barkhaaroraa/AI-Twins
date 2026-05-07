import logging
import threading
from typing import Optional

from app.agents import AGENTS, AgentSpec
from app.db.mongo import create_user_if_not_exists, get_user
from app.llm.ollama_client import OllamaUnavailable, generate_response
from app.memory.memory_updater import update_memory
from app.memory.summarizer import fast_store_payload, summarize_memory
from app.memory.vector import init_vector_collection, search_memory
from twin.prompt_builder import PromptBuilder

log = logging.getLogger(__name__)


class TwinOrchestrator:
    def __init__(self):
        self.prompt_builder = PromptBuilder()

    def initialize(self):
        init_vector_collection()

    def process_agent_message(
        self, user_id: str, message: str, agent_name: str
    ) -> dict:
        spec = AGENTS.get(agent_name)
        if spec is None:
            raise ValueError(
                f"Unknown agent '{agent_name}'. Available: {sorted(AGENTS)}"
            )
        return self._process(user_id, message, spec)

    def _process(self, user_id: str, message: str, agent: AgentSpec) -> dict:
        create_user_if_not_exists(user_id)

        retrieved = search_memory(user_id, message, limit=4)

        user = get_user(user_id) or {}
        prompt = self.prompt_builder.build_prompt(
            user_profile=user,
            retrieved_memories=retrieved,
            message=message,
            agent_role=agent.role_prompt,
        )

        ollama_error = None
        try:
            response = generate_response(prompt)
        except OllamaUnavailable as e:
            log.warning("Ollama unavailable: %s", e)
            ollama_error = str(e)
            response = (
                "I can't reach the local LLM right now, so I can't answer this turn. "
                "Your message will still be saved to memory if it contains anything noteworthy."
            )
        except Exception as e:
            log.exception("generate_response failed")
            ollama_error = str(e)
            response = "Something went wrong generating a reply, but your message was processed."

        threading.Thread(
            target=self._store_memory,
            args=(user_id, message, agent.name, agent.force_store),
            daemon=True,
        ).start()

        out = {
            "response": response,
            "memory_used": retrieved,
            "agent": agent.name,
        }
        if ollama_error:
            out["error"] = ollama_error
        return out

    @staticmethod
    def _store_memory(
        user_id: str, message: str, source_agent: str, force_store: bool
    ):
        try:
            summarized = None
            try:
                summarized = summarize_memory(message)
            except Exception:
                log.exception("summarize_memory failed; using fast payload")
            if not summarized and force_store:
                summarized = fast_store_payload(message)
            if summarized:
                update_memory(user_id, summarized, source_agent=source_agent)
        except Exception:
            log.exception("Background memory work failed for user %s", user_id)
