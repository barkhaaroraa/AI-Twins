import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

from app.agents import AGENTS, AgentSpec
from app.cognition.cascade import cascade, reflect
from app.cognition.consolidator.light import mark_request
from app.db.mongo import create_user_if_not_exists, get_user
from app.llm.ollama_client import OllamaUnavailable, generate_response
from app.memory.backing import append_event, attach_produced_memory
from app.memory.memory_updater import update_memory
from app.memory.summarizer import fast_store_payload, summarize_memory
from app.memory.vector import init_vector_collection
from twin.prompt_builder import PromptBuilder

log = logging.getLogger(__name__)

# Shared pool for off-critical-path work (reflect + memory write). Bounded so the process
# can't accumulate hundreds of threads under burst load. Daemon=True so test/dev shutdown
# isn't blocked by in-flight tail work.
_TAIL_POOL = ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="orchestrator-tail",
)


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
        mark_request()  # bumps the idle timer; light consolidator stays asleep while busy
        create_user_if_not_exists(user_id)

        # Backing-store first (every event recorded).
        event_id = append_event(
            user_id=user_id,
            agent_name=agent.name,
            event_type="user_message",
            payload={"message": message},
        )

        # 6-stage cascade (ACL applied at Stage 5; vector seed also has native ACL filter).
        retrieved = cascade(message, user_id, agent.name, top_n=4)

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

        # Stage 6: reflect off the critical path.
        used_ids: List[str] = [m["memory_id"] for m in retrieved]
        _TAIL_POOL.submit(self._safe_reflect, used_ids, user_id, agent.name, message)

        # Async write tail.
        _TAIL_POOL.submit(
            self._store_memory,
            user_id, message, agent.name, agent.force_store, event_id,
        )

        out = {
            "response": response,
            "memory_used": retrieved,
            "agent": agent.name,
            "event_id": event_id,
        }
        if ollama_error:
            out["error"] = ollama_error
        return out

    @staticmethod
    def _safe_reflect(used_ids, user_id, agent_name, message):
        try:
            reflect(used_ids, user_id, agent_name, message)
        except Exception:
            log.exception("reflect failed")

    @staticmethod
    def _store_memory(
        user_id: str, message: str, source_agent: str, force_store: bool, event_id: str
    ):
        try:
            summarized = None
            try:
                summarized = summarize_memory(message)
            except Exception:
                log.exception("summarize_memory failed; using fast payload")
            if not summarized and force_store:
                # logger is the only force_store agent: it captures body-state log entries
                # ("ran 5k this morning", "ate eggs at 9am") which are episodic facts, not
                # generic Semantic memories.
                summarized = fast_store_payload(
                    message, intent="fact", memory_type="Episodic", importance=0.6,
                )
            if not summarized:
                return
            doc = update_memory(
                user_id, summarized,
                source_agent=source_agent,
                source_event_id=event_id,
            )
            if doc:
                attach_produced_memory(event_id, doc["_id"])
        except Exception:
            log.exception("Background memory work failed for user %s", user_id)
