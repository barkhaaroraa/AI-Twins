from typing import Dict, List, Optional


class PromptBuilder:
    def build_prompt(
        self,
        user_profile: dict,
        retrieved_memories: List[dict],
        message: str,
        agent_role: Optional[str] = None,
    ) -> str:
        sections = []

        if agent_role:
            sections.append(agent_role.strip())
            sections.append("")

        sections.append("Relevant Memories:")
        if retrieved_memories:
            by_type: Dict[str, list] = {}
            for mem in retrieved_memories:
                mtype = mem.get("memory_type", "Semantic")
                by_type.setdefault(mtype, []).append(mem)

            for mtype, mems in by_type.items():
                for mem in mems:
                    score = mem.get("similarity_score", 0)
                    sections.append(
                        f"- [{mtype}, score={score:.2f}] {mem.get('text', '')}"
                    )
        else:
            sections.append("- None")

        sections.append("")
        sections.append(f'User Message:\n"{message}"')
        sections.append("")
        sections.append("AI Response:")

        return "\n".join(sections)
