from typing import Dict, List


class PromptBuilder:
    def build_prompt(
        self,
        user_profile: dict,
        retrieved_memories: List[dict],
        message: str,
    ) -> str:
        sections = []

        # User profile
        profile_lines = []
        prefs = user_profile.get("preferences", {})
        tasks = user_profile.get("tasks", [])
        if prefs:
            profile_lines.append(f"Preferences: {prefs}")
        if tasks:
            active = [t["title"] for t in tasks if t.get("status") != "completed"]
            if active:
                profile_lines.append("Active Tasks: " + ", ".join(active))

        sections.append("User Profile:")
        if profile_lines:
            sections.extend(f"- {line}" for line in profile_lines)
        else:
            sections.append("- None")

        # Retrieved memories grouped by type
        sections.append("")
        sections.append("Relevant Memories:")
        if retrieved_memories:
            by_type: Dict[str, list] = {}
            for mem in retrieved_memories:
                mtype = mem.get("memory_type", "Semantic")
                by_type.setdefault(mtype, []).append(mem)

            for mtype, mems in by_type.items():
                for mem in mems:
                    score = mem.get("final_score", mem.get("similarity_score", 0))
                    sections.append(
                        f"- [{mtype}, score={score:.2f}] {mem.get('text', '')}"
                    )
        else:
            sections.append("- None")

        # User message
        sections.append("")
        sections.append(f'User Message:\n"{message}"')
        sections.append("")
        sections.append("AI Response:")

        return "\n".join(sections)
