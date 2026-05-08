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
            # Surface contradiction pairs first if both sides are present in this batch.
            ids = {m.get("memory_id") for m in retrieved_memories}
            contradicting_pairs = []
            for mem in retrieved_memories:
                lineage = mem.get("lineage", {}) or {}
                for other_id in lineage.get("contradicted_by", []) or []:
                    if other_id in ids:
                        pair = tuple(sorted([mem["memory_id"], other_id]))
                        if pair not in contradicting_pairs:
                            contradicting_pairs.append(pair)

            if contradicting_pairs:
                by_id = {m["memory_id"]: m for m in retrieved_memories}
                sections.append("  [CONTRADICTIONS — user previously said vs. now]")
                for a_id, b_id in contradicting_pairs:
                    a, b = by_id[a_id], by_id[b_id]
                    a_ts = (a.get("created_at") or "")[:10]
                    b_ts = (b.get("created_at") or "")[:10]
                    sections.append(f"  - [{a_ts}] {a.get('summary', a.get('text', ''))}")
                    sections.append(f"  - [{b_ts}] {b.get('summary', b.get('text', ''))}")
                sections.append("  Treat the more-recent statement as current unless the user asks about the change.")

            # Group by tier first (warm before cold), then by memory_type.
            by_tier: Dict[str, Dict[str, list]] = {}
            for mem in retrieved_memories:
                tier = mem.get("tier", "warm")
                by_type = by_tier.setdefault(tier, {})
                by_type.setdefault(mem.get("memory_type", "Semantic"), []).append(mem)

            for tier in ("warm", "cold"):
                if tier not in by_tier:
                    continue
                if len(by_tier) > 1:
                    sections.append(f"  [{tier} tier]")
                for mtype, mems in by_tier[tier].items():
                    for mem in mems:
                        score = mem.get("similarity_score", 0)
                        text = mem.get("text") or mem.get("summary", "")
                        sections.append(
                            f"- [{mtype}, score={score:.2f}] {text}"
                        )
        else:
            sections.append("- None")

        sections.append("")
        sections.append(f'User Message:\n"{message}"')
        sections.append("")
        sections.append("AI Response:")

        return "\n".join(sections)
