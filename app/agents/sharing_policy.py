from typing import Dict, Set

from app.agents import AgentSpec


SHARING_POLICY: Dict[str, object] = {
    "auto_public_types": {"Preference", "Identity"},
    "families": {
        "health":       {"logger", "nutritionist", "trainer"},
        "productivity": {"project", "school", "research"},
    },
}


def auto_visibility(memory_type: str, agent_default: str) -> str:
    """Type-level rule: Preference/Identity memories are public regardless of authoring agent."""
    if memory_type in SHARING_POLICY["auto_public_types"]:
        return "public"
    return agent_default


def acl_check(memory: dict, agent: AgentSpec) -> bool:
    """Single chokepoint for visibility. Default-deny on missing visibility."""
    v = memory.get("visibility", "private")
    if v == "public":
        return True
    if v == "shared":
        return agent.name in (memory.get("shared_with") or [])
    if v == "private":
        return memory.get("agent_owner") == agent.name
    return False


def family_of(agent_name: str) -> str:
    for fam, members in SHARING_POLICY["families"].items():
        if agent_name in members:
            return fam
    return "unknown"
