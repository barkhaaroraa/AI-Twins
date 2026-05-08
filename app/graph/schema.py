"""Single source of truth for graph node labels and edge types/weights."""

# Node labels
USER = "User"
AGENT = "Agent"
MEMORY = "Memory"
ENTITY = "Entity"
EPISODE = "Episode"
CONCEPT = "Concept"

# Edge types
OWNED_BY = "OWNED_BY"
AUTHORED_BY = "AUTHORED_BY"
SHARED_WITH = "SHARED_WITH"
MENTIONS = "MENTIONS"
DEPENDS_ON = "DEPENDS_ON"
DERIVED_FROM = "DERIVED_FROM"
SUPERSEDES = "SUPERSEDES"
CONTRADICTS = "CONTRADICTS"
REFINES = "REFINES"
ABSTRACTION_OF = "ABSTRACTION_OF"
CAUSES = "CAUSES"
PRECEDES = "PRECEDES"
CO_ACTIVATED = "CO_ACTIVATED"

# Static edge weights for PPR (CO_ACTIVATED weight is per-edge, learned).
# CONTRADICTS is inert (0.0) here: contradiction handling is centralised in record_contradiction
# (inline SUPERSEDES + lineage mirror + confidence drift), and Stage 5 drops superseded memories
# via lineage.supersedes_by. A negative PPR weight on top of that double-counted the suppression
# AND, because PPR treats edges as undirected, pulled the winner's activation down too.
EDGE_WEIGHTS = {
    DEPENDS_ON:    0.9,
    MENTIONS:      0.5,
    DERIVED_FROM:  0.7,
    REFINES:       0.6,
    CAUSES:        0.8,
    PRECEDES:      0.4,
    ABSTRACTION_OF:0.6,
    CONTRADICTS:   0.0,
    SUPERSEDES:    0.0,
}

# Visibility enum
PRIVATE = "private"
SHARED = "shared"
PUBLIC = "public"
