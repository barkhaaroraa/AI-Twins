import logging
from typing import Optional

from neo4j import GraphDatabase, Driver

from app.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

log = logging.getLogger(__name__)

_driver: Optional[Driver] = None


class Neo4jUnavailable(RuntimeError):
    pass


def driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            # "UnknownRelationshipTypeWarning" fires for rel types that exist in our schema
            # but haven't been written yet — benign for a system that grows edges over time.
            notifications_disabled_categories=["UNRECOGNIZED", "DEPRECATION"],
        )
    return _driver


def close():
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


CONSTRAINTS = [
    "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
    "CREATE CONSTRAINT agent_name_unique IF NOT EXISTS FOR (a:Agent) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
    "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX entity_lookup IF NOT EXISTS FOR (e:Entity) ON (e.name, e.kind, e.user_id)",
    "CREATE INDEX memory_user IF NOT EXISTS FOR (m:Memory) ON (m.user_id)",
    "CREATE INDEX memory_visibility IF NOT EXISTS FOR (m:Memory) ON (m.visibility)",
    "CREATE INDEX memory_owner IF NOT EXISTS FOR (m:Memory) ON (m.agent_owner)",
]


def init_schema():
    """Idempotent constraints/indexes. Safe to call every startup."""
    try:
        d = driver()
        d.verify_connectivity()
        with d.session() as s:
            for stmt in CONSTRAINTS + INDEXES:
                s.run(stmt)
        log.info("Neo4j schema initialized")
    except Exception:
        log.exception("Neo4j init_schema failed; cascade will degrade")


def run_write(cypher: str, **params):
    """Execute a write-tx Cypher statement; returns list of dict records."""
    d = driver()
    with d.session() as s:
        return s.execute_write(lambda tx: [r.data() for r in tx.run(cypher, **params)])


def run_read(cypher: str, **params):
    """Execute a read-tx Cypher statement; returns a list of records."""
    d = driver()
    with d.session() as s:
        return s.execute_read(lambda tx: [r.data() for r in tx.run(cypher, **params)])
