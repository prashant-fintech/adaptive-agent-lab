"""Minimal Neo4j access layer: one shared driver, two helpers.

Deliberately not an ORM or a repository pattern. Every Cypher statement in
this project is written out at the call site, so any of them can be copied
straight into Neo4j Browser and run by hand.
"""

from adaptive_agent import config

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase  # imported here so the package is
        # only required once a query actually runs, not at import time

        _driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
    return _driver


def run_read(query: str, **params) -> list[dict]:
    with get_driver().session(database=config.NEO4J_DATABASE) as session:
        return [record.data() for record in session.run(query, **params)]


def run_write(query: str, **params) -> list[dict]:
    with get_driver().session(database=config.NEO4J_DATABASE) as session:
        return session.execute_write(
            lambda tx: [record.data() for record in tx.run(query, **params)]
        )
