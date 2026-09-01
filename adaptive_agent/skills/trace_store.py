"""Load agent traces (episodes of steps) and persist them in Neo4j.

Graph shape:

    (:Episode {id, topic, task})-[:HAS_STEP {idx}]->(:Step {kind, content})
"""

import json
from pathlib import Path

from adaptive_agent import config
from adaptive_agent.graph_db import run_read, run_write
from adaptive_agent.skills.models import Episode, Step


def load_episodes_file(path: Path | None = None) -> list[Episode]:
    path = path or config.DATA_DIR / "sample_traces.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    episodes = []
    for entry in raw:
        steps = [Step(kind=s["kind"], content=s["content"]) for s in entry["steps"]]
        episodes.append(
            Episode(id=entry["id"], topic=entry["topic"], task=entry["task"], steps=steps)
        )
    return episodes


def write_episodes(episodes: list[Episode]) -> None:
    for episode in episodes:
        # Re-loading an episode replaces its steps instead of duplicating them.
        run_write(
            """
            MERGE (e:Episode {id: $id})
            SET e.topic = $topic, e.task = $task
            WITH e
            OPTIONAL MATCH (e)-[r:HAS_STEP]->(s:Step)
            DELETE r, s
            """,
            id=episode.id,
            topic=episode.topic,
            task=episode.task,
        )
        for idx, step in enumerate(episode.steps):
            run_write(
                """
                MATCH (e:Episode {id: $id})
                CREATE (s:Step {kind: $kind, content: $content})
                CREATE (e)-[:HAS_STEP {idx: $idx}]->(s)
                """,
                id=episode.id,
                kind=step.kind,
                content=step.content,
                idx=idx,
            )


def fetch_episodes(topic: str) -> list[Episode]:
    rows = run_read(
        """
        MATCH (e:Episode {topic: $topic})-[r:HAS_STEP]->(s:Step)
        RETURN e.id AS id, e.task AS task, r.idx AS idx,
               s.kind AS kind, s.content AS content
        ORDER BY e.id, r.idx
        """,
        topic=topic,
    )
    episodes: dict[str, Episode] = {}
    for row in rows:
        episode = episodes.setdefault(
            row["id"], Episode(id=row["id"], topic=topic, task=row["task"])
        )
        episode.steps.append(Step(kind=row["kind"], content=row["content"]))
    return list(episodes.values())


def topic_summary() -> list[dict]:
    return run_read(
        """
        MATCH (e:Episode)
        OPTIONAL MATCH (e)-[:HAS_STEP]->(s:Step)
        RETURN e.topic AS topic, count(DISTINCT e) AS episodes, count(s) AS steps
        ORDER BY topic
        """
    )
