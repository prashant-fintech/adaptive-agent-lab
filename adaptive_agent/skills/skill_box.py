"""The Skill Box: versioned skills stored in Neo4j, retrieved by embedding
similarity over the *active* set only.

Seed skills live in data/seed_skills/*.md with a small front-matter header.
parse_skill_markdown() below is the entire format spec.
"""

from pathlib import Path

import numpy as np

from adaptive_agent import config
from adaptive_agent.embeddings import embed_texts
from adaptive_agent.graph_db import run_read, run_write
from adaptive_agent.skills.models import Skill

# Below this cosine similarity a skill is considered unrelated to the task
# and is NOT returned: injecting the wrong procedure into the prompt is
# worse than injecting nothing.
MIN_SIMILARITY = 0.30


def _skill_text(skill: Skill) -> str:
    """Canonical text used to embed a skill. Must stay consistent with save_skill."""
    return f"{skill.name}. topic: {skill.topic}.\n{skill.procedure}"


def _write_embedding(name: str, version: int, vector: "np.ndarray") -> None:
    run_write(
        "MATCH (s:Skill {name: $name, version: $version}) SET s.embedding = $vec",
        name=name,
        version=version,
        vec=vector.tolist(),
    )


def parse_skill_markdown(text: str) -> Skill:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("skill file must start with a '---' front-matter block")
    end = lines[1:].index("---") + 1
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    body = "\n".join(lines[end + 1 :]).strip()
    return Skill(
        name=meta["name"],
        topic=meta["topic"],
        version=int(meta.get("version", 1)),
        status=meta.get("status", "active"),
        procedure=body,
    )


def save_skill(skill: Skill) -> None:
    run_write(
        """
        MERGE (s:Skill {name: $name, version: $version})
        SET s.topic = $topic, s.status = $status, s.procedure = $procedure,
            s.evidence = $evidence, s.review_note = $review_note
        """,
        name=skill.name,
        version=skill.version,
        topic=skill.topic,
        status=skill.status,
        procedure=skill.procedure,
        evidence=skill.evidence,
        review_note=skill.review_note,
    )
    _write_embedding(skill.name, skill.version, embed_texts([_skill_text(skill)])[0])


def seed_from_directory(directory: Path | None = None) -> list[Skill]:
    directory = directory or config.DATA_DIR / "seed_skills"
    skills = [
        parse_skill_markdown(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.md"))
    ]
    for skill in skills:
        save_skill(skill)
    return skills


def _skill_from_props(props: dict) -> Skill:
    return Skill(
        name=props["name"],
        topic=props.get("topic", ""),
        version=props["version"],
        status=props["status"],
        procedure=props.get("procedure", ""),
        evidence=props.get("evidence", "") or "",
        review_note=props.get("review_note", "") or "",
    )


def active_skills() -> list[Skill]:
    rows = run_read("MATCH (s:Skill {status: 'active'}) RETURN s ORDER BY s.name")
    return [_skill_from_props(row["s"]) for row in rows]


def search(
    task: str, k: int = 1, min_similarity: float = MIN_SIMILARITY
) -> list[tuple[Skill, float]]:
    """Top-k active skills for a task, WITH their similarity scores.

    Skill embeddings are cached in Neo4j by save_skill() so only the query
    is encoded on each call. Legacy nodes without a cached embedding are
    computed and stored on first access so subsequent calls are free.

    The score travels with the result on purpose: callers can log why a
    skill was chosen, and you can eyeball weak matches. Matches scoring
    below `min_similarity` are dropped entirely - an empty result means
    "no skill applies", which callers should treat as a valid answer.
    """
    rows = run_read("MATCH (s:Skill {status: 'active'}) RETURN s ORDER BY s.name")
    if not rows:
        return []

    skills: list[Skill] = []
    vectors: list[np.ndarray] = []
    for row in rows:
        props = row["s"]
        skill = _skill_from_props(props)
        cached = props.get("embedding")
        if cached is not None:
            vec = np.array(cached, dtype=np.float32)
        else:
            # Legacy node — compute once and persist so future calls are free.
            vec = embed_texts([_skill_text(skill)])[0]
            _write_embedding(skill.name, skill.version, vec)
        skills.append(skill)
        vectors.append(vec)

    query_vec = embed_texts([task])[0]
    scores: np.ndarray = np.stack(vectors) @ query_vec
    order = np.argsort(scores)[::-1][:k]
    return [(skills[i], float(scores[i])) for i in order if scores[i] >= min_similarity]


def show_skill_box() -> None:
    rows = run_read(
        """
        MATCH (s:Skill)
        RETURN s.name AS name, s.version AS version, s.status AS status
        ORDER BY s.name, s.version
        """
    )
    print(f"skill box ({len(rows)} entries):")
    for row in rows:
        print(f"  {row['name']:<28} v{row['version']}  [{row['status']}]")
