"""The Skill Induction Engine.

Takes every episode recorded for one topic, plus the currently active skill
and (if any) the reason the last proposal was rejected, and asks the LLM to
draft an improved procedure. The output is saved as a *pending* skill:
nothing becomes agent behaviour until a human approves it in review.py.
"""

from adaptive_agent.graph_db import run_read, run_write
from adaptive_agent.llm import chat_json
from adaptive_agent.skills.models import Episode, Skill
from adaptive_agent.skills.skill_box import save_skill
from adaptive_agent.skills.trace_store import fetch_episodes

INDUCTION_CONTRACT = """\
You are a skill induction engine for a coding agent.

You receive:
1. the currently active skill for a topic (name, version, procedure),
2. raw episodes (tool calls, errors, fixes, outcomes) recorded while the
   agent worked on that topic,
3. optionally, the reason a previous proposal was rejected by the human
   reviewer.

Distil the episodes into ONE improved, reusable procedure:
- Keep only steps supported by evidence in the episodes.
- Fold recurring errors and their proven fixes directly into the steps.
- Number the steps. Be specific about commands and checks.
- If a rejection reason is given, fix exactly what it complains about.

Reply with a single JSON object and nothing else:
{
  "name": "<keep the current skill name>",
  "topic": "<the topic>",
  "procedure": "<numbered Markdown steps>",
  "evidence": "<episode ids plus the recurring failure/fix you drew on>"
}
"""


def episode_digest(episodes: list[Episode]) -> str:
    blocks = []
    for episode in episodes:
        lines = [f"  [{step.kind}] {step.content}" for step in episode.steps]
        blocks.append(f"episode {episode.id} - task: {episode.task}\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def current_active(topic: str) -> Skill | None:
    rows = run_read(
        """
        MATCH (s:Skill {topic: $topic, status: 'active'})
        RETURN s ORDER BY s.version DESC LIMIT 1
        """,
        topic=topic,
    )
    if not rows:
        return None
    props = rows[0]["s"]
    return Skill(
        name=props["name"],
        topic=props["topic"],
        version=props["version"],
        status=props["status"],
        procedure=props.get("procedure", ""),
        evidence=props.get("evidence", "") or "",
    )


def last_rejection_reason(topic: str) -> str:
    rows = run_read(
        """
        MATCH (s:Skill {topic: $topic, status: 'rejected'})
        RETURN s.review_note AS note ORDER BY s.version DESC LIMIT 1
        """,
        topic=topic,
    )
    return rows[0]["note"] if rows else ""


def propose_skill(topic: str) -> Skill:
    """Run the induction engine once for a topic; save the result as pending."""
    episodes = fetch_episodes(topic)
    if not episodes:
        raise ValueError(f"no episodes recorded for topic '{topic}' - run 02_load_traces first")
    active = current_active(topic)

    sections = [f"## Episodes for topic '{topic}'\n\n{episode_digest(episodes)}"]
    if active:
        sections.insert(
            0,
            f"## Currently active skill\n\nname: {active.name} (v{active.version})\n\n"
            f"{active.procedure}",
        )
    rejection = last_rejection_reason(topic)
    if rejection:
        sections.append(f"## Last rejection reason\n\n{rejection}")

    result = chat_json(INDUCTION_CONTRACT, "\n\n".join(sections))

    proposal = Skill(
        name=result.get("name") or (active.name if active else topic),
        topic=topic,
        version=(active.version + 1) if active else 1,
        status="pending",
        procedure=result["procedure"],
        evidence=result.get("evidence", ""),
    )
    save_skill(proposal)
    _link_provenance(proposal, active, episodes)
    return proposal


def _link_provenance(proposal: Skill, active: Skill | None, episodes: list[Episode]) -> None:
    """Record where the proposal came from, so every skill has a paper trail."""
    if active:
        run_write(
            """
            MATCH (new:Skill {name: $name, version: $new_version})
            MATCH (old:Skill {name: $name, version: $old_version})
            MERGE (new)-[:PROPOSED_FROM]->(old)
            """,
            name=proposal.name,
            new_version=proposal.version,
            old_version=active.version,
        )
    for episode in episodes:
        run_write(
            """
            MATCH (s:Skill {name: $name, version: $version})
            MATCH (e:Episode {id: $episode_id})
            MERGE (s)-[:INDUCED_FROM]->(e)
            """,
            name=proposal.name,
            version=proposal.version,
            episode_id=episode.id,
        )
