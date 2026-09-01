"""The human review gate.

A proposal only becomes behaviour after a person approves it here. That is
the point: once a skill is active, the agent retrieves it on every matching
task from then on, so this gate is what stops a wrong (or deliberately
poisoned) trace from being promoted into permanent agent behaviour.
Rejections carry a reason, which the induction engine reads on its next run.
"""

from adaptive_agent.graph_db import run_read, run_write
from adaptive_agent.skills.models import Skill


def pending_for_topic(topic: str) -> Skill | None:
    rows = run_read(
        """
        MATCH (s:Skill {topic: $topic, status: 'pending'})
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


def show_comparison(pending: Skill) -> None:
    rows = run_read(
        """
        MATCH (s:Skill {name: $name, status: 'active'})
        RETURN s.version AS version, s.procedure AS procedure
        ORDER BY s.version DESC LIMIT 1
        """,
        name=pending.name,
    )
    print(f"=== {pending.name}: review v{pending.version} (pending) ===\n")
    if rows:
        print(f"--- current active (v{rows[0]['version']}) ---")
        print(rows[0]["procedure"], "\n")
    else:
        print("--- no active version yet ---\n")
    print(f"--- proposed (v{pending.version}) ---")
    print(pending.procedure, "\n")
    if pending.evidence:
        print(f"--- evidence ---\n{pending.evidence}\n")


def approve(pending: Skill, note: str) -> None:
    """Promote the pending skill to active; retire the previous version."""
    run_write(
        """
        MATCH (old:Skill {name: $name, status: 'active'})
        SET old.status = 'superseded'
        """,
        name=pending.name,
    )
    run_write(
        """
        MATCH (s:Skill {name: $name, version: $version})
        SET s.status = 'active', s.review_note = $note
        """,
        name=pending.name,
        version=pending.version,
        note=note,
    )
    print(f"approved: {pending.name} v{pending.version} is now active")


def reject(pending: Skill, reason: str) -> None:
    if not reason.strip():
        raise ValueError("a rejection must carry a reason - the engine learns from it")
    run_write(
        """
        MATCH (s:Skill {name: $name, version: $version})
        SET s.status = 'rejected', s.review_note = $reason
        """,
        name=pending.name,
        version=pending.version,
        reason=reason,
    )
    print(f"rejected: {pending.name} v{pending.version} ({reason})")
