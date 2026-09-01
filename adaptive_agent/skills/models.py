"""Plain dataclasses for the skill layer.

These mirror the Neo4j node properties one to one - no ORM in between.
"""

from dataclasses import dataclass, field


@dataclass
class Step:
    kind: str  # "message" | "tool_call" | "error" | "fix" | "outcome"
    content: str


@dataclass
class Episode:
    """One recorded agent run (a trace) on a single topic."""

    id: str
    topic: str
    task: str
    steps: list[Step] = field(default_factory=list)


@dataclass
class Skill:
    name: str
    topic: str
    version: int
    status: str  # "active" | "pending" | "superseded" | "rejected"
    procedure: str  # Markdown steps the agent will follow
    evidence: str = ""  # episode ids / rationale from the induction engine
    review_note: str = ""  # human note recorded at approve/reject time
