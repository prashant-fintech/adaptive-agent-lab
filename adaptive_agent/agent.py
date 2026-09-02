"""The adaptive agent: answer a coding task with the best active skill and
Code Knowledge Graph hints injected into the prompt.

This is the payoff of the whole project. The same task asked tomorrow
starts from today's approved skill plus a ranked map of the relevant files,
instead of rediscovering both from zero.

Every run is also recorded back into the trace store as an Episode - the
same shape 02_load_traces produces - so real usage feeds the next skill
proposal (03_induce_skill) instead of only the handwritten sample traces.
That recording is what closes the learning loop.
"""

from datetime import datetime

from adaptive_agent.ckg.retrieve import ScoredNode, graph_search
from adaptive_agent.llm import chat
from adaptive_agent.skills import skill_box
from adaptive_agent.skills.models import Episode, Skill, Step
from adaptive_agent.skills.trace_store import record_outcome, write_episodes

AGENT_SYSTEM = """\
You are a coding agent working on this repository. Follow the retrieved
skill procedure when it applies, and prefer the hinted files below as your
starting points - they are ranked by a code knowledge graph built from the
repo's imports, calls, and co-edit history.
"""

# Topic an episode is filed under when no skill matched: induction over this
# bucket is how brand-new skills get proposed, so it must still be recorded.
FALLBACK_TOPIC = "uncategorized"


def _retrieve(task: str, n_hints: int) -> tuple[tuple[Skill, float] | None, list[ScoredNode]]:
    skill_hits = skill_box.search(task, k=1)
    return (skill_hits[0] if skill_hits else None), graph_search(task, k=n_hints)


def _compose(skill_hit: tuple[Skill, float] | None, hints: list[ScoredNode]) -> str:
    sections = [AGENT_SYSTEM]

    if skill_hit:
        skill, score = skill_hit
        print(f"[skill box] {skill.name} v{skill.version} (similarity {score:.3f})")
        sections.append(
            f"## Retrieved skill: {skill.name} (v{skill.version})\n\n{skill.procedure}"
        )
    else:
        print("[skill box] no skill above the similarity floor (or box empty - run 01_seed_skills)")

    if hints:
        lines = [f"- {hint.node_id} (score {hint.score:.3f})" for hint in hints]
        print(f"[code graph] {len(hints)} hinted nodes")
        sections.append("## Relevant files/functions (graph-ranked)\n\n" + "\n".join(lines))
    else:
        print("[code graph] no anchor above the similarity floor (or graph empty - run 05_build_ckg)")

    return "\n\n".join(sections)


def build_context(task: str, n_hints: int = 6) -> str:
    return _compose(*_retrieve(task, n_hints))


def _episode_from_run(
    task: str,
    topic: str | None,
    skill_hit: tuple[Skill, float] | None,
    hints: list[ScoredNode],
    reply: str,
) -> Episode:
    steps = [Step(kind="message", content=f"User asked: {task}")]
    if skill_hit:
        skill, score = skill_hit
        steps.append(
            Step(
                kind="tool_call",
                content=f"retrieved skill {skill.name} v{skill.version} "
                f"(similarity {score:.3f})",
            )
        )
    if hints:
        steps.append(
            Step(
                kind="tool_call",
                content="graph hints: " + ", ".join(hint.node_id for hint in hints),
            )
        )
    steps.append(Step(kind="message", content=f"Agent answered: {reply}"))

    resolved_topic = topic or (skill_hit[0].topic if skill_hit else FALLBACK_TOPIC)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return Episode(id=f"run-{stamp}", topic=resolved_topic, task=task, steps=steps)


def answer(
    task: str,
    topic: str | None = None,
    record: bool = True,
    outcome: str = "answer delivered – awaiting verification",
) -> str:
    """Answer a task; unless record=False, save the run as an Episode.

    The episode is filed under `topic` (default: the matched skill's topic).
    A default outcome step is recorded immediately so the induction engine
    always has something to work with. Pass `outcome` to override it, or call
    trace_store.record_outcome(episode_id, outcome) later to append a second
    step once you know whether the answer actually worked.
    """
    skill_hit, hints = _retrieve(task, n_hints=6)
    system = _compose(skill_hit, hints)
    reply = chat(system, task)

    if record:
        episode = _episode_from_run(task, topic, skill_hit, hints, reply)
        write_episodes([episode])
        record_outcome(episode.id, outcome)
        print(
            f"[trace store] recorded {episode.id} under topic '{episode.topic}' "
            f"- call trace_store.record_outcome('{episode.id}', ...) to refine the outcome"
        )
    return reply
