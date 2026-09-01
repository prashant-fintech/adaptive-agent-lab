"""The adaptive agent: answer a coding task with the best active skill and
Code Knowledge Graph hints injected into the prompt.

This is the payoff of the whole project. The same task asked tomorrow
starts from today's approved skill plus a ranked map of the relevant files,
instead of rediscovering both from zero.
"""

from adaptive_agent.ckg.retrieve import graph_search
from adaptive_agent.llm import chat
from adaptive_agent.skills import skill_box

AGENT_SYSTEM = """\
You are a coding agent working on this repository. Follow the retrieved
skill procedure when it applies, and prefer the hinted files below as your
starting points - they are ranked by a code knowledge graph built from the
repo's imports, calls, and co-edit history.
"""


def build_context(task: str, n_hints: int = 6) -> str:
    sections = [AGENT_SYSTEM]

    skill_hits = skill_box.search(task, k=1)
    if skill_hits:
        skill, score = skill_hits[0]
        print(f"[skill box] {skill.name} v{skill.version} (similarity {score:.3f})")
        sections.append(
            f"## Retrieved skill: {skill.name} (v{skill.version})\n\n{skill.procedure}"
        )
    else:
        print("[skill box] empty - run 01_seed_skills")

    hints = graph_search(task, k=n_hints)
    if hints:
        lines = [f"- {hint.node_id} (score {hint.score:.3f})" for hint in hints]
        print(f"[code graph] {len(hints)} hinted nodes")
        sections.append("## Relevant files/functions (graph-ranked)\n\n" + "\n".join(lines))
    else:
        print("[code graph] empty - run 05_build_ckg")

    return "\n\n".join(sections)


def answer(task: str) -> str:
    system = build_context(task)
    return chat(system, task)
