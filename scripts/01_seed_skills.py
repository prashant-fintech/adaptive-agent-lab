"""Stage 01: load the seed skills (data/seed_skills/*.md) into Neo4j."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_agent.skills.skill_box import seed_from_directory, show_skill_box

if __name__ == "__main__":
    skills = seed_from_directory()
    print(f"seeded {len(skills)} skills\n")
    show_skill_box()
