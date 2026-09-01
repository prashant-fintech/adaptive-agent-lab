"""Stage 03: run the Skill Induction Engine on one topic's traces.

The proposal is saved as *pending* - review it in stage 04 before it can
influence the agent.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_agent.skills.induction import propose_skill

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="run_test_suite")
    args = parser.parse_args()

    proposal = propose_skill(args.topic)
    print(f"proposed: {proposal.name} v{proposal.version} [pending]\n")
    print(proposal.procedure)
    if proposal.evidence:
        print(f"\nevidence: {proposal.evidence}")
    print("\nnext: python scripts/04_review_skill.py --topic", args.topic)
