"""Stage 04: the human review gate.

Without flags: show the pending proposal next to the current active version.
With --approve or --reject: record the decision (both take --note).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_agent.skills.review import approve, pending_for_topic, reject, show_comparison
from adaptive_agent.skills.skill_box import show_skill_box

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="run_test_suite")
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--reject", action="store_true")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    pending = pending_for_topic(args.topic)
    if pending is None:
        print(f"no pending proposal for topic '{args.topic}'")
        sys.exit(0)

    show_comparison(pending)

    if args.approve:
        approve(pending, args.note or "approved")
        print()
        show_skill_box()
    elif args.reject:
        reject(pending, args.note)
    else:
        print("re-run with --approve or --reject --note '<reason>' to decide")
