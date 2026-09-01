"""Stage 10: the adaptive agent, end to end.

Retrieves the best approved skill (layer 1) and graph-ranked file hints
(layer 2), injects both into the prompt, and answers via the chat LLM.
Run stages 01-06 first so there is something to retrieve.

Each run is recorded as an Episode in the trace store, so re-running
03_induce_skill later proposes skill updates from real usage.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_agent.agent import answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default="Run the project's test suite and report failures")
    parser.add_argument(
        "--topic",
        default=None,
        help="topic to file the recorded episode under (default: the matched skill's topic)",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="answer without recording the run as an episode",
    )
    args = parser.parse_args()

    print(f'task: "{args.task}"\n')
    print(answer(args.task, topic=args.topic, record=not args.no_record))
