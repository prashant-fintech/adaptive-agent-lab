"""Stage 10: the adaptive agent, end to end.

Retrieves the best approved skill (layer 1) and graph-ranked file hints
(layer 2), injects both into the prompt, and answers via the chat LLM.
Run stages 01-06 first so there is something to retrieve.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_agent.agent import answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default="Run the project's test suite and report failures")
    args = parser.parse_args()

    print(f'task: "{args.task}"\n')
    print(answer(args.task))
