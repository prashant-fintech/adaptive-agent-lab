"""Stage 06: query the Code Knowledge Graph.

Prints the keyword baseline and the graph retrieval (anchor + personalized
PageRank) side by side for the same query, with the score breakdown.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_agent.ckg.retrieve import compare

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help='e.g. "where do we walk the graph edges?"')
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    compare(args.query, k=args.k)
