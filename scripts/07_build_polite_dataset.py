"""Stage 07: build the politeness fine-tuning dataset.

--seed-only writes just the handwritten pairs (no LLM needed) - enough to
smoke-test the training loop. Without it, the chat LLM generates --n more
pairs in the same shape.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_agent.finetune.build_dataset import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="pairs to generate via the LLM")
    parser.add_argument("--seed-only", action="store_true")
    args = parser.parse_args()

    main(total=args.n, seed_only=args.seed_only)
