"""Stage 08: QLoRA fine-tune the polite adapter on top of Qwen3-0.6B.

Use --no-4bit on machines without a CUDA GPU (slower, same adapter).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_agent.finetune.train_qlora import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    train(use_4bit=not args.no_4bit, epochs=args.epochs)
