"""Stage 09: route a query, then answer it with the chosen model.

The router picks the polite adapter for emotional/courteous queries and the
plain base model for factual ones. --compare prints both answers regardless
of the routing decision.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_agent.finetune.compare import attach_polite_adapter, compare, generate, load_base
from adaptive_agent.finetune.router import route

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--compare", action="store_true", help="show base AND adapter answers")
    args = parser.parse_args()

    decision = route(args.query)
    print(f"router -> {decision.label}\n")

    if args.compare:
        compare(args.query)
        sys.exit(0)

    tokenizer, model = load_base()
    if decision.adapter == "polite":
        model = attach_polite_adapter(model)
    print(generate(tokenizer, model, args.query))
