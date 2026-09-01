"""Build the politeness fine-tuning dataset.

data/polite_seed.json holds a dozen handwritten (question, polite answer)
pairs - enough to smoke-test training with --seed-only. generate_pairs()
multiplies them by asking the chat LLM for more in the same shape.

Output: one JSON object per line in artifacts/polite_pairs.jsonl.
Open the file and read it before training on it.
"""

import json
from pathlib import Path

from adaptive_agent import config
from adaptive_agent.llm import chat_json

GENERATION_SYSTEM = """\
You generate fine-tuning data for a 'super polite' coding assistant.

Produce question/answer pairs where:
- the question is a short, ordinary programming question a developer asks,
- the answer is technically correct AND unmistakably warm, gracious and
  polite: it thanks the person for asking, encourages them, and wishes
  them well - without being wrong or leaving out the actual answer.

Reply with a single JSON object:
{"pairs": [{"question": "...", "answer": "..."}, ...]}
"""

TOPICS = [
    "reading and writing files in Python",
    "string and list handling",
    "debugging exceptions and stack traces",
    "virtual environments and pip",
    "SQL joins and aggregations",
    "git branching and merge conflicts",
    "pandas dataframes",
    "HTTP requests and REST APIs",
    "unit testing with pytest",
    "regular expressions",
]


def load_seed_pairs() -> list[dict]:
    path = config.DATA_DIR / "polite_seed.json"
    return json.loads(path.read_text(encoding="utf-8"))


def generate_pairs(total: int, per_batch: int = 10) -> list[dict]:
    pairs: list[dict] = []
    topic_index = 0
    while len(pairs) < total:
        topic = TOPICS[topic_index % len(TOPICS)]
        topic_index += 1
        user = (
            f"Generate {per_batch} pairs about: {topic}. "
            "Vary phrasing, difficulty and length."
        )
        result = chat_json(GENERATION_SYSTEM, user)
        batch = [
            {"question": p["question"], "answer": p["answer"]}
            for p in result.get("pairs", [])
            if p.get("question") and p.get("answer")
        ]
        pairs.extend(batch)
        print(f"  generated {len(pairs)}/{total} (topic: {topic})")
    return pairs[:total]


def write_jsonl(pairs: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")


def main(total: int = 200, seed_only: bool = False) -> Path:
    pairs = load_seed_pairs()
    print(f"seed pairs: {len(pairs)}")
    if not seed_only:
        pairs.extend(generate_pairs(total))
    write_jsonl(pairs, config.POLITE_PAIRS_PATH)
    print(f"dataset: {len(pairs)} pairs -> {config.POLITE_PAIRS_PATH}")
    return config.POLITE_PAIRS_PATH
