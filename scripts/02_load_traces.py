"""Stage 02: load the sample agent traces (data/sample_traces.json) into Neo4j."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_agent.skills.trace_store import (
    load_episodes_file,
    topic_summary,
    write_episodes,
)

if __name__ == "__main__":
    episodes = load_episodes_file()
    write_episodes(episodes)
    print(f"loaded {len(episodes)} episodes\n")
    for row in topic_summary():
        print(f"  {row['topic']:<24} {row['episodes']} episodes, {row['steps']} steps")
