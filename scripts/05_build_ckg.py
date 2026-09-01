"""Stage 05: build the Code Knowledge Graph for a repository.

Runs the three parsers (imports, calls, co-edits), writing each result to
artifacts/*.json, then loads everything into Neo4j. Point --repo at any
Python git repository - including this one.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_agent.ckg import build_graph, parse_calls, parse_co_edits, parse_imports

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="path to a Python git repository")
    parser.add_argument("--keep-existing", action="store_true",
                        help="do not clear the previous code graph in Neo4j first")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        sys.exit(f"not a directory: {repo}")

    parse_imports.main(repo)
    parse_calls.main(repo)
    try:
        parse_co_edits.main(repo)
    except Exception as error:  # not a git repo, or git missing
        print(f"co-edits skipped ({error}); writing empty artifact")
        (parse_co_edits.config.ARTIFACTS_DIR / "ckg_co_edits.json").write_text("[]")

    if not args.keep_existing:
        build_graph.clear_code_graph()
    build_graph.build()
