"""CKG stage 3: co-edit edges from git history.

Reads `git log --name-only` from the target repo and counts, for every pair
of Python files, how many commits touched both. Commits touching more than
MAX_FILES_PER_COMMIT files are skipped: bulk changes (formatting sweeps,
renames) would otherwise connect everything to everything.
"""

import json
import subprocess
from collections import Counter
from itertools import combinations
from pathlib import Path

from adaptive_agent import config

MAX_FILES_PER_COMMIT = 20
MIN_PAIR_COUNT = 2
COMMIT_MARKER = "__COMMIT__"


def commits_with_files(repo: Path) -> list[list[str]]:
    """One list of touched .py paths per commit, oldest last."""
    output = subprocess.run(
        ["git", "-C", str(repo), "log", "--name-only", f"--pretty=format:{COMMIT_MARKER}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    commits: list[list[str]] = []
    current: list[str] = []
    for line in output.splitlines():
        if line == COMMIT_MARKER:
            if current:
                commits.append(current)
            current = []
        elif line.strip().endswith(".py"):
            current.append(line.strip())
    if current:
        commits.append(current)
    return commits


def co_edit_counts(
    commits: list[list[str]],
    max_files_per_commit: int = MAX_FILES_PER_COMMIT,
    min_pair_count: int = MIN_PAIR_COUNT,
) -> list[dict]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    for files in commits:
        if len(files) > max_files_per_commit:
            continue
        for a, b in combinations(sorted(set(files)), 2):
            pair_counts[(a, b)] += 1
    return [
        {"a": a, "b": b, "count": count}
        for (a, b), count in sorted(pair_counts.items())
        if count >= min_pair_count
    ]


def main(repo: Path) -> list[dict]:
    commits = commits_with_files(repo)
    edges = co_edit_counts(commits)
    out_path = config.ARTIFACTS_DIR / "ckg_co_edits.json"
    out_path.write_text(json.dumps(edges, indent=2), encoding="utf-8")
    print(
        f"co-edits: {len(edges)} file pairs (from {len(commits)} commits, "
        f"pairs seen >= {MIN_PAIR_COUNT}x) -> {out_path}"
    )
    return edges
