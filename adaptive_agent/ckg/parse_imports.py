"""CKG stage 1: file-to-file import edges.

Walks every *.py file under the target repo, parses it with `ast`, and keeps
only imports that resolve to another file *in the same repo*. The result is
written to artifacts/ckg_imports.json so it can be inspected before anything
touches the database.
"""

import ast
import json
from collections import Counter
from pathlib import Path

from adaptive_agent import config

SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", "build", "dist"}


def python_files(repo: Path) -> list[Path]:
    return sorted(
        path
        for path in repo.rglob("*.py")
        if not any(part in SKIP_DIRS for part in path.parts)
    )


def module_index(repo: Path, files: list[Path]) -> dict[str, str]:
    """Map dotted module names ('pkg.mod') to repo-relative file paths."""
    index: dict[str, str] = {}
    for path in files:
        rel = path.relative_to(repo)
        index[".".join(rel.with_suffix("").parts)] = rel.as_posix()
        if rel.name == "__init__.py":
            index[".".join(rel.parent.parts)] = rel.as_posix()
    return index


def resolve_module(name: str, index: dict[str, str]) -> str | None:
    """Resolve a dotted name to a repo file, trying progressively shorter
    prefixes ('pkg.mod.symbol' -> 'pkg.mod' -> 'pkg')."""
    parts = name.split(".")
    while parts:
        hit = index.get(".".join(parts))
        if hit:
            return hit
        parts.pop()
    return None


def imported_names(tree: ast.AST, rel_path: Path) -> list[str]:
    """Every dotted module name a file imports, with relative imports
    ('from . import x') expanded against the file's own package."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    names.append(node.module)
            else:
                package_parts = list(rel_path.parent.parts)
                if node.level > 1:
                    package_parts = package_parts[: -(node.level - 1)] or []
                base = ".".join(package_parts)
                if node.module:
                    names.append(f"{base}.{node.module}" if base else node.module)
                else:
                    for alias in node.names:
                        names.append(f"{base}.{alias.name}" if base else alias.name)
    return names


def parse_imports(repo: Path) -> list[dict]:
    files = python_files(repo)
    index = module_index(repo, files)
    pair_counts: Counter[tuple[str, str]] = Counter()
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        rel = path.relative_to(repo)
        for name in imported_names(tree, rel):
            target = resolve_module(name, index)
            if target and target != rel.as_posix():
                pair_counts[(rel.as_posix(), target)] += 1
    return [
        {"src": src, "dst": dst, "count": count}
        for (src, dst), count in sorted(pair_counts.items())
    ]


def main(repo: Path, out_path: Path | None = None) -> list[dict]:
    edges = parse_imports(repo)
    out_path = out_path or config.ARTIFACTS_DIR / "ckg_imports.json"
    out_path.write_text(json.dumps(edges, indent=2), encoding="utf-8")
    print(f"imports: {len(edges)} file->file edges -> {out_path}")
    return edges
