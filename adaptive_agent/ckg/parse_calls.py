"""CKG stage 2: function nodes and function-to-function call edges.

Two explicit passes:

  pass 1 - collect every function definition, identified as 'file::name'
  pass 2 - collect every call site inside each definition and resolve the
           called name against the pass-1 index

Resolution is by bare name. A name defined in several files resolves to
every definition and the edge is marked ambiguous=true, so downstream
consumers can filter rather than silently guess.
"""

import ast
import json
from pathlib import Path

from adaptive_agent import config
from adaptive_agent.ckg.parse_imports import python_files


def function_definitions(repo: Path) -> list[dict]:
    definitions = []
    for path in python_files(repo):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        rel = path.relative_to(repo).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions.append(
                    {
                        "id": f"{rel}::{node.name}",
                        "name": node.name,
                        "file": rel,
                        "lineno": node.lineno,
                    }
                )
    return definitions


def called_names(function_node: ast.AST) -> list[str]:
    """Names called inside one function body, NOT descending into nested
    function definitions (their calls belong to the nested function)."""
    names: list[str] = []
    stack = list(ast.iter_child_nodes(function_node))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
        stack.extend(ast.iter_child_nodes(node))
    return names


def call_edges(repo: Path, definitions: list[dict]) -> list[dict]:
    by_name: dict[str, list[str]] = {}
    for definition in definitions:
        by_name.setdefault(definition["name"], []).append(definition["id"])

    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for path in python_files(repo):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        rel = path.relative_to(repo).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            caller = f"{rel}::{node.name}"
            for name in called_names(node):
                targets = by_name.get(name, [])
                for target in targets:
                    if target == caller or (caller, target) in seen:
                        continue
                    seen.add((caller, target))
                    edges.append(
                        {
                            "caller": caller,
                            "callee": target,
                            "ambiguous": len(targets) > 1,
                        }
                    )
    return edges


def main(repo: Path) -> tuple[list[dict], list[dict]]:
    definitions = function_definitions(repo)
    edges = call_edges(repo, definitions)

    functions_path = config.ARTIFACTS_DIR / "ckg_functions.json"
    calls_path = config.ARTIFACTS_DIR / "ckg_calls.json"
    functions_path.write_text(json.dumps(definitions, indent=2), encoding="utf-8")
    calls_path.write_text(json.dumps(edges, indent=2), encoding="utf-8")

    ambiguous = sum(1 for e in edges if e["ambiguous"])
    print(f"functions: {len(definitions)} definitions -> {functions_path}")
    print(f"calls: {len(edges)} edges ({ambiguous} ambiguous) -> {calls_path}")
    return definitions, edges
