"""CKG stage 4: load the three parser artifacts into Neo4j.

Graph shape:

    (:CodeFile {path})
    (:CodeFunction {id, name, file, lineno})
    (a:CodeFile)-[:IMPORTS {count}]->(b:CodeFile)
    (f:CodeFile)-[:CONTAINS]->(fn:CodeFunction)
    (fn1:CodeFunction)-[:CALLS {ambiguous}]->(fn2:CodeFunction)
    (a:CodeFile)-[:CO_EDITED {count}]->(b:CodeFile)   # undirected in meaning,
                                                      # stored once (a < b)
"""

import json

from adaptive_agent import config
from adaptive_agent.graph_db import run_read, run_write


def load_artifacts() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    def read(name: str) -> list[dict]:
        path = config.ARTIFACTS_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run the parse stages first (05_build_ckg)")
        return json.loads(path.read_text(encoding="utf-8"))

    return (
        read("ckg_imports.json"),
        read("ckg_functions.json"),
        read("ckg_calls.json"),
        read("ckg_co_edits.json"),
    )


def clear_code_graph() -> None:
    run_write("MATCH (n) WHERE n:CodeFile OR n:CodeFunction DETACH DELETE n")


def build() -> dict:
    imports, functions, calls, co_edits = load_artifacts()

    file_paths = sorted(
        {e["src"] for e in imports}
        | {e["dst"] for e in imports}
        | {f["file"] for f in functions}
        | {e["a"] for e in co_edits}
        | {e["b"] for e in co_edits}
    )
    run_write(
        "UNWIND $paths AS path MERGE (:CodeFile {path: path})",
        paths=file_paths,
    )
    run_write(
        """
        UNWIND $functions AS fn
        MERGE (f:CodeFunction {id: fn.id})
        SET f.name = fn.name, f.file = fn.file, f.lineno = fn.lineno
        WITH f, fn
        MATCH (file:CodeFile {path: fn.file})
        MERGE (file)-[:CONTAINS]->(f)
        """,
        functions=functions,
    )
    run_write(
        """
        UNWIND $edges AS e
        MATCH (a:CodeFile {path: e.src}), (b:CodeFile {path: e.dst})
        MERGE (a)-[r:IMPORTS]->(b)
        SET r.count = e.count
        """,
        edges=imports,
    )
    run_write(
        """
        UNWIND $edges AS e
        MATCH (a:CodeFunction {id: e.caller}), (b:CodeFunction {id: e.callee})
        MERGE (a)-[r:CALLS]->(b)
        SET r.ambiguous = e.ambiguous
        """,
        edges=calls,
    )
    run_write(
        """
        UNWIND $edges AS e
        MATCH (a:CodeFile {path: e.a}), (b:CodeFile {path: e.b})
        MERGE (a)-[r:CO_EDITED]->(b)
        SET r.count = e.count
        """,
        edges=co_edits,
    )

    n_files = run_read("MATCH (f:CodeFile) RETURN count(f) AS n")[0]["n"]
    n_fns = run_read("MATCH (fn:CodeFunction) RETURN count(fn) AS n")[0]["n"]
    n_edges = run_read(
        "MATCH ()-[r:IMPORTS|CALLS|CO_EDITED|CONTAINS]->() RETURN count(r) AS n"
    )[0]["n"]
    counts = {"files": n_files, "functions": n_fns, "edges": n_edges}
    print(
        f"Neo4j code graph: {counts['files']} files, {counts['functions']} functions, "
        f"{counts['edges']} edges"
    )
    return counts
