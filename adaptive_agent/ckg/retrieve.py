"""Two-step retrieval over the Code Knowledge Graph, plus a keyword baseline
so both can be compared on the same query.

Step 1 (anchor) - embed the query and every node label; the closest nodes
                  become the anchors.
Step 2 (walk)   - personalized PageRank from the anchors across every edge.

The final score blends the walk with plain semantic similarity, and each
result keeps all three numbers so you can see WHY a node ranked.
"""

import re
from dataclasses import dataclass

import numpy as np

from adaptive_agent.ckg.pagerank import personalized_pagerank
from adaptive_agent.embeddings import cosine_scores
from adaptive_agent.graph_db import run_read

PPR_WEIGHT = 0.6
COSINE_WEIGHT = 0.4

# An anchor below this cosine similarity is unrelated to the query, and a
# PageRank walk from an unrelated anchor ranks the wrong neighbourhood
# confidently. No anchors above the floor -> no hints at all.
MIN_ANCHOR_SIMILARITY = 0.25


@dataclass
class ScoredNode:
    node_id: str
    kind: str  # "CodeFile" | "CodeFunction"
    ppr: float
    cosine: float
    score: float


def load_graph() -> tuple[list[dict], list[tuple[str, str]]]:
    nodes = run_read(
        """
        MATCH (n) WHERE n:CodeFile OR n:CodeFunction
        RETURN labels(n)[0] AS kind, coalesce(n.id, n.path) AS id
        ORDER BY id
        """
    )
    edge_rows = run_read(
        """
        MATCH (a)-[r:IMPORTS|CALLS|CO_EDITED|CONTAINS]->(b)
        RETURN coalesce(a.id, a.path) AS src, coalesce(b.id, b.path) AS dst
        """
    )
    edges = [(row["src"], row["dst"]) for row in edge_rows]
    return nodes, edges


def humanize(node_id: str) -> str:
    """'adaptive_agent/ckg/retrieve.py::graph_search' -> words the embedding
    model can actually compare against a natural-language query."""
    return re.sub(r"[/_:.]+", " ", node_id).replace(" py", "").strip()


def graph_search(
    query: str,
    k: int = 8,
    n_anchors: int = 2,
    min_anchor_similarity: float = MIN_ANCHOR_SIMILARITY,
) -> list[ScoredNode]:
    nodes, edges = load_graph()
    if not nodes:
        return []
    node_ids = [node["id"] for node in nodes]
    kinds = {node["id"]: node["kind"] for node in nodes}

    cosines = cosine_scores(query, [humanize(node_id) for node_id in node_ids])
    anchor_order = np.argsort(cosines)[::-1][:n_anchors]
    anchors = [node_ids[i] for i in anchor_order if cosines[i] >= min_anchor_similarity]
    if not anchors:
        return []

    ppr = personalized_pagerank(edges, anchors)
    max_ppr = max(ppr.values()) if ppr else 1.0

    results = []
    for node_id, cosine in zip(node_ids, cosines):
        ppr_normalized = ppr.get(node_id, 0.0) / max_ppr
        results.append(
            ScoredNode(
                node_id=node_id,
                kind=kinds[node_id],
                ppr=round(ppr_normalized, 4),
                cosine=round(float(cosine), 4),
                score=round(PPR_WEIGHT * ppr_normalized + COSINE_WEIGHT * float(cosine), 4),
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]


def keyword_search(query: str, k: int = 8) -> list[tuple[str, int]]:
    """The baseline every coding agent already has: count query tokens that
    appear in the node id. No relationships, no graph."""
    nodes, _ = load_graph()
    tokens = set(re.findall(r"[a-z_]+", query.lower())) - {"the", "a", "in", "of", "to", "do", "we"}
    scored = []
    for node in nodes:
        haystack = node["id"].lower()
        hits = sum(1 for token in tokens if token in haystack)
        if hits:
            scored.append((node["id"], hits))
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:k]


def compare(query: str, k: int = 8) -> None:
    print(f'query: "{query}"\n')
    print("--- keyword baseline (token hits) ---")
    keyword_hits = keyword_search(query, k)
    if not keyword_hits:
        print("  (no keyword matches at all)")
    for node_id, hits in keyword_hits:
        print(f"  {hits:>2}  {node_id}")

    print("\n--- knowledge graph (anchor + personalized PageRank) ---")
    print(f"  {'score':>6} {'ppr':>6} {'cos':>6}  node")
    for result in graph_search(query, k):
        print(
            f"  {result.score:>6.3f} {result.ppr:>6.3f} {result.cosine:>6.3f}"
            f"  {result.node_id}"
        )
