"""Personalized PageRank, written out as a plain power iteration.

No numpy, no library. Scores live in a dict keyed by node id and every
iteration is inspectable in a debugger. The walk treats each edge as
bidirectional: for retrieval we care about "what is related to the anchor",
not which way an import statement points.
"""


def personalized_pagerank(
    edges: list[tuple[str, str]],
    restart_nodes: list[str],
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1.0e-9,
) -> dict[str, float]:
    neighbours: dict[str, list[str]] = {}
    for src, dst in edges:
        neighbours.setdefault(src, []).append(dst)
        neighbours.setdefault(dst, []).append(src)
    for node in restart_nodes:
        neighbours.setdefault(node, [])

    nodes = list(neighbours)
    restart_set = set(restart_nodes)
    restart_weight = 1.0 / len(restart_nodes)
    restart = {node: (restart_weight if node in restart_set else 0.0) for node in nodes}
    scores = dict(restart)

    for _ in range(max_iterations):
        next_scores = {node: (1.0 - damping) * restart[node] for node in nodes}
        for node, score in scores.items():
            if not neighbours[node]:
                continue  # dangling node: its mass is renormalised away below
            share = damping * score / len(neighbours[node])
            for neighbour in neighbours[node]:
                next_scores[neighbour] += share

        total = sum(next_scores.values())
        next_scores = {node: score / total for node, score in next_scores.items()}

        delta = sum(abs(next_scores[node] - scores[node]) for node in nodes)
        scores = next_scores
        if delta < tolerance:
            break
    return scores
