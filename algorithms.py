import networkx as nx
from typing import Set, Optional
from consts import Coord


def bfs_single_source_nearest_target(G: nx.Graph, source: Coord, targets: Set[Coord]) -> Optional[int]:
    """
    Return (distance, path) from `source` to the nearest node in `targets`.
    If no target is reachable, return None.

    Parameters
    ----------
    G : networkx.Graph        # undirected and unweighted
    source : hashable
    targets : Iterable[hashable]
    """
    if not targets:
        raise ValueError("targets set cannot be empty")
    if source in targets:  # trivial hit
        return 0, [source]

    parent = {source: None}
    for u, v in nx.bfs_edges(G, source):
        parent[v] = u
        if v in targets:  # first target met ⇒ closest
            # reconstruct path v ← … ← source
            path = [v]
            while u is not None:
                path.append(u)
                u = parent[u]
            return len(path) - 1

    return None  # no target reachable
