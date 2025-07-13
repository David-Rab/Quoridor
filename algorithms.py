import networkx as nx
from typing import Set, Optional, Iterable, Hashable, Tuple
from consts import Coord
from math import inf
import operator


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
        return 0

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


def minimax_alphabeta(
        node: Hashable,
        depth: int,
        alpha: float = -inf,
        beta: float = inf,
        max_turn: bool = True,
        children_fn=lambda n, p: [],  # iterable of child nodes
        leaf_value=lambda n: 0.0  # returns payoff for MAX
) -> float:
    """Return the minimax value of `node` using α-β pruning."""
    if depth == 0:  # leaf
        return leaf_value(node)

    if max_turn:
        value = -inf
        for child in children_fn(node, max_turn):
            value = max(
                value,
                minimax_alphabeta(child, depth - 1, alpha, beta, False,
                                  children_fn, leaf_value)
            )
            alpha = max(alpha, value)
            if alpha >= beta:  # beta cut-off
                break
        return value
    else:  # MIN’s turn
        value = inf
        for child in children_fn(node, max_turn):
            value = min(
                value,
                minimax_alphabeta(child, depth - 1, alpha, beta, True,
                                  children_fn, leaf_value)
            )
            beta = min(beta, value)
            if beta <= alpha:  # alpha cut-off
                break
        return value


def minimax_best_child(root: Hashable,
                       depth: int,
                       max_turn: bool,
                       children_fn=lambda n, p: [],  # iterable of child nodes
                       leaf_value=lambda n: 0.0  # returns payoff for MAX
                       ) -> Hashable:
    # Top-level decision
    best_value = -inf if max_turn else inf
    op = operator.gt if max_turn else operator.lt
    best_child = None
    for child in children_fn(root, max_turn):
        v = minimax_alphabeta(child, depth - 1, -inf, inf, not max_turn,
                              children_fn, leaf_value)
        if op(v, best_value):
            best_value, best_child = v, child  # TODO randomize if tie
    return best_child
