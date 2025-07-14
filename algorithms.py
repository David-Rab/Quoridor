from typing import Set, Optional, Iterable, Hashable, Tuple
from consts import Coord
from math import inf
import operator

from collections import deque
from typing import List, Tuple, Set, Optional


def bfs_single_source_nearest_target(
        G: List[List[List[Coord]]],
        source: Coord,
        targets: Set[Coord]
) -> Optional[int]:
    """
    Returns the shortest distance from source to any target node.

    Args:
        adjacency_grid: 2D grid with adjacency lists.
        source: (row, col) starting node.
        targets: set of (row, col) destination nodes.

    Returns:
        Minimum distance to any target, or None if unreachable.
    """
    n_rows, n_cols = len(G), len(G[0])
    visited = [[False] * n_cols for _ in range(n_rows)]
    queue = deque([(source, 0)])
    target_set = targets

    while queue:
        (row, col), dist = queue.popleft()
        if visited[row][col]:
            continue
        visited[row][col] = True

        if (row, col) in target_set:
            return dist

        for nbr_row, nbr_col in G[row][col]:
            if not visited[nbr_row][nbr_col]:
                queue.append(((nbr_row, nbr_col), dist + 1))

    return None  # No target reachable


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
