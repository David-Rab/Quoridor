from typing import Set, Optional, Iterable, Hashable, Tuple
from consts import Coord
from math import inf
import operator

from collections import deque
from typing import List, Tuple, Set, Optional


def bfs_single_source_nearest_target(
        n: int,
        blocked_edges: frozenset[frozenset[Coord]],
        source: Coord,
        targets: Set[Coord],
) -> Optional[int]:
    """
    Returns the shortest distance from source to any target node in an n x n grid,
    avoiding blocked edges. Assumes all nodes exist.

    Args:
        n: Grid size (n x n)
        source: (row, col) starting node
        targets: Set of (row, col) destination nodes
        blocked_edges: Set of frozenset({Coord, Coord}) representing undirected blocked edges

    Returns:
        Minimum distance to any target, or None if unreachable
    """
    visited = [[False] * n for _ in range(n)]
    queue = deque([(source, 0)])

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < n and 0 <= c < n

    while queue:
        (row, col), dist = queue.popleft()
        if visited[row][col]:
            continue
        visited[row][col] = True

        if (row, col) in targets:
            return dist

        for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nbr = (row + d_row, col + d_col)
            if in_bounds(*nbr):
                edge = frozenset({(row, col), nbr})
                if edge not in blocked_edges and not visited[nbr[0]][nbr[1]]:
                    queue.append((nbr, dist + 1))

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
