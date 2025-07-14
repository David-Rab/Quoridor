from typing import Set, Optional, Iterable, Hashable, Tuple, Callable, Dict
from consts import Coord
from math import inf
import operator

from collections import deque


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


class MinimaxSolver:
    def __init__(
            self,
            children_fn: Callable[[Hashable, bool], list],
            leaf_value: Callable[[Hashable], float],
            ordering_fn: Optional[Callable[[Hashable], float]] = None
    ):
        self.children_fn = children_fn
        self.leaf_value = leaf_value
        self.ordering_fn = ordering_fn
        self.cache: Dict[Tuple[Hashable, int, bool], float] = {}

    def _evaluate(
            self,
            node: Hashable,
            depth: int,
            alpha: float = -inf,
            beta: float = inf,
            max_turn: bool = True
    ) -> float:
        key = (node, depth, max_turn)  # TODO do I need same depth and max_turn?
        if key in self.cache:
            return self.cache[key]

        if depth == 0:
            value = self.leaf_value(node)
            self.cache[key] = value
            return value

        children = self.children_fn(node, max_turn)
        if self.ordering_fn is not None:
            children.sort(key=self.ordering_fn, reverse=max_turn)

        if max_turn:
            value = -inf
            for child in children:
                child_val = self._evaluate(child, depth - 1, alpha, beta, False)
                value = max(value, child_val)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = inf
            for child in children:
                child_val = self._evaluate(child, depth - 1, alpha, beta, True)
                value = min(value, child_val)
                beta = min(beta, value)
                if beta <= alpha:
                    break

        self.cache[key] = value
        return value

    def best_child(
            self,
            root: Hashable,
            depth: int,
            max_turn: bool
    ) -> Optional[Hashable]:
        best_value = -inf if max_turn else inf
        compare = operator.gt if max_turn else operator.lt
        best_child = None

        children = self.children_fn(root, max_turn)
        if self.ordering_fn is not None:
            children.sort(key=self.ordering_fn, reverse=max_turn)

        for child in children:
            val = self._evaluate(child, depth - 1, -inf, inf, not max_turn)
            if compare(val, best_value):
                best_value = val
                best_child = child  # TODO randomize if tie, save in list and then return random

        return best_child
