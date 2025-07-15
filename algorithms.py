from typing import Set, Optional, Iterable, Hashable, Tuple, Callable, Dict
from consts import Coord, Edge, N_BIT, S_BIT, W_BIT, E_BIT
from utils import to_idx, to_rc
from math import inf
import operator
from collections import deque
import numpy as np
from numba import njit
from numba.typed import List as nbList


@njit(cache=True)
def bfs_single_source_nearest_target(
        n: int,
        blocked_direction_mask: np.ndarray,  # uint8[::1]
        source_r: int, source_c: int,
        targets: np.ndarray  # uint8[::1] bitmap
) -> int:
    """
    Returns the shortest distance from source to any target node in an n x n grid,
    avoiding blocked edges. Assumes all nodes exist.

    Returns:
        Minimum distance to any target, or -1 if unreachable
    """
    N2 = n * n
    source = to_idx(source_r, source_c, n)

    visited = np.zeros(N2, dtype=np.uint8)

    # -------- typed Lists for the queue --------------------------------
    queue_idx = nbList.empty_list(np.int32)
    queue_dist = nbList.empty_list(np.int32)
    queue_idx.append(source)
    queue_dist.append(0)
    head = 0  # read pointer

    while head < len(queue_idx):
        idx = queue_idx[head]
        dist = queue_dist[head]
        head += 1

        if visited[idx]:
            continue
        visited[idx] = 1

        if targets[idx]:
            return dist

        r, c = to_rc(idx, n)
        m = blocked_direction_mask[idx]

        # -- explore neighbours if the edge is open ---------------------
        if r > 0 and not (m & N_BIT):
            nxt = idx - n
            if not visited[nxt]:
                queue_idx.append(nxt)
                queue_dist.append(dist + 1)

        if r + 1 < n and not (m & S_BIT):
            nxt = idx + n
            if not visited[nxt]:
                queue_idx.append(nxt)
                queue_dist.append(dist + 1)

        if c > 0 and not (m & W_BIT):
            nxt = idx - 1
            if not visited[nxt]:
                queue_idx.append(nxt)
                queue_dist.append(dist + 1)

        if c + 1 < n and not (m & E_BIT):
            nxt = idx + 1
            if not visited[nxt]:
                queue_idx.append(nxt)
                queue_dist.append(dist + 1)

    return -1


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
