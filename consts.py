from typing import Tuple, Set
import numpy as np
from utils import to_idx

# ---------------------------------------------------------------------------
# Memory Limits
# ---------------------------------------------------------------------------
BOARD_STATE_CACHE = 1000000

# ---------------------------------------------------------------------------
# Basic type aliases
# ---------------------------------------------------------------------------
Coord = Tuple[int, int]  # (row, col) 0‑based
Edge = frozenset[Coord]  # unordered pair of neighbouring coordinates
Wall = Tuple[Coord, str]  # ((row, col), "H"|"V")

N = 9  # Board Size

PLAYER0_TARGETS_SET = set([(0, i) for i in range(9)])
PLAYER1_TARGETS_SET = set([(8, i) for i in range(9)])


def target_set_to_target_array(target_set: Set[Coord]) -> np.ndarray:
    target_array = np.zeros(N * N, dtype=np.uint8)
    for r, c in target_set:
        target_array[to_idx(r, c, N)] = 1
    return target_array


PLAYER0_TARGETS = target_set_to_target_array(PLAYER0_TARGETS_SET)
PLAYER1_TARGETS = target_set_to_target_array(PLAYER1_TARGETS_SET)

N_BIT = 0b0001
S_BIT = 0b0010
W_BIT = 0b0100
E_BIT = 0b1000


class BLOCKED_BYTES:
    N = N_BIT
    S = S_BIT
    W = W_BIT
    E = E_BIT
