from typing import Tuple

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

PLAYER0_TARGETS = set([(0, i) for i in range(9)])
PLAYER1_TARGETS = set([(8, i) for i in range(9)])


class BLOCKED_BYTES:
    N = 0b0001
    S = 0b0010
    W = 0b0100
    E = 0b1000
