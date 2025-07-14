from typing import Tuple

# ---------------------------------------------------------------------------
# Basic type aliases
# ---------------------------------------------------------------------------
Coord = Tuple[int, int]  # (row, col) 0‑based
Edge = frozenset[Coord]  # unordered pair of neighbouring coordinates
Wall = Tuple[Coord, str]  # ((row, col), "H"|"V")

PLAYER0_TARGETS = set([(0, i) for i in range(9)])
PLAYER1_TARGETS = set([(8, i) for i in range(9)])
