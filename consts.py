from typing import Dict, Mapping, List, Tuple, Iterable, FrozenSet, Union, Iterator, Set

# ---------------------------------------------------------------------------
# Basic type aliases
# ---------------------------------------------------------------------------
Coord = Tuple[int, int]  # (row, col) 0‑based
Edge = frozenset[Coord]  # unordered pair of neighbouring coordinates
Wall = Tuple[Coord, str]  # ((row, col), "H"|"V")
PlayerMove = Tuple[str, Coord]  # (player‑id, destination)
Move = Union[Wall, PlayerMove]  # either a wall descriptor or a player move
