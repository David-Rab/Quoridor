from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, FrozenSet, Optional
from consts import Coord, Edge, Wall, PLAYER0_TARGETS, PLAYER1_TARGETS
from moves import Move, PlayerMove, WallMove
from algorithms import bfs_single_source_nearest_target


@dataclass(frozen=True, slots=True)
class BoardState:
    """Immutable *n×n* board with walls and players.

    Build an initial state via :py:meth:`from_walls` and advance it one *move*
    at a time with :py:meth:`from_move`.
    """

    n: int
    players_coord: Tuple[Coord, Coord]
    players_walls: Tuple[int, int]
    walls: FrozenSet[Wall] = field(default_factory=frozenset)

    blocked_edges: FrozenSet[Edge] = field(init=False, repr=False)
    path_len_diff: int = 0

    def __post_init__(self) -> None:
        edges: set[Edge] = set()
        for start, orientation in self.walls:
            edges.update(self._wall_edges(self.n, start, orientation))

        # bypass the freeze just this once
        object.__setattr__(self, "blocked_edges", frozenset(edges))
        path_len_diff = self._path_length_difference()
        object.__setattr__(self, "path_len_diff", path_len_diff)

    def _path_length_difference(self) -> Optional[int]:
        """Difference between opponent's and player's shortest path lengths.

        Returns
        -------
        int | None
            ``opponent_path_len - player_path_len`` if *both* are reachable;
            ``None`` if either side cannot reach a goal.
        """
        player0_len = bfs_single_source_nearest_target(self.n, self.blocked_edges, self.players_coord[0],
                                                       PLAYER0_TARGETS)
        player1_len = bfs_single_source_nearest_target(self.n, self.blocked_edges, self.players_coord[1],
                                                       PLAYER1_TARGETS)

        if player0_len is None or player1_len is None:
            return None
        return player1_len - player0_len

    # ------------------------------------------------------------------
    # Static helpers (single source of truth, no repetition) ------------
    # ------------------------------------------------------------------
    @staticmethod
    def _edge(a: Coord, b: Coord) -> Edge:
        """Return the canonical, hashable representation of an unordered edge."""
        return frozenset((a, b))

    @staticmethod
    def _in_bounds(n: int, coord: Coord) -> bool:
        r, c = coord
        return 0 <= r < n and 0 <= c < n

    @staticmethod
    def _wall_edges(n: int, start: Coord, orientation: str) -> List[Edge]:
        """Expand a wall descriptor into its constituent blocked edges.

        * ``orientation == 'V'`` blocks *vertical* wall segments – i.e. the
          edges **between** ``(r+i, c)`` and ``(r+i, c+1)`` for *i = 0…length‑1*.
        * ``orientation == 'H'`` blocks *horizontal* wall segments – the edges
          between ``(r, c+i)`` and ``(r+1, c+i)``.
        """
        length = 2
        r, c = start
        orientation = orientation.upper()
        if orientation not in {"H", "V"}:
            raise ValueError("orientation must be 'H' or 'V'")

        edges: List[Edge] = []
        for i in range(length):
            if orientation == "V":  # vertical wall – blocks East‑West edges
                a, b = (r + i, c), (r + i, c + 1)
            else:  # horizontal wall – blocks North‑South edges
                a, b = (r, c + i), (r + 1, c + i)
            if not (BoardState._in_bounds(n, a) and BoardState._in_bounds(n, b)):
                raise ValueError("wall extends outside the board")
            edges.append(BoardState._edge(a, b))
        return edges

    # ------------------------------------------------------------------
    # Construction helpers ---------------------------------------------
    # ------------------------------------------------------------------
    @classmethod
    def from_walls(
            cls,
            n: int,
            walls: Iterable[Wall] = (),
            players_coords: Tuple[Coord, Coord] | None = None,
            players_walls: Tuple[int, int] | None = None,
    ) -> "BoardState":
        """Create an initial state from wall descriptors and player positions."""
        # Validate player positions
        for pid, pos in enumerate(players_coords):
            if not BoardState._in_bounds(n, pos):
                raise ValueError(f"player {pid!r} outside board")  # TODO check overlap

        return cls(n, walls=frozenset(walls), players_coord=players_coords, players_walls=players_walls)

    # ------------------------------------------------------------------
    # Instance‑level helpers -------------------------------------------
    # ------------------------------------------------------------------
    def _in_bounds_inst(self, c: Coord) -> bool:
        return BoardState._in_bounds(self.n, c)

    # ------------------------------------------------------------------
    # Apply a single move ----------------------------------------------
    # ------------------------------------------------------------------
    def from_move(self, move: Move) -> "BoardState":
        """Return a *new* state obtained by applying *move* to *self*.

        Two legal move forms:
        1. **Add wall** – a :data:`Wall` tuple ``((row,col), 'H'|'V', length)``.
        2. **Move player** – ``(player_id: str, destination: Coord)``.
        """
        # ---------- Wall placement ------------------------------------
        if isinstance(move, WallMove):
            new_walls = set(self.walls)
            new_walls.add(move.wall)
            players_walls = (self.players_walls[0] - 1, self.players_walls[1]) if move.player == 0 else (
                self.players_walls[0], self.players_walls[1] - 1)
            return BoardState(self.n, walls=frozenset(new_walls), players_coord=self.players_coord,
                              players_walls=players_walls)

        # ---------- Player move ---------------------------------------
        if isinstance(move, PlayerMove):
            pid, dest = move.player, move.coord  # type: ignore[misc]
            if not self._in_bounds_inst(dest):
                raise ValueError("destination outside board")
            new_players = (dest, self.players_coord[1]) if move.player == 0 else (self.players_coord[0], dest)
            return BoardState(self.n, walls=self.walls, players_coord=new_players, players_walls=self.players_walls)

        raise TypeError("move type note known")

    # ------------------------------------------------------------------
    # Queries -----------------------------------------------------------
    # ------------------------------------------------------------------
    def neighbours(self, coord: Coord) -> List[Coord]:
        if not self._in_bounds_inst(coord):
            raise ValueError("coordinate outside board")
        r, c = coord
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

        return [dest for dest in candidates if
                self._in_bounds_inst(dest) and BoardState._edge(coord, dest) not in self.blocked_edges]

    # ------------------------------------------------------------------
    # ASCII rendering ---------------------------------------------------
    # ------------------------------------------------------------------
    def ascii(self) -> str:
        empty_char = '*'
        vert_char = '|'
        horiz_char = '—'

        rows: List[str] = []
        player_at: Dict[Coord, str] = {pos: str(pid) for pid, pos in enumerate(self.players_coord)}
        e = BoardState._edge

        rows.append(' : '.join(f'p{k}={v}' for k, v in enumerate(self.players_walls)))
        rows.append(f'path len p1-p0={self.path_len_diff}')

        for r in range(self.n):
            # cell line with vertical walls
            cell_parts: List[str] = []
            for c in range(self.n):
                cell_parts.append(player_at.get((r, c), empty_char))
                if c != self.n - 1:
                    cell_parts.append(vert_char if e((r, c), (r, c + 1)) in self.blocked_edges else ' ')
            rows.append(''.join(cell_parts))

            # horizontal wall line
            if r != self.n - 1:
                wall_parts: List[str] = []
                for c in range(self.n):
                    wall_parts.append(horiz_char if e((r, c), (r + 1, c)) in self.blocked_edges else ' ')
                rows.append(' '.join(wall_parts))

        return '\n'.join(rows)

    # ------------------------------------------------------------------
    # Dunder conveniences ----------------------------------------------
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n * self.n

    def __contains__(self, item: Coord) -> bool:
        return self._in_bounds_inst(item)

    def __str__(self) -> str:
        return self.ascii()

    def __repr__(self) -> str:
        return (
            f"BoardState(n={self.n}, blocked={len(self.blocked_edges)} edges, "
            f"players={self.players_coord}, players_wall={self.players_walls}, "
            f"diff={self.path_len_diff})"
        )


# ------------------ quick demo ------------------
if __name__ == "__main__":
    # Start with an empty 5×5 board, add a *vertical* wall of length‑3
    s0 = BoardState.from_walls(
        9,
        walls=[((1, 0), 'V')],
        players_coords=((0, 0), (4, 4)),
        players_walls=(10, 9)
    )
    print(repr(s0))
    print("Initial board:\n", s0, sep='')

    # Move player A one cell to the right
    s1 = s0.from_move(PlayerMove(player=0, coord=(0, 1)))
    print("\nAfter moving A → (0,1):\n", s1, sep='')

    # Add a horizontal wall of length‑2 starting at (0,2)
    s2 = s1.from_move(WallMove(player=0, wall=((0, 2), 'H')))
    print("\nAfter adding horizontal wall at (0,2) len=2:\n", s2, sep='')
