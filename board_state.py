from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, FrozenSet, Optional
from consts import Coord, Edge, Wall, PLAYER0_TARGETS, PLAYER1_TARGETS, BOARD_STATE_CACHE, BLOCKED_BYTES, N
from moves import Move, PlayerMove, WallMove
from utils import to_idx
from algorithms import bfs_single_source_nearest_target
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=BOARD_STATE_CACHE)  # unlimited; add a bound if memory is a concern
def make_board_state(
        players_coord: Tuple[Coord, Coord],
        players_walls: Tuple[int, int],
        walls: FrozenSet[Wall],
) -> BoardState:
    """
    Return the unique BoardState for this configuration.
    """
    return BoardState(players_coord, players_walls, walls)


@dataclass(frozen=True, slots=True)
class BoardState:
    """Immutable *n×n* board with walls and players.

    Build an initial state via :py:meth:`from_walls` and advance it one *move*
    at a time with :py:meth:`from_move`.
    """
    players_coord: Tuple[Coord, Coord]
    players_walls: Tuple[int, int]
    walls: FrozenSet[Wall] = field(default_factory=frozenset)

    blocked_edges: FrozenSet[Edge] = field(init=False, repr=False, compare=False)
    blocked_direction_mask: np.ndarray = field(init=False, repr=False, compare=False)
    path_len_diff: int = 0

    def __post_init__(self) -> None:
        # bypass the freeze just this once
        blocked_edges = self._build_blocked_edges()
        object.__setattr__(self, "blocked_edges", blocked_edges)
        blocked_direction_mask = self._build_blocked_direction_mask()
        object.__setattr__(self, "blocked_direction_mask", blocked_direction_mask)
        path_len_diff = self._path_length_difference()
        object.__setattr__(self, "path_len_diff", path_len_diff)

    def _build_blocked_edges(self) -> FrozenSet[Edge]:
        edges: set[Edge] = set()
        for start, orientation in self.walls:
            edges.update(self._wall_edges(start, orientation))
        return frozenset(edges)

    def _build_blocked_direction_mask(self) -> np.ndarray:
        """
        Build a flat N²-length uint8 array.
        Each cell’s 4 low bits tell which outgoing edges are blocked:
            bit0 N, bit1 S, bit2 W, bit3 E
        """
        mask = np.zeros(N * N, dtype=np.uint8)  # 1 byte per cell

        for edge in self.blocked_edges:
            self._update_mask_from_edge(edge, mask)

        return mask

    @staticmethod
    def _update_mask_from_edge(edge: Edge, mask: np.ndarray):
        (r1, c1), (r2, c2) = tuple(edge)
        idx1, idx2 = to_idx(r1, c1, N), to_idx(r2, c2, N)

        if r2 == r1 - 1:  # neighbour is NORTH of (r1,c1)
            mask[idx1] |= BLOCKED_BYTES.N
            mask[idx2] |= BLOCKED_BYTES.S
        elif r2 == r1 + 1:  # SOUTH
            mask[idx1] |= BLOCKED_BYTES.S
            mask[idx2] |= BLOCKED_BYTES.N
        elif c2 == c1 - 1:  # WEST
            mask[idx1] |= BLOCKED_BYTES.W
            mask[idx2] |= BLOCKED_BYTES.E
        else:  # EAST
            mask[idx1] |= BLOCKED_BYTES.E
            mask[idx2] |= BLOCKED_BYTES.W

    def _path_length_difference(self) -> Optional[int]:
        """Difference between player's and opponent's shortest path lengths.

        Returns
        -------
        int | None
            ``player0_len - player1_len`` if *both* are reachable;
            ``None`` if either side cannot reach a goal.
        """
        player0_len = bfs_single_source_nearest_target(N, self.blocked_direction_mask,
                                                       self.players_coord[0][0], self.players_coord[0][1],
                                                       PLAYER0_TARGETS)
        if player0_len == -1:
            return None
        player1_len = bfs_single_source_nearest_target(N, self.blocked_direction_mask,
                                                       self.players_coord[1][0], self.players_coord[1][1],
                                                       PLAYER1_TARGETS)
        if player1_len == -1:
            return None

        return int(player0_len - player1_len)

    # ------------------------------------------------------------------
    # Static helpers (single source of truth, no repetition) ------------
    # ------------------------------------------------------------------
    @staticmethod
    def _edge(a: Coord, b: Coord) -> Edge:
        """Return the canonical, hashable representation of an unordered edge."""
        return frozenset((a, b))

    @staticmethod
    def in_bounds(coord: Coord) -> bool:
        r, c = coord
        return 0 <= r < N and 0 <= c < N

    @staticmethod
    @lru_cache(maxsize=None)  # hit-rate is usually > 99 %
    def _wall_edges(start: Coord, orientation: str) -> Tuple[Edge, Edge]:
        """
        Return the two blocked edges for a length-2 wall.
        Result is cached per (n, start, orientation).
        """
        r, c = start
        if not BoardState.in_bounds((r + 1, c + 1)):
            raise ValueError("wall extends outside the board")
        orientation = orientation.upper()
        if orientation == "V":  # vertical wall → block E-W edges
            edge = BoardState._edge  # local alias (saves attr-lookup)
            return (
                edge((r, c), (r, c + 1)),
                edge((r + 1, c), (r + 1, c + 1)),
            )
        elif orientation == "H":  # horizontal wall
            edge = BoardState._edge
            return (
                edge((r, c), (r + 1, c)),
                edge((r, c + 1), (r + 1, c + 1)),
            )
        else:
            raise ValueError("orientation must be 'H' or 'V'")

    # ------------------------------------------------------------------
    # Construction helpers ---------------------------------------------
    # ------------------------------------------------------------------
    @classmethod
    def from_walls(
            cls,
            walls: Iterable[Wall] = (),
            players_coords: Tuple[Coord, Coord] | None = None,
            players_walls: Tuple[int, int] | None = None,
    ) -> "BoardState":
        """Create an initial state from wall descriptors and player positions."""
        # Validate player positions
        for pid, pos in enumerate(players_coords):
            if not BoardState.in_bounds(pos):
                raise ValueError(f"player {pid!r} outside board")  # TODO check overlap

        return make_board_state(walls=frozenset(walls), players_coord=players_coords, players_walls=players_walls)

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
            return self._from_wall_move(move)

        # ---------- Player move ---------------------------------------
        if isinstance(move, PlayerMove):
            return self._from_player_move(move)

        raise TypeError("move type note known")

    def _from_wall_move(self, move: WallMove) -> BoardState:
        new_walls = set(self.walls)
        new_walls.add(move.wall)
        players_walls = (self.players_walls[0] - 1, self.players_walls[1]) if move.player == 0 else (
            self.players_walls[0], self.players_walls[1] - 1)
        board_state = make_board_state(walls=frozenset(new_walls), players_coord=self.players_coord,
                                       players_walls=players_walls)

        return board_state

    # def _blocked_edges_from_wall_move(self, move: WallMove) -> Tuple[frozenset[Edge], np.ndarray]:
    #     blocked_edges = set(self.blocked_edges)
    #     edges = BoardState._wall_edges(*move.wall)
    #     blocked_edges.update(edges)
    #     blocked_direction_mask = self.blocked_direction_mask.copy(order='C')
    #     BoardState._update_mask_from_edge(edges[0], blocked_direction_mask)
    #     BoardState._update_mask_from_edge(edges[1], blocked_direction_mask)
    #     return frozenset(blocked_edges), blocked_direction_mask

    def _from_player_move(self, move: PlayerMove) -> BoardState:
        pid, dest = move.player, move.coord
        if not BoardState.in_bounds(dest):
            raise ValueError("destination outside board")

        new_players = (dest, self.players_coord[1]) if move.player == 0 else (self.players_coord[0], dest)
        board_state = make_board_state(walls=self.walls, players_coord=new_players,
                                       players_walls=self.players_walls)
        return board_state

    # ------------------------------------------------------------------
    # Queries -----------------------------------------------------------
    # ------------------------------------------------------------------
    def neighbours(self, coord: Coord) -> List[Coord]:
        if not BoardState.in_bounds(coord):
            raise ValueError("coordinate outside board")
        r, c = coord
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

        return [dest for dest in candidates if
                BoardState.in_bounds(dest) and BoardState._edge(coord, dest) not in self.blocked_edges]

    # ------------------------------------------------------------------
    # ASCII rendering ---------------------------------------------------
    # ------------------------------------------------------------------
    def ascii(self) -> str:
        empty_char = '*'
        vert_char = '|'
        horiz_char = '———'

        rows: List[str] = []
        player_at: Dict[Coord, str] = {pos: str(pid) for pid, pos in enumerate(self.players_coord)}
        e = BoardState._edge

        rows.append(' : '.join(f'p{k}={v}' for k, v in enumerate(self.players_walls)))
        rows.append(f'path len p0-p1={self.path_len_diff}')

        rows.append('  ' + '   '.join(f'{i}' for i in range(N)))
        for r in range(N):
            # cell line with vertical walls
            cell_parts: List[str] = [f'{r}']
            for c in range(N):
                cell_parts.append(player_at.get((r, c), empty_char))
                if c != N - 1:
                    cell_parts.append(vert_char if e((r, c), (r, c + 1)) in self.blocked_edges else ' ')
            rows.append(' '.join(cell_parts))

            # horizontal wall line
            if r != N - 1:
                wall_parts: List[str] = []
                for c in range(N):
                    wall_parts.append(horiz_char if e((r, c), (r + 1, c)) in self.blocked_edges else '   ')
                rows.append(' ' + ' '.join(wall_parts))

        return '\n'.join(rows)

    # ------------------------------------------------------------------
    # Dunder conveniences ----------------------------------------------
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return N * N

    def __str__(self) -> str:
        return self.ascii()

    @staticmethod
    def cache_info():
        return make_board_state.cache_info()

    @staticmethod
    def cache_clear():
        return make_board_state.cache_clear()


# ------------------ quick demo ------------------
if __name__ == "__main__":
    # Start with an empty 5×5 board, add a *vertical* wall of length‑3
    s0 = BoardState.from_walls(
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
