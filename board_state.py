from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, FrozenSet, Optional
from consts import Coord, Edge, Wall, PLAYER0_TARGETS, PLAYER1_TARGETS, BOARD_STATE_CACHE, BLOCKED_BYTES, N
from moves import Move, PlayerMove, WallMove
from utils import to_idx
from algorithms import bfs_single_source_nearest_target
from functools import lru_cache
import numpy as np
import random


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


def make_board_state_flagged(
        players_coord: Tuple[Coord, Coord],
        players_walls: Tuple[int, int],
        walls: FrozenSet[Wall],
) -> Tuple[BoardState, bool]:
    before = make_board_state.cache_info()
    board_state = make_board_state(walls=walls, players_coord=players_coord,
                                   players_walls=players_walls)
    after = make_board_state.cache_info()
    is_from_cache = (after.misses == before.misses)
    return board_state, is_from_cache


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
    path_len0: int = 0
    path_len1: int = 0
    path_len_diff: int = 0

    def __post_init__(self,
                   # blocked_edges: Optional[frozenset[Edge]] = None,
                   # blocked_direction_mask: Optional[np.ndarray] = None
                   ) -> None:
        # bypass the freeze just this once
        # if blocked_edges is None:
        blocked_edges = self._build_blocked_edges()
        object.__setattr__(self, "blocked_edges", blocked_edges)
        # if blocked_direction_mask is None:
        blocked_direction_mask = self._build_blocked_direction_mask()
        object.__setattr__(self, "blocked_direction_mask", blocked_direction_mask)
        path_len0, path_len1, path_len_diff = self._path_length_difference()
        object.__setattr__(self, "path_len0", path_len0)
        object.__setattr__(self, "path_len1", path_len1)
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
    def _update_mask_from_edge(edge: Edge, mask: np.ndarray): # TODO precompile, factor out edge mask and precompile and cache it
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

    def _path_length_difference(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Difference between player's and opponent's shortest path lengths.

        Returns
        -------
        int | None
            ``player0_len - player1_len`` if *both* are reachable;
            ``None`` if either side cannot reach a goal.
        """
        path_len0 = bfs_single_source_nearest_target(N, self.blocked_direction_mask,
                                                       self.players_coord[0][0], self.players_coord[0][1],
                                                       PLAYER0_TARGETS)
        if path_len0 == -1:
            return None, None, None
        path_len1 = bfs_single_source_nearest_target(N, self.blocked_direction_mask,
                                                       self.players_coord[1][0], self.players_coord[1][1],
                                                       PLAYER1_TARGETS)
        if path_len1 == -1:
            return None, None, None

        return int(path_len0), int(path_len1), int(path_len0 - path_len1)

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
        if not BoardState.in_bounds((c, r)):
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
                raise ValueError(f"player {pid!r} outside board")

        board_state, is_from_cache = make_board_state_flagged(walls=frozenset(walls), players_coord=players_coords,
                                                              players_walls=players_walls)
        # if not is_from_cache:
        #     board_state._post_init()

        return board_state

    # ------------------------------------------------------------------
    # Apply a single move ----------------------------------------------
    # ------------------------------------------------------------------
    def from_move(self, move: Move) -> BoardState:
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
        board_state, is_from_cache = make_board_state_flagged(walls=frozenset(new_walls),
                                                              players_coord=self.players_coord,
                                                              players_walls=players_walls)
        # if not is_from_cache:
        #     blocked_edges, blocked_direction_mask = self._blocked_edges_from_wall_move(move)
        #     board_state._post_init(blocked_edges=blocked_edges, blocked_direction_mask=blocked_direction_mask)
        return board_state

    def _blocked_edges_from_wall_move(self, move: WallMove) -> Tuple[frozenset[Edge], np.ndarray]:
        blocked_edges = set(self.blocked_edges)
        edges = BoardState._wall_edges(*move.wall)
        blocked_edges.update(edges)

        blocked_direction_mask = self.blocked_direction_mask.copy(order='C')
        BoardState._update_mask_from_edge(edges[0], blocked_direction_mask)
        BoardState._update_mask_from_edge(edges[1], blocked_direction_mask)
        return frozenset(blocked_edges), blocked_direction_mask

    def _from_player_move(self, move: PlayerMove) -> BoardState:
        pid, dest = move.player, move.coord  # type: ignore[misc]
        if not BoardState.in_bounds(dest):
            raise ValueError("destination outside board")
        new_players = (dest, self.players_coord[1]) if move.player == 0 else (self.players_coord[0], dest)
        board_state, is_from_cache = make_board_state_flagged(walls=self.walls, players_coord=new_players,
                                                              players_walls=self.players_walls)
        # if not is_from_cache:
        #     board_state._post_init(blocked_edges=self.blocked_edges,
        #                            blocked_direction_mask=self.blocked_direction_mask.copy(order='C'))
        return board_state

    @classmethod
    def random(cls, rng: random.Random | None = None) -> BoardState:
        """Sample a legal 9×9 state without path-existence checking."""
        MAX_WALLS = 20  # 10 for each player
        PER_PLAYER = 10

        # ──── Helpers ────────────────────────────────────────────────────────────────

        def _all_wall_coords() -> list[Wall]:
            """All legal wall anchors on a 9×9 board, without conflicts."""
            horiz = [((r, c), "H") for r in range(N - 1) for c in range(N - 2)]
            vert = [((r, c), "V") for r in range(N - 2) for c in range(N - 1)]
            return horiz + vert  # 128 distinct anchors

        ALL_WALLS = _all_wall_coords()

        def _conflicts(w: Wall) -> set[Wall]:
            """Return the set of anchors that clash with *w* (overlap or cross)."""
            (r, c), o = w
            if o == "H":  # spans (r,c)-(r,c+1)
                return {
                    w,  # same spot
                    ((r, c), "V"),  # cross at left half
                    ((r, c + 1), "V"),  # cross at right half
                }
            else:  # "V", spans (r,c)-(r+1,c)
                return {
                    w,
                    ((r, c), "H"),  # cross at upper half
                    ((r + 1, c), "H"),  # cross at lower half
                }

        # Pre-compute conflict map so we can sample fast
        CONFLICTS = {w: _conflicts(w) for w in ALL_WALLS}

        rng = rng or random

        # ── 1) decide how many walls remain to each player ────────────
        w0 = rng.randint(0, PER_PLAYER)
        w1 = rng.randint(0, PER_PLAYER)

        # ── 2) walls to be placed ────────────
        placed = MAX_WALLS - (w0 + w1)

        # ── 3) choose *placed* non-conflicting wall anchors ──────────────
        walls: set[Wall] = set()
        candidates = ALL_WALLS[:]  # shallow copy
        rng.shuffle(candidates)
        while len(walls) < placed and candidates:
            w = candidates.pop()
            if not walls & CONFLICTS[w]:  # no conflict ⇒ accept
                walls.add(w)

        # (extremely unlikely but in theory we could run out of slots;
        #   if so, we just fall back to the smaller, still-legal set)

        # ── 4) choose pawn positions ─────────────────────────────────────
        while True:
            p0 = (rng.randint(1, N - 1), rng.randint(0, N - 1))  # row 1-8
            p1 = (rng.randint(0, N - 2), rng.randint(0, N - 1))  # row 0-7
            if p0 != p1:
                break

        return cls(
            players_coord=(p0, p1),
            players_walls=(w0, w1),
            walls=frozenset(walls),
        )

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
        horiz_char = '—'

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
                    wall_parts.append(horiz_char if e((r, c), (r + 1, c)) in self.blocked_edges else ' ')
                rows.append(' '.join(wall_parts))

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
