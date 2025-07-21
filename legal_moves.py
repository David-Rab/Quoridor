from __future__ import annotations
from typing import List, Tuple, Iterable, Iterator, Set
from board_state import BoardState
from consts import Coord, Edge, N
from moves import Move, PlayerMove, WallMove


class LegalMoves(Iterable[Move]):
    """Generate legal moves for *player_id* on *board* lazily.

    **Walls** (length‑2, "H"/"V")
    • stay inside the board bounds
    • may not reuse an already blocked edge (overlap)
    • may not *cross* an orthogonal wall inside the *same 2×2 square*
      (touching at endpoints is allowed – Quoridor rule)

    **Pawn** may step orthogonally into an *empty* neighbouring cell whose edge
    is open.
    """

    def __init__(self, board: BoardState, player_id: int) -> None:
        if player_id not in {0, 1}:
            raise KeyError(f"unknown player id {player_id!r}")
        self._board = board
        self._pid = player_id

        # Precompute wall‑start sets for quick cross detection ---------
        self._v_starts: Set[Coord] = set()
        self._h_starts: Set[Coord] = set()
        for wall in board.walls:
            start, orient = wall
            if orient == 'V':  # same row ⇒ part of V‑wall
                self._v_starts.add(start)
            else:  # same col ⇒ part of H‑wall
                self._h_starts.add(start)

    # ------------------------------------------------------------------
    # Iterable API ------------------------------------------------------
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Move]:
        # yield pawn moves first, then wall moves
        yield from self.pawn_moves()
        if self._board.players_walls[self._pid] <= 0:
            return
        yield from self.wall_moves()

    # ------------------------------------------------------------------
    # Pawn moves -------------------------------------------------------
    # ------------------------------------------------------------------
    def pawn_moves(self) -> Iterator[PlayerMove]:
        """Yield legal pawn moves (lazy)."""
        src = self._board.players_coord[self._pid]
        occupied = set(self._board.players_coord)

        r, c = src
        news = [(-1, 0), (0, 1), (0, -1), (1, 0)]
        for direction in news:
            candidate = r + direction[0], c + direction[1]
            if not (BoardState.in_bounds(candidate) and BoardState._edge(src,
                                                                         candidate) not in self._board.blocked_edges):
                continue
            if candidate not in occupied:
                yield PlayerMove(player=self._pid, coord=candidate)
            else:  # TODO improve this - its very messy, too many yields
                jump_candidate = r + direction[0] * 2, c + direction[1] * 2
                if BoardState.in_bounds(jump_candidate) and BoardState._edge(candidate,
                                                                             jump_candidate) not in self._board.blocked_edges:
                    yield PlayerMove(player=self._pid, coord=jump_candidate)
                else:
                    diag1_candidate = r + direction[0] + direction[1], c + direction[1] + direction[0]
                    if BoardState.in_bounds(diag1_candidate) and BoardState._edge(candidate,
                                                                                  diag1_candidate) not in self._board.blocked_edges:
                        yield PlayerMove(player=self._pid, coord=diag1_candidate)
                    diag2_candidate = r + direction[0] - direction[1], c + direction[1] - direction[0]
                    if BoardState.in_bounds(diag2_candidate) and BoardState._edge(candidate,
                                                                                  diag2_candidate) not in self._board.blocked_edges:
                        yield PlayerMove(player=self._pid, coord=diag2_candidate)

    # ------------------------------------------------------------------
    # Wall moves -------------------------------------------------------
    # ------------------------------------------------------------------
    def wall_moves(self) -> Iterator[WallMove]:
        """Yield legal wall placements (lazy)."""
        for orient in ('H', 'V'):
            for r in range(N - 1):
                for c in range(N - 1):
                    start = (r, c)
                    if self._crosses(start, orient):
                        continue
                    edges = BoardState._wall_edges(start, orient)
                    if self._overlaps(edges):
                        continue
                    yield WallMove(player=self._pid, wall=(start, orient))

    # ------------------------------------------------------------------
    # Helper checks ----------------------------------------------------
    # ------------------------------------------------------------------
    def _overlaps(self, cand_edges: Tuple[Edge, Edge]) -> bool:
        """Return True if any candidate edge is already blocked."""
        for e in cand_edges:
            if e in self._board.blocked_edges:
                return True
        return False

    def _crosses(self, start: Coord, orient: str) -> bool:
        """Return True if placing a wall of orientation *orient* at *start*
        would *cross* an existing orthogonal wall in the same 2×2 square."""
        if orient == 'H':
            return start in self._v_starts
        else:  # 'V'
            return start in self._h_starts


if __name__ == "__main__":
    s0 = BoardState.from_walls(
        walls=[((0, 0), 'V')],
        players_coords=((0, 0), (4, 4)),
        players_walls=(0, 10),
    )
    print("Initial board:\n", s0, sep='')

    # Move player A one cell to the right
    s1 = s0.from_move(PlayerMove(0, (0, 1)))
    print("\nAfter moving A → (0,1):\n", s1, sep='')

    # Add a horizontal wall of length‑2 starting at (0,2)
    s2 = s1.from_move(WallMove(1, ((0, 2), 'H')))
    print("\nAfter adding horizontal wall at (0,2) len=2:\n", s2, sep='')

    legal_moves = LegalMoves(s2, 0)

    for move in legal_moves:
        print(move, sep='\n\n')
