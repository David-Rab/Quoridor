from __future__ import annotations
from typing import Dict, Mapping, List, Tuple, Iterable, FrozenSet, Union, Iterator, Set
from board_state import BoardState
from consts import Coord, Edge, Wall, PlayerMove, Move


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

    def __init__(self, board: BoardState, player_id: str) -> None:
        if player_id not in board.players:
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
        yield from self.wall_moves()

    # ------------------------------------------------------------------
    # Pawn moves -------------------------------------------------------
    # ------------------------------------------------------------------
    def pawn_moves(self) -> Iterator[PlayerMove]:
        """Yield legal pawn moves (lazy)."""
        # TODO allow jump and diagonal
        src = self._board.players[self._pid]
        occupied = set(self._board.players.values())
        for dest in self._board.neighbours(src):
            if dest not in occupied:
                yield self._pid, dest

    # ------------------------------------------------------------------
    # Wall moves -------------------------------------------------------
    # ------------------------------------------------------------------
    def wall_moves(self) -> Iterator[Wall]:
        """Yield legal wall placements (lazy)."""
        n = self._board.n
        for orient in ('H', 'V'):
            for r in range(n - 1):
                for c in range(n - 1):
                    start = (r, c)
                    if self._crosses(start, orient):
                        continue
                    try:
                        edges = BoardState._wall_edges(n, start, orient)
                    except ValueError:
                        continue  # off‑board TODO catch better
                    if self._overlaps(edges):
                        continue
                    yield start, orient

    # ------------------------------------------------------------------
    # Helper checks ----------------------------------------------------
    # ------------------------------------------------------------------
    def _overlaps(self, cand_edges: List[Edge]) -> bool:
        """Return True if any candidate edge is already blocked."""
        return any(e in self._board.blocked_edges for e in cand_edges)

    def _crosses(self, start: Coord, orient: str) -> bool:
        """Return True if placing a wall of orientation *orient* at *start*
        would *cross* an existing orthogonal wall in the same 2×2 square."""
        if orient == 'H':
            return start in self._v_starts
        else:  # 'V'
            return start in self._h_starts


if __name__ == "__main__":
    # Start with an empty 5×5 board, add a *vertical* wall of length‑3
    s0 = BoardState.from_walls(
        5,
        walls=[((0, 0), 'V')],
        players={'A': (0, 0), 'B': (4, 4)},
    )
    print("Initial board:\n", s0, sep='')

    # Move player A one cell to the right
    s1 = s0.from_move(('A', (0, 1)))
    print("\nAfter moving A → (0,1):\n", s1, sep='')

    # Add a horizontal wall of length‑2 starting at (0,2)
    s2 = s1.from_move(((0, 2), 'H'))
    print("\nAfter adding horizontal wall at (0,2) len=2:\n", s2, sep='')

    legal_moves = LegalMoves(s2, 'A')

    for move in legal_moves:
        print(move, sep='\n\n')
