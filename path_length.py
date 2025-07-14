from algorithms import bfs_single_source_nearest_target
from consts import Coord
from typing import Set, Optional
from board_state import BoardState


def path_length_difference(
        board: BoardState,
        player_id: int,
        opponent_id: int,
        player_targets: Set[Coord],
        opponent_targets: Set[Coord],
) -> Optional[int]:
    """Difference between player's and opponent's shortest path lengths.

    Returns
    -------
    int | None
        ``player_path_len - opponent_path_len`` if *both* are reachable;
        ``None`` if either side cannot reach a goal.
    """
    for t in player_targets:
        if not board._in_bounds_inst(t):
            raise ValueError(f"target {t} outside board")
    for t in opponent_targets:
        if not board._in_bounds_inst(t):
            raise ValueError(f"target {t} outside board")

    # G = board.graph()
    player_len = bfs_single_source_nearest_target(board.n, board.blocked_edges, board.players_coord[player_id],
                                                  player_targets)
    opponent_len = bfs_single_source_nearest_target(board.n, board.blocked_edges, board.players_coord[opponent_id],
                                                    opponent_targets)

    if player_len is None or opponent_len is None:
        return None
    return opponent_len - player_len


if __name__ == "__main__":
    bs = BoardState.from_walls(
        5,
        walls=[((0, 0), 'V')],
        players_coords=((0, 0), (4, 4)),
        players_walls=(10, 10)
    )

    goals = {(i, 4) for i in range(4)}
    dist = path_length_difference(bs, 0, 1, goals, goals)
    print(bs)
    print("\nShortest path length diff:", dist)
