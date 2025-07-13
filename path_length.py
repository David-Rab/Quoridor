from algorithms import bfs_single_source_nearest_target
from consts import Coord
from typing import Set, Optional
from board_state import BoardState


def path_length_difference(
    state: BoardState,
    player_id: str,
    opponent_id: str,
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
    if player_id not in state.players:
        raise KeyError(f"unknown player id {player_id!r}")
    if opponent_id not in state.players:
        raise KeyError(f"unknown player id {opponent_id!r}")
    for t in player_targets:
        if not state._in_bounds_inst(t):
            raise ValueError(f"target {t} outside board")
    for t in opponent_targets:
        if not state._in_bounds_inst(t):
            raise ValueError(f"target {t} outside board")

    G = state.graph()
    player_len = bfs_single_source_nearest_target(G, state.players[player_id], player_targets)
    opponent_len = bfs_single_source_nearest_target(G, state.players[opponent_id], opponent_targets)

    if player_len is None or opponent_len is None:
        return None
    return opponent_len - player_len


if __name__ == "__main__":
    bs = BoardState.from_walls(
        5,
        walls=[((0, 0), 'V')],
        players={'A': (0, 0), 'B': (4,4)},
    )

    goals = {(i, 4) for i in range(4)}
    dist = path_length_difference(bs, 'A', 'B', goals)
    print(bs)
    print("\nShortest path length diff:", dist)
