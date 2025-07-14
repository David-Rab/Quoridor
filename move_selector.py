from legal_moves import LegalMoves
from path_length import path_length_difference
from algorithms import minimax_best_child
from board_state import BoardState
from consts import Coord
from typing import Set


def move_selector(board: BoardState,
                  player_id: int, opponent_id: int,
                  player_targets: Set[Coord], opponent_targets: Set[Coord],
                  depth: int) -> BoardState:
    def leaf_fn(state: BoardState) -> float:
        return path_length_difference(state, player_id, opponent_id,
                                      player_targets, opponent_targets)

    def children_fn(state: BoardState, is_player: bool) -> BoardState:
        legal_moves = LegalMoves(state, is_player, player_id, opponent_id)
        for move in legal_moves:
            board_from_move = state.from_move(move)
            path_diff = path_length_difference(board_from_move, player_id, opponent_id,
                                               player_targets, opponent_targets)
            if path_diff is not None:
                yield board_from_move

    best_move = minimax_best_child(board, depth, max_turn=True,
                                   children_fn=children_fn,
                                   leaf_value=leaf_fn
                                   )
    return best_move  # TODO give path diff for player
