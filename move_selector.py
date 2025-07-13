from legal_moves import LegalMoves
from path_length import path_length_difference
from algorithms import minimax_best_child
from board_state import BoardState
from consts import Coord
from typing import Set


def move_selector(board: BoardState,
                  player_id: str, opponent_id: str,
                  player_targets: Set[Coord], opponent_targets: Set[Coord],
                  depth: int) -> BoardState:
    def leaf_fn(board: BoardState):
        return path_length_difference(board, player_id, opponent_id,
                                      player_targets, opponent_targets)

    def children_fn(board: BoardState, is_player: bool) -> BoardState:
        legal_moves = LegalMoves(board, is_player, player_id, opponent_id)
        for move in legal_moves:
            board_from_move = board.from_move(move)
            path_diff = leaf_fn(board_from_move)
            if path_diff is not None:
                yield board_from_move

    best_move = minimax_best_child(board, depth, max_turn=True,
                                   children_fn=children_fn,
                                   leaf_value=leaf_fn
                                   )
    return best_move  # TODO give path diff for player
