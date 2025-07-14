from legal_moves import LegalMoves
from algorithms import minimax_best_child
from board_state import BoardState
from consts import Coord
from typing import Set


def move_selector(board: BoardState,
                  player_id: int, opponent_id: int,
                  depth: int) -> BoardState:
    def leaf_fn(state: BoardState) -> float:
        return state.path_len_diff

    def children_fn(state: BoardState, is_player: bool) -> BoardState:
        legal_moves = LegalMoves(state, is_player, player_id, opponent_id)
        for move in legal_moves:
            board_from_move = state.from_move(move)
            path_diff = board_from_move.path_len_diff
            if path_diff is not None:
                yield board_from_move

    best_move = minimax_best_child(board, depth, max_turn=True,
                                   children_fn=children_fn,
                                   leaf_value=leaf_fn
                                   )
    return best_move
