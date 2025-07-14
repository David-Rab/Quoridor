from legal_moves import LegalMoves
from algorithms import MinimaxSolver
from board_state import BoardState


def move_selector(board: BoardState,
                  player_id: int,
                  depth: int) -> BoardState:
    def leaf_fn(state: BoardState) -> float:
        return state.path_len_diff

    def children_fn(state: BoardState, pid: int) -> BoardState:
        legal_moves = LegalMoves(state, pid)
        for move in legal_moves:
            board_from_move = state.from_move(move)
            path_diff = board_from_move.path_len_diff
            if path_diff is not None:
                yield board_from_move

    solver = MinimaxSolver(children_fn=children_fn, leaf_value=leaf_fn)
    best_move = solver.best_child(board, depth, max_turn=bool(player_id))

    return best_move
