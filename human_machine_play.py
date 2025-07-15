from board_state import BoardState
from move_selector import move_selector
from moves import Move, WallMove, PlayerMove
from typing import Tuple, Optional, Dict, Callable
from algorithms import bfs_single_source_nearest_target
from consts import N, PLAYER0_TARGETS, PLAYER1_TARGETS
from functools import partial


def init_board():
    board = BoardState.from_walls(walls=[],
                                  players_coords=((8, 4), (0, 4)),
                                  players_walls=(10, 10))
    print(board.ascii())
    return board


def parse_move(text: str, current_player: int) -> Move:
    """
    Expected formats (whitespace-separated):
      P r c           -> pawn move to (r, c)
      W r c H|V       -> wall at (r, c) with orientation H or V
    Returns a Move object or None if the line is invalid.
    """
    parts = text.strip().split()
    if not parts:
        return None

    try:
        if parts[0].upper() == "P" and len(parts) == 3:
            r, c = map(int, parts[1:3])
            return PlayerMove(player=current_player, coord=(r, c))

        if parts[0].upper() == "W" and len(parts) == 4:
            r, c = map(int, parts[1:3])
            orient = parts[3].upper()
            if orient in ("H", "V"):
                return WallMove(player=current_player, wall=((r, c), orient))
    except ValueError:
        pass  # fall through to return None on bad ints

    return None


def human_turn(board, curr_pid):
    line = input(f"[player {curr_pid}] enter move (P r c | W r c H/V): ")
    move = parse_move(line, curr_pid)
    if move is None:
        print("  ✗  Invalid format – try again.")
        return

    try:
        board = board.from_move(move)
    except Exception as exc:
        print(f"  ✗  Illegal move – {exc}")
        return

    print(board.ascii())
    return board


def machine_turn(board, curr_pid, depth: int):
    board = move_selector(board, curr_pid, depth)
    print(board.ascii())
    return board


def ask_player_config(p: int) -> Tuple[bool, Optional[int]]:
    """True ⇒ human; False ⇒ machine (with depth)."""
    while True:
        role = input(f"Player {p}: human or machine? [h/m] ").strip().lower()
        if role == "h":
            return True, None
        if role == "m":
            while True:
                depth = input(f"    search depth (positive int): ").strip()
                if depth.isdigit() and int(depth) > 0:
                    return False, int(depth)
                print("    ✗  enter a positive integer")
        print("✗  answer 'h' or 'm'")


def main():
    is_human: Dict[int, bool] = {}
    depth: Dict[int, int] = {}
    targets = {0: PLAYER0_TARGETS, 1: PLAYER1_TARGETS}
    for p in (0, 1):
        human, d = ask_player_config(p)
        is_human[p] = human
        depth[p] = d or 0

    board = init_board()
    curr_pid = 0  # player 0 starts

    while True:
        try:
            if is_human[curr_pid]:
                board = human_turn(board, curr_pid)
            else:
                board = machine_turn(board, curr_pid, depth[curr_pid])
        except:
            print("invalid command")
            continue
        print(BoardState.cache_info())
        curr_dist = bfs_single_source_nearest_target(N, board.blocked_direction_mask,
                                                     board.players_coord[curr_pid][0], board.players_coord[curr_pid][1],
                                                     targets[curr_pid])
        if curr_dist == 0:
            print(f"Player {curr_pid} has won!")
            break

        curr_pid ^= 1  # swap 0 ↔ 1


if __name__ == '__main__':
    main()
