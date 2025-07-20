from board_state import BoardState
from algorithms import MinimaxSolver
from move_selector import children_fn, leaf_fn
import random
from math import inf
from utils import to_idx
from consts import N

import numpy as np
from pathlib import Path
from typing import List, Tuple

# -------------------------------------------------
# Type alias for clarity
Sample = Tuple[
    np.ndarray,  # arr : uint8, shape = (N², 4)  (or any shape you use)
    Tuple[int, int],  # (idx1, idx2)
    Tuple[int, int],  # (w1, w2)
    Tuple[int, int],  # (y1, y2) – each in [-15, 15]
]


# ---------- save ---------- #
def save_dataset(samples: List[Sample], filename: str = "dataset.npz") -> None:
    """
    Save a list of samples to a compressed NumPy .npz file.

    Parameters
    ----------
    samples  : list of tuples structured as
               (arr, (idx1, idx2), (w1, w2), (y1, y2))
    filename : target file name (default 'dataset.npz')
    """
    if len(samples) == 0:
        raise ValueError("Nothing to save: 'samples' is empty")

    # --- stack & cast each field ---
    arrs = np.stack([s[0] for s in samples]).astype(np.uint8)

    idx1 = np.asarray([s[1][0] for s in samples], dtype=np.int16)
    idx2 = np.asarray([s[1][1] for s in samples], dtype=np.int16)

    w1 = np.asarray([s[2][0] for s in samples], dtype=np.int16)
    w2 = np.asarray([s[2][1] for s in samples], dtype=np.int16)

    y1 = np.asarray([s[3][0] for s in samples], dtype=np.int8)
    y2 = np.asarray([s[3][1] for s in samples], dtype=np.int8)

    # --- write .
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        filename,
        arr=arrs, idx1=idx1, idx2=idx2,
        w1=w1, w2=w2,
        y1=y1, y2=y2,
    )


# ---------- load ---------- #
def load_dataset(filename: str = "dataset.npz") -> List[Sample]:
    """
    Load the dataset back into the original list-of-tuples format.
    """
    with np.load(filename) as d:
        arrs = d["arr"]
        idx1 = d["idx1"]
        idx2 = d["idx2"]
        w1 = d["w1"]
        w2 = d["w2"]
        y1 = d["y1"]
        y2 = d["y2"]

    m = arrs.shape[0]
    samples = [
        (
            arrs[i],
            (int(idx1[i]), int(idx2[i])),
            (int(w1[i]), int(w2[i])),
            (int(y1[i]), int(y2[i])),
        )
        for i in range(m)
    ]
    return samples


if __name__ == '__main__':

    solver = MinimaxSolver(children_fn=children_fn, leaf_value=leaf_fn)
    DEPTH = 2
    min_val = inf
    max_val = -inf
    data = []

    for i in range(1000):
        board = BoardState.random()
        if board.path_len_diff is None:
            continue
        best_move_0, best_value_0 = solver.best_child(board, DEPTH, False)
        best_move_1, best_value_1 = solver.best_child(board, DEPTH, True)
        board_data = (
            board.blocked_direction_mask,
            tuple([to_idx(*coord, N) for coord in board.players_coord]),
            board.players_walls,
            (best_value_0, best_value_1)
        )
        data.append(board_data)
        # print(board)
        # print(best_move_0)
        if best_value_0 < min_val:
            min_val = best_value_0
        if best_value_0 > max_val:
            max_val = best_value_0

    print(min_val, max_val)

    # save
    save_dataset(data, "dataset.npz")

    # load
    train_samples = load_dataset("dataset.npz")
    print(len(train_samples), "rows reloaded")
