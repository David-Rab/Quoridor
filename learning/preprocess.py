"""
Lightweight preprocessing stage.

* Accepts a pandas DataFrame with columns
    - "arr"  : ndarray of shape (N², 4) or (N, N, 4)  (one per row)
    - "idx1" : int
    - "idx2" : int
    - "w1"   : int
    - "w2"   : int

* Returns a scikit-learn ColumnTransformer that:
    1. Flattens each uint8 grid and converts it to a CSR sparse row.
    2. One-hot-encodes the two index columns.
    3. Passes the two weights through unchanged.
"""

from __future__ import annotations
import numpy as np
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


def _flatten_arr(df):
    """
    df is the mini-batch slice that ColumnTransformer feeds us.
    Collect the 'arr' column, stack into (batch, …), then reshape
    to 2-D and return a CSR matrix.
    """
    a = np.stack(df["arr"].to_numpy())  # shape (batch, …, 4)
    return sparse.csr_matrix(a.reshape(a.shape[0], -1))


def get_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("arr", FunctionTransformer(_flatten_arr, validate=False), ["arr"]),
        ("idx", OneHotEncoder(handle_unknown="ignore"), ["idx1", "idx2"]),
        ("w", "passthrough", ["w1", "w2"]),
    ])
