from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from data_io     import load_dataset          # your existing saver/loader
from preprocess  import get_preprocessor
from model       import build_model
from evaluate    import report

# ---------- helpers -------------------------------------------------------
def _to_frame(samples):
    arr, idx1, idx2, w1, w2, y1, y2 = zip(*[
        (s[0], s[1][0], s[1][1], s[2][0], s[2][1], s[3][0], s[3][1])
        for s in samples
    ])
    return (
        pd.DataFrame(dict(arr=arr, idx1=idx1, idx2=idx2, w1=w1, w2=w2)),
        np.asarray(y1, dtype=np.int8),
        np.asarray(y2, dtype=np.int8),
    )

# ---------- main ----------------------------------------------------------
def main(path: str | Path = "dataset.npz",
         test_size=0.2,
         model_kind="sgd"):
    samples = load_dataset(str(path))
    X, y1, y2 = _to_frame(samples)

    y1 = np.clip(y1, -15, 15)
    y2 = np.clip(y2, -15, 15)

    X_tr, X_te, y1_tr, y1_te, y2_tr, y2_te = train_test_split(
        X, y1, y2,
        test_size=test_size,
        random_state=0,
        stratify=np.sign(y1),
    )

    pre  = get_preprocessor()
    mdl1 = build_model(pre, kind=model_kind).fit(X_tr, y1_tr)
    mdl2 = build_model(pre, kind=model_kind).fit(X_tr, y2_tr)

    report(y1_te, mdl1.predict(X_te), "y1")
    report(y2_te, mdl2.predict(X_te), "y2")

if __name__ == "__main__":
    main()
