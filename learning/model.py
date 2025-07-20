"""
build_model(preproc, kind=…)

* kind="sgd"  – sparse-aware linear regressor (default POC)
* kind="ordinal" – mord.LogisticIT (for later)
* kind="lgbm" – LightGBMRegressor (when data ≥10 k)

Change the body once and the rest of the app stays untouched.
"""
from __future__ import annotations
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
# from mord import LogisticIT
# import lightgbm as lgb

def build_model(preproc, kind="sgd", **kwargs):
    if kind == "sgd":
        base = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=1e-4,
            learning_rate="optimal",
            max_iter=1000,
            tol=1e-3,
            **kwargs,
        )
    # elif kind == "ordinal":
    #     base = LogisticIT(alpha=kwargs.get("alpha", 3.0), n_classes=31)
    # elif kind == "lgbm":
    #     base = lgb.LGBMRegressor(objective="regression", **kwargs)
    else:
        raise ValueError(f"Unknown kind={kind}")
    return make_pipeline(preproc, base)
