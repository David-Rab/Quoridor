from __future__ import annotations
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_absolute_error, r2_score

def report(y_true, y_pred, label="y"):
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"{label}:  MAE={mae:.3f}   R²={r2:.3f}")

    fig = px.scatter(
        x=y_true, y=y_pred,
        labels=dict(x="True", y="Predicted"),
        title=f"{label} — True vs Predicted",
    )
    fig.add_shape(type="line", x0=-15, y0=-15, x1=15, y1=15, line_dash="dot")
    fig.show()
