from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeightPolicy:
    """ необработанные весовые коэф  ребер в неотрицательные величины.
    Приложение использует расстояния в виде dist = 1 / weight и несколько процессов в стиле случайного блуждания,
    в которых предполагается, что весовые коэффициенты неотрицательны
    """

    mode: str = "drop_nonpositive"
    eps: float = 1e-9
    shift: float = 0.0

    def normalize_mode(self) -> str:
        return str(self.mode or "").strip().lower()


def apply_weight_policy_to_series(w: pd.Series, policy: WeightPolicy) -> Tuple[pd.Series, pd.Series]:

    eps = float(policy.eps)
    shift = float(policy.shift)
    mode = policy.normalize_mode()

    w_num = pd.to_numeric(w.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    finite = np.isfinite(w_num.to_numpy())

    if mode == "abs":
        w2 = w_num.abs()
        keep = finite & (w2.to_numpy() > 0)
        return w2, pd.Series(keep, index=w.index)

    if mode == "clip":
        w2 = w_num.copy()
        w2[~pd.Series(finite, index=w.index)] = np.nan
        w2 = w2.clip(lower=eps)
        keep = finite
        return w2, pd.Series(keep, index=w.index)

    if mode == "shift":
        w2 = w_num.copy()
        w2[~pd.Series(finite, index=w.index)] = np.nan
        w2 = w2 + shift
        w2 = w2.clip(lower=eps)
        keep = finite
        return w2, pd.Series(keep, index=w.index)

    # default: drop_nonpositive
    w2 = w_num
    keep = finite & (w2.to_numpy() > 0)
    return w2, pd.Series(keep, index=w.index)


def apply_weight_policy_scalar(w: float, policy: WeightPolicy) -> float | None:
    """Convert scalar weight. Returns None if the edge should be dropped."""
    try:
        x = float(w)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(x):
        return None

    mode = policy.normalize_mode()
    eps = float(policy.eps)
    shift = float(policy.shift)

    if mode == "abs":
        x = abs(x)
        return x if x > 0 else None
    if mode == "clip":
        return max(x, eps)
    if mode == "shift":
        x = x + shift
        return max(x, eps)

    # drop_nonpositive
    return x if x > 0 else None


def policy_from_settings(mode: str, eps: float, shift: float) -> WeightPolicy:
    """Create a WeightPolicy from settings values."""
    return WeightPolicy(mode=str(mode), eps=float(eps), shift=float(shift))
