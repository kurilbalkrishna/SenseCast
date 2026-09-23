"""Conformalized quantile regression (CQR, Romano et al. 2019).

Raw P10/P90 from quantile models are often mis-calibrated. On a calibration block
the model never trained on, compute conformity scores
    E = max(P10 - y, y - P90)
and widen (or tighten, if E is negative) every interval by the
ceil((n+1)(1-alpha))/n empirical quantile of E, separately per group
(horizon bucket x demand class). Coverage is then guaranteed on average under
exchangeability and is checked empirically on the test block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def horizon_bucket(h: np.ndarray) -> np.ndarray:
    return np.where(h <= 3, "h1-3", np.where(h <= 7, "h4-7", "h8-14"))


class CQR:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.q: dict[tuple, float] = {}
        self.q_global = 0.0

    def _q(self, scores: np.ndarray) -> float:
        n = len(scores)
        if n == 0:
            return 0.0
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        return float(np.quantile(scores, level, method="higher"))

    def fit(self, y, lo, hi, groups: pd.DataFrame) -> CQR:
        scores = np.maximum(np.asarray(lo) - np.asarray(y), np.asarray(y) - np.asarray(hi))
        self.q_global = self._q(scores)
        keys = list(groups.itertuples(index=False, name=None))
        df = pd.DataFrame(dict(k=keys, s=scores))
        for k, grp in df.groupby("k"):
            self.q[k] = self._q(grp.s.values) if len(grp) >= 50 else self.q_global
        return self

    def apply(self, lo, hi, groups: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        keys = list(groups.itertuples(index=False, name=None))
        adj = np.array([self.q.get(k, self.q_global) for k in keys], dtype=np.float32)
        return np.maximum(np.asarray(lo) - adj, 0), np.maximum(np.asarray(hi) + adj, 0)

    def to_dict(self) -> dict:
        return dict(alpha=self.alpha, global_adjustment=self.q_global,
                    groups={"|".join(map(str, k)): v for k, v in self.q.items()})
