"""Intermittent-demand methods and the Syntetos-Boylan demand classification.

ADI  = average inter-demand interval (days between non-zero sales)
CV2  = squared coefficient of variation of the non-zero sizes

            CV2 < 0.49      CV2 >= 0.49
ADI < 1.32  smooth          erratic
ADI >= 1.32 intermittent    lumpy

Croston: smooth size z and interval p separately, forecast z/p.
SBA:     Croston x (1 - alpha/2), removes Croston's known upward bias.
TSB:     smooth size z and demand probability pi every day, forecast pi * z;
         pi decays on zero days, so dying items fade to zero.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

CLASSES = ["smooth", "erratic", "intermittent", "lumpy"]


def classify(Ym: np.ndarray, upto: int, adi_cut: float = 1.32, cv2_cut: float = 0.49) -> pd.DataFrame:
    hist = Ym[: upto + 1]
    n_obs = (~np.isnan(hist)).sum(0)
    nz = np.nan_to_num(hist) > 0
    n_nz = nz.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        adi = np.where(n_nz > 0, n_obs / n_nz, np.inf)
        sizes = np.where(nz, hist, np.nan)
        mu = np.nanmean(sizes, axis=0)
        sd = np.nanstd(sizes, axis=0)
        cv2 = np.where(n_nz > 1, (sd / mu) ** 2, 0.0)
    cls = np.where(adi < adi_cut, np.where(cv2 < cv2_cut, "smooth", "erratic"),
                   np.where(cv2 < cv2_cut, "intermittent", "lumpy"))
    cls = np.where(n_obs < 28, "new", cls)
    return pd.DataFrame(dict(adi=adi, cv2=cv2, demand_class=cls))


def croston_family(Ym: np.ndarray, alpha: float = 0.1, beta: float = 0.1) -> dict[str, np.ndarray]:
    """State after observing each day t, for all series. NaN days (stockout /
    not launched) do not update the state."""
    T, N = Ym.shape
    out = {k: np.full((T, N), np.nan, dtype=np.float32) for k in ("croston", "sba", "tsb")}
    z = np.full(N, np.nan)      # size
    p = np.full(N, np.nan)      # interval (Croston)
    q = np.ones(N)              # periods since last demand
    z_t = np.full(N, np.nan)    # size (TSB)
    pi = np.full(N, np.nan)     # probability (TSB)
    for t in range(T):
        y = Ym[t]
        obs = ~np.isnan(y)
        pos = obs & (y > 0)
        # initialise on first observation
        init = obs & np.isnan(pi)
        pi = np.where(init, pos.astype(float) * 0.5 + 0.25, pi)
        z_t = np.where(init & pos, y, z_t)
        # Croston / SBA
        first_pos = pos & np.isnan(z)
        z = np.where(first_pos, y, z)
        p = np.where(first_pos, q, p)
        upd = pos & ~first_pos
        z = np.where(upd, z + alpha * (y - z), z)
        p = np.where(upd, p + alpha * (q - p), p)
        q = np.where(pos, 1, np.where(obs, q + 1, q))
        # TSB
        pi = np.where(obs & ~init, pi + beta * (pos.astype(float) - pi), pi)
        z_t = np.where(pos & ~np.isnan(z_t) & ~init, z_t + alpha * (y - z_t), np.where(pos & np.isnan(z_t), y, z_t))
        with np.errstate(invalid="ignore", divide="ignore"):
            cro = z / p
        out["croston"][t] = cro
        out["sba"][t] = cro * (1 - alpha / 2)
        out["tsb"][t] = pi * z_t
    for k in out:
        out[k] = np.nan_to_num(out[k], nan=0.0)
    return out
