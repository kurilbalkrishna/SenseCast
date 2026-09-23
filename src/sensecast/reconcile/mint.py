"""Hierarchical reconciliation.

Base forecasts are produced independently at every level (total, region, store,
store x category, item x store), so they do not add up. Reconciliation maps them
to a coherent set  y~ = S P y^.

* Bottom-up:   P = [0 | I]  (ignore every aggregate forecast)
* MinT-shrink: P = (S' W^-1 S)^-1 S' W^-1, with W the shrunk covariance of base
  forecast errors (Wickramasuriya, Athanasopoulos & Hyndman 2019; shrinkage of
  Schafer & Strimmer 2005). It weights each level by how reliable it was.
"""
from __future__ import annotations

import numpy as np


def shrunk_covariance(E: np.ndarray, sd_floor: float | None = None) -> tuple[np.ndarray, float]:
    """Schafer-Strimmer shrinkage of the correlation matrix towards identity.

    E: [n_obs, m] residual matrix. Returns (W, lambda)."""
    n, m = E.shape
    X = E - E.mean(0)
    sd = X.std(0, ddof=1)
    floor = sd_floor if sd_floor is not None else max(1e-3, 0.05 * float(np.median(sd[sd > 0])) if (sd > 0).any() else 1e-3)
    sd = np.maximum(sd, floor)
    Z = X / sd
    R = (Z.T @ Z) / (n - 1)
    Z2 = Z**2
    # var(r_ij) = n / (n-1)^3 * sum_k (z_ki z_kj - mean)^2
    s2 = Z2.T @ Z2
    w_mean = (Z.T @ Z) / n
    var_r = n / (n - 1) ** 3 * (s2 - n * w_mean**2)
    off = ~np.eye(m, dtype=bool)
    lam = float(np.clip(var_r[off].sum() / max((R[off] ** 2).sum(), 1e-12), 0.0, 1.0))
    R_s = lam * np.eye(m) + (1 - lam) * R
    np.fill_diagonal(R_s, 1.0)
    W = R_s * np.outer(sd, sd)
    return W, lam


def mint_projection(S: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Return G = S P (n_total x n_total) so that y_tilde = G @ y_hat."""
    W_inv_S = np.linalg.solve(W, S)                # W^-1 S
    A = S.T @ W_inv_S                              # S' W^-1 S
    P = np.linalg.solve(A, W_inv_S.T)              # (S' W^-1 S)^-1 S' W^-1
    return S @ P


def bottom_up_projection(S: np.ndarray) -> np.ndarray:
    n_total, n_b = S.shape
    P = np.zeros((n_b, n_total))
    P[:, n_total - n_b:] = np.eye(n_b)
    return S @ P


def reconcile(G: np.ndarray, Y_hat: np.ndarray, S: np.ndarray, nonneg: bool = True) -> np.ndarray:
    """Y_hat: [n_total, n_cases]. Returns coherent forecasts (bottom clipped at 0 and
    re-aggregated so coherence survives the clipping)."""
    Y_t = G @ Y_hat
    if nonneg:
        n_b = S.shape[1]
        bottom = np.maximum(Y_t[-n_b:], 0)
        Y_t = S @ bottom
    return Y_t


def coherence_error(Y: np.ndarray, S: np.ndarray) -> float:
    n_b = S.shape[1]
    return float(np.max(np.abs(S @ Y[-n_b:] - Y)))
