"""LightGBM global models (one model across all series and horizons)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from sensecast.features.build import CATEGORICAL, FAMILY


def _params(cfg: dict, objective: str, alpha: float | None = None) -> dict:
    p = cfg["lgbm"]
    params = dict(
        objective=objective, learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
        min_data_in_leaf=p["min_data_in_leaf"], feature_fraction=p["feature_fraction"],
        bagging_fraction=p["bagging_fraction"], bagging_freq=p["bagging_freq"], lambda_l2=p["lambda_l2"],
        num_threads=p["n_jobs"] if p["n_jobs"] > 0 else 0, seed=cfg["seed"], deterministic=True,
        force_row_wise=True, verbose=-1, max_bin=p.get("max_bin", 255),
    )
    if objective == "tweedie":
        params["tweedie_variance_power"] = p["tweedie_variance_power"]
    if objective == "quantile":
        params["alpha"] = alpha
        params["bagging_fraction"] = p.get("quantile_bagging_fraction", p["bagging_fraction"])
    return params


def train(df: pd.DataFrame, features: list[str], cfg: dict, objective: str = "tweedie",
          alpha: float | None = None, target: str = "y", n_estimators: int | None = None) -> lgb.Booster:
    cats = [c for c in CATEGORICAL if c in features]
    ds = lgb.Dataset(df[features], label=df[target], categorical_feature=cats, free_raw_data=True)
    return lgb.train(_params(cfg, objective, alpha), ds, num_boost_round=n_estimators or cfg["lgbm"]["n_estimators"])


def predict(model: lgb.Booster, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict(df[features], num_threads=0).astype(np.float32)


def driver_families(model: lgb.Booster, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Per-row contribution of each feature family, in log space (tweedie link).

    exp(contribution) - 1 reads as "this family moved the forecast by x%".
    """
    contrib = model.predict(df[features], pred_contrib=True, num_threads=0)
    fam = pd.DataFrame(contrib[:, :-1], columns=features).T.groupby(lambda f: FAMILY.get(f, "other")).sum().T
    fam["baseline"] = contrib[:, -1]
    return fam.astype(np.float32)


def importance(model: lgb.Booster, features: list[str]) -> pd.DataFrame:
    gain = model.feature_importance("gain")
    imp = pd.DataFrame(dict(feature=features, gain=gain / gain.sum()))
    imp["family"] = imp.feature.map(lambda f: FAMILY.get(f, "other"))
    return imp.sort_values("gain", ascending=False).reset_index(drop=True)
