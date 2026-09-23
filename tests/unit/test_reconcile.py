import numpy as np
import pandas as pd

from sensecast.features.panel import summing_matrix
from sensecast.reconcile import mint


def _series():
    rows = []
    for s, city in (("S01", "Mumbai"), ("S02", "Pune")):
        for i, cat in enumerate(["beverages", "beverages", "snacks"]):
            rows.append(dict(store_id=s, item_id=f"I{i:03d}", category=cat, region=city, city=city,
                             city_idx=0 if city == "Mumbai" else 1, cat_idx=0 if cat == "beverages" else 1,
                             store_idx=0 if s == "S01" else 1))
    df = pd.DataFrame(rows)
    df["node_id"] = df.store_id + "|" + df.item_id
    return df


def test_summing_matrix_shape_and_totals():
    S, ids, levels = summing_matrix(_series())
    # total + 2 regions + 2 stores + 4 store-categories + 6 bottom
    assert S.shape == (15, 6)
    assert ids[0] == "total" and S[0].sum() == 6
    assert levels[-1] == "item_store"


def test_mint_output_is_coherent():
    S, _, _ = summing_matrix(_series())
    rng = np.random.default_rng(0)
    E = rng.normal(size=(60, S.shape[0]))
    W, lam = mint.shrunk_covariance(E)
    assert 0 <= lam <= 1
    G = mint.mint_projection(S, W)
    y_hat = rng.uniform(0, 10, size=(S.shape[0], 5))
    y_t = mint.reconcile(G, y_hat, S)
    assert mint.coherence_error(y_t, S) < 1e-9
    assert mint.coherence_error(y_hat, S) > 0.1


def test_projection_keeps_already_coherent_forecasts():
    S, _, _ = summing_matrix(_series())
    rng = np.random.default_rng(1)
    W, _ = mint.shrunk_covariance(rng.normal(size=(60, S.shape[0])))
    G = mint.mint_projection(S, W)
    coherent = S @ rng.uniform(1, 5, size=(S.shape[1], 3))
    assert np.allclose(G @ coherent, coherent)


def test_bottom_up_equals_summing_bottom():
    S, _, _ = summing_matrix(_series())
    y_hat = np.random.default_rng(2).uniform(0, 10, size=(S.shape[0], 2))
    y_bu = mint.reconcile(mint.bottom_up_projection(S), y_hat, S)
    assert np.allclose(y_bu, S @ y_hat[-S.shape[1]:])


def test_nonnegative_after_reconciliation():
    S, _, _ = summing_matrix(_series())
    y_hat = np.zeros((S.shape[0], 1))
    y_hat[0] = -50  # silly negative total
    W = np.eye(S.shape[0])
    y_t = mint.reconcile(mint.mint_projection(S, W), y_hat, S)
    assert (y_t >= 0).all()
