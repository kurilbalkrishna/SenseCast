"""Backtest origin layout: train / calibration / test blocks with no target overlap.

An origin ``o`` is a day index; it means "end of day o" - everything up to and
including day o is known, and we forecast days o+1 .. o+H.

Blocks are separated so that no training target date falls inside the
calibration block and no calibration target date falls inside the test block.
"""
from __future__ import annotations

import math


def origin_layout(n_obs_days: int, cfg: dict) -> dict:
    bt = cfg["backtest"]
    H, step = bt["horizon"], bt["origin_step"]
    gap = step * math.ceil((H + 1) / step)  # smallest multiple of step > H

    last_test = n_obs_days - 1 - H
    test = [last_test - step * i for i in range(bt["n_test_origins"])][::-1]
    last_cal = test[0] - gap
    cal = [last_cal - step * i for i in range(bt["n_cal_origins"])][::-1]
    last_train = cal[0] - gap
    train = []
    o = last_train
    while o >= bt["min_history_days"] and len(train) < bt["max_train_origins"]:
        train.append(o)
        o -= step
    train = train[::-1]
    if not train:
        raise ValueError("Not enough history for a training block; lengthen the world or shrink blocks")
    return dict(train=train, cal=cal, test=test, production=n_obs_days - 1, horizon=H)


def window_days(origins: list[int], horizon: int) -> tuple[int, int]:
    """First and last target day covered by a block of origins."""
    return origins[0] + 1, origins[-1] + horizon
