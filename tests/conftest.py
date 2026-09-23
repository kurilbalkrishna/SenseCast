"""Shared fixtures. The small profile runs the whole pipeline in ~10 seconds."""
from __future__ import annotations

import os

import pytest

from sensecast.config import load_config


@pytest.fixture(scope="session")
def small_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("sensecast")
    os.environ["SENSECAST_DATA_DIR"] = str(root / "data")
    os.environ["SENSECAST_ARTIFACTS"] = str(root / "artifacts")
    os.environ["SENSECAST_REPORTS"] = str(root / "reports")
    return root


@pytest.fixture(scope="session")
def small_cfg(small_root):
    return load_config("small")


@pytest.fixture(scope="session")
def ingested(small_cfg):
    """World generated and ingested (no models)."""
    from sensecast.generate.world import generate_world
    from sensecast.ingest.load import ingest

    generate_world(small_cfg)
    ingest(small_cfg)
    return small_cfg


@pytest.fixture(scope="session")
def pipeline(ingested):
    """Full small pipeline: backtest, faults, simulation, monitoring, report."""
    from sensecast.cli import run_step

    for step in ("backtest", "faults", "simulate", "monitor", "report"):
        run_step(step, ingested)
    return ingested


@pytest.fixture(scope="session")
def ctx(ingested):
    from sensecast.evaluation.backtest import Context

    return Context(ingested)
