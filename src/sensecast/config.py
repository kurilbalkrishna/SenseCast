"""Configuration loading.

A run is fully described by (code commit, config file, seed). Profiles such as
``configs/small.yaml`` are deep-merged over ``configs/default.yaml``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(profile: str | os.PathLike | None = None, **overrides: Any) -> dict:
    """Load default config, optionally merged with a profile file and overrides.

    ``profile`` may be a path or a bare name such as ``"small"``.
    The environment variable ``SENSECAST_CONFIG`` is used when ``profile`` is None.
    """
    with open(DEFAULT_CONFIG, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    profile = profile or os.getenv("SENSECAST_CONFIG")
    if profile:
        path = Path(profile)
        if not path.suffix:
            path = REPO_ROOT / "configs" / f"{profile}.yaml"
        with open(path, encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
        cfg["profile"] = path.stem
    else:
        cfg["profile"] = "default"

    # environment overrides for paths (docker, CI, tests)
    for env, key in (("SENSECAST_DATA_DIR", "data"), ("SENSECAST_ARTIFACTS", "artifacts"), ("SENSECAST_REPORTS", "reports")):
        if os.getenv(env):
            cfg["paths"][key] = os.environ[env]

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    # validate category mix
    mix = cfg["world"]["category_mix"]
    if sum(mix.values()) != cfg["world"]["n_items"]:
        raise ValueError(
            f"category_mix sums to {sum(mix.values())}, n_items is {cfg['world']['n_items']}"
        )
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """Return an absolute path for ``paths.<key>`` (relative paths are repo-relative)."""
    p = Path(cfg["paths"][key])
    if not p.is_absolute():
        p = REPO_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
