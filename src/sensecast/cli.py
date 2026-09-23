"""Command line entry point.

    sensecast all        --profile small      # generate -> ingest -> backtest -> faults -> simulate -> report
    sensecast generate | ingest | backtest | faults | simulate | report
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from sensecast.config import load_config
from sensecast.logging_utils import get_logger

log = get_logger("sensecast.cli")
STEPS = ["generate", "ingest", "backtest", "faults", "simulate", "monitor", "report"]


def run_step(step: str, cfg: dict):
    if step == "generate":
        from sensecast.generate.world import generate_world
        return generate_world(cfg)
    if step == "ingest":
        from sensecast.ingest.load import ingest
        return ingest(cfg)
    if step == "backtest":
        from sensecast.evaluation.backtest import run_backtest
        return run_backtest(cfg)
    if step == "faults":
        from sensecast.evaluation.faults import run_faults
        return run_faults(cfg)
    if step == "simulate":
        from sensecast.simulate.inventory import run_simulation
        return run_simulation(cfg)
    if step == "monitor":
        from sensecast.monitor.drift import run_monitoring
        return run_monitoring(cfg)
    if step == "report":
        from sensecast.evaluation.report import build_report
        return build_report(cfg)
    raise ValueError(step)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sensecast")
    ap.add_argument("step", choices=STEPS + ["all"])
    ap.add_argument("--profile", default=None, help="config profile name or path (e.g. small)")
    args = ap.parse_args(argv)
    cfg = load_config(args.profile)
    t0 = time.perf_counter()
    steps = STEPS if args.step == "all" else [args.step]
    for s in steps:
        if s == "report" and args.step == "all":
            # record batch runtime (everything before reporting) so the report can gate on it
            from sensecast.config import resolve_path
            minutes = (time.perf_counter() - t0) / 60
            (resolve_path(cfg, "artifacts") / "run_summary.json").write_text(
                json.dumps({"minutes": round(minutes, 2), "profile": cfg["profile"], "steps": steps[:-1]}, indent=2))
        run_step(s, cfg)
    minutes = (time.perf_counter() - t0) / 60
    log.info("pipeline_done", extra={"steps": steps, "minutes": round(minutes, 2), "profile": cfg["profile"]})
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
