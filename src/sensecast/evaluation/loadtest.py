"""Small concurrent load test against a running API; writes artifacts/loadtest.json.

    python -m sensecast.evaluation.loadtest --url http://localhost:8000 --requests 400 --concurrency 8
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

from sensecast.config import load_config, resolve_path


def run(url: str, n: int, concurrency: int) -> dict:
    with httpx.Client(base_url=url, timeout=10) as c:
        stores = [s["store_id"] for s in c.get("/v1/stores").json()]
        items = {s: [i["item_id"] for i in c.get("/v1/items", params={"store_id": s}).json()] for s in stores}
    rng = random.Random(0)
    paths = []
    for _ in range(n):
        s = rng.choice(stores)
        r = rng.random()
        if r < 0.7:
            paths.append(f"/v1/forecast/{s}/{rng.choice(items[s])}")
        elif r < 0.9:
            paths.append(f"/v1/orders?store_id={s}")
        else:
            paths.append(f"/v1/stores/{s}/forecast")

    def hit(path):
        with httpx.Client(base_url=url, timeout=10) as c:
            t0 = time.perf_counter()
            code = c.get(path).status_code
            return (time.perf_counter() - t0) * 1000, code

    # warm-up (lazy artifact loading happens on first touch)
    for p in paths[:10]:
        hit(p)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as ex:
        res = list(ex.map(hit, paths))
    wall = time.perf_counter() - t0
    ms = np.array([r[0] for r in res])
    errors = sum(1 for r in res if r[1] >= 400)
    return dict(requests=n, concurrency=concurrency, p50_ms=float(np.percentile(ms, 50)),
                p95_ms=float(np.percentile(ms, 95)), p99_ms=float(np.percentile(ms, 99)),
                max_ms=float(ms.max()), errors=errors, throughput_rps=n / wall,
                note="single uvicorn worker, parquet artifacts cached in memory")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--profile", default=None)
    a = ap.parse_args()
    out = run(a.url, a.requests, a.concurrency)
    cfg = load_config(a.profile)
    (resolve_path(cfg, "artifacts") / "loadtest.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
