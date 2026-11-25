"""Command line entry point: ``uv run gulfheat <step>``.

Steps, in order:
  era5      fetch ERA5 at the grid points around stations and the map area
  table     download station readings, build static features, join everything
  check     run data checks and print them
  evaluate  tune, train, test and write reports/
  map       make the street-level map
  report    let the LLM agent audit the results and write reports/REPORT.md
  all       era5, table, check, evaluate, map
"""

from __future__ import annotations

import argparse
import json

from .config import PATHS, load_config


def step_era5(cfg, cached_only=False):
    """Fetch ERA5 only around stations with enough readings, and only for those years.

    Stations with no training year are skipped: they could never be scored on
    in the "seen stations" setting. The free API allows about 380 one-year
    grid point downloads a day, so the map area and the stations nearest the
    Gulf coast are fetched first.
    """
    import numpy as np
    from . import era5, stations
    meta, _ = stations.load_observations(cfg)
    kept = meta["years_kept"].fillna("").map(lambda s: [int(y) for y in s.split(",") if y])
    st = meta[kept.map(lambda ys: bool(set(ys) & set(cfg["train_years"])))].copy()
    st["years"] = kept
    box, step = cfg["gulf_box"], cfg["era5"]["grid_step"]
    clat, clon = (box["lat_min"] + box["lat_max"]) / 2, (box["lon_min"] + box["lon_max"]) / 2
    st["dist"] = np.hypot(st["lat"] - clat, (st["lon"] - clon) * np.cos(np.deg2rad(clat)))
    m = cfg["map"]
    map_years = {int(t[:4]) for t in m["times_utc"]}
    needs = {}
    for lat in (m["lat_min"], m["lat_max"]):
        for lon in (m["lon_min"], m["lon_max"]):
            for n in era5.corner_nodes(lat, lon, step):
                needs.setdefault(n, set()).update(map_years)
    for row in st.sort_values("dist").itertuples():
        for n in era5.corner_nodes(row.lat, row.lon, step):
            needs.setdefault(n, set()).update(row.years)
    todo = sum(len(era5.missing_years(*n, ys)) for n, ys in needs.items())
    print(f"ERA5: {len(meta)} candidate stations, {len(st)} with enough readings in a training year, "
          f"{len(needs)} grid points, {todo} grid point years to download", flush=True)
    era5.build_cube(needs, cfg, fetch=not cached_only)


def step_table(cfg):
    from .dataset import build_table
    build_table(cfg)


def step_check(cfg):
    from .checks import run_checks
    res = run_checks()
    res.pop("per_station")
    print(json.dumps(res, indent=2))


def step_evaluate(cfg):
    from . import evaluate
    from .dataset import load_table
    res = evaluate.run(load_table(), cfg)
    print(json.dumps(res["summary"], indent=2))


def step_map(cfg):
    from .heatmap import make_map
    make_map(cfg)
    print(f"map written to {PATHS.figures / 'street_level_map.png'}")


def step_report(cfg, args):
    from .agent import run_agent
    run_agent(model=args.model, max_turns=args.max_turns)


def main():
    p = argparse.ArgumentParser(prog="gulfheat", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["era5", "table", "check", "evaluate", "map", "report", "all"])
    p.add_argument("--config", default=None)
    p.add_argument("--model", default="claude-opus-5-5", help="model for the report agent")
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--cached-only", action="store_true",
                   help="era5 step: build the cube from grid points already downloaded")
    args = p.parse_args()
    cfg = load_config(args.config)
    PATHS.ensure()
    steps = {"era5": step_era5, "table": step_table, "check": step_check,
             "evaluate": step_evaluate, "map": step_map}
    if args.step == "all":
        for name in ["era5", "table", "check", "evaluate", "map"]:
            print(f"== {name}")
            steps[name](cfg)
    elif args.step == "report":
        step_report(cfg, args)
    elif args.step == "era5":
        step_era5(cfg, args.cached_only)
    else:
        steps[args.step](cfg)


if __name__ == "__main__":
    main()
