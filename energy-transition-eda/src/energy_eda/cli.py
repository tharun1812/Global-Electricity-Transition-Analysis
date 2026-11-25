"""Command line: `python -m energy_eda download | analyze | notebook`."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import data, report


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="energy_eda", description="Energy transition EDA on the OWID energy dataset.")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download", help="download the OWID energy dataset into data/")
    d.add_argument("--path", default=str(data.DEFAULT_PATH))
    d.add_argument("--force", action="store_true")
    a = sub.add_parser("analyze", help="run the analysis; write figures, facts.json and insights.md")
    a.add_argument("--data", default=str(data.DEFAULT_PATH))
    a.add_argument("--out", default="reports")
    a.add_argument("--start", type=int, default=2014)
    a.add_argument("--end", type=int, default=2024)
    a.add_argument("--gdp-year", type=int, default=2022)
    a.add_argument("--min-twh", type=float, default=5.0)
    n = sub.add_parser("notebook", help="rebuild and execute notebooks/energy_transition_eda.ipynb")
    n.add_argument("--data", default=str(data.DEFAULT_PATH))
    args = p.parse_args(argv)

    if args.command == "download":
        print(f"data ready: {data.download(args.path, force=args.force)}")
    elif args.command == "analyze":
        path = Path(args.data)
        if not path.exists():
            data.download(path)
        facts = report.run_all(data.load(path), args.out, y0=args.start, y1=args.end, gdp_year=args.gdp_year, min_twh=args.min_twh)
        print(f"wrote {args.out}/insights.md, facts.json and 12 figures "
              f"({facts['data']['countries_compared']} countries compared, {facts['clusters']['k']} archetypes)")
    else:
        from .notebook import build_and_run
        print(f"wrote {build_and_run(args.data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
