"""Shard runner for ab_stability_decomp_v2.py (2026-09-25): runs the same run_cell for a given encoder
list and writes v2cells_p<pass>s<tag>.csv / v2parent_p<pass>s<tag>.csv, which report() merges through its
v2cells_*.csv / v2parent_*.csv globs (duplicates dropped). The analysis code is imported unchanged."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ab_stability_decomp_v2 as v2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_no", type=int, choices=[1, 2], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--encoders", nargs="+", required=True)
    a = ap.parse_args()
    v2.common_struct.set_task(v2.TASK)
    C, C0, C1, D = v2.base.load_tables()
    dest = v2.DIR / f"v2cells_p{a.pass_no}s{a.tag}.csv"
    pdest = v2.DIR / f"v2parent_p{a.pass_no}s{a.tag}.csv"
    rows = pd.read_csv(dest).to_dict("records") if dest.exists() else []
    prow = pd.read_csv(pdest).to_dict("records") if pdest.exists() else []
    done = {r["encoder"] for r in rows}
    for e in a.encoders:
        if e in done:
            continue
        r_, p_ = v2.run_cell(e, a.pass_no, C, C0, C1, D)
        rows += r_
        prow += p_
        pd.DataFrame(rows).to_csv(dest, index=False)
        pd.DataFrame(prow).to_csv(pdest, index=False)
        print("done", e, flush=True)
    print("written", dest)


if __name__ == "__main__":
    main()
