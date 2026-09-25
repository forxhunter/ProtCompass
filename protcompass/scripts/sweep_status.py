"""Coverage of the corrected erasure sweep, recomputed from the part files.

  python sweep_status.py                  the table
  python sweep_status.py --missing-tasks  the tasks with at least one missing encoder, for the driver
"""
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from erasure import EMB, TASK_KIND


def coverage():
    encs = sorted(d.name for d in (EMB / "stability").iterdir() if (d / "meta.json").exists())
    files = sorted(set(glob.glob("**/parts/erasure_v2*.csv", recursive=True)))
    # A worker that is still starting, or that died before its first flush, leaves a part file with no header.
    # Reading is best-effort for the same reason the sweep's own resume is: an unreadable part means those cells
    # are not yet done, never that the status is wrong.
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, usecols=["task", "encoder"]))
        except Exception:
            continue
    have = pd.concat(frames).drop_duplicates() if frames else pd.DataFrame(columns=["task", "encoder"])
    return encs, {t: sorted(set(have[have.task == t].encoder)) for t in TASK_KIND}


def main():
    encs, done = coverage()
    missing = {t: [e for e in encs if e not in v] for t, v in done.items()}
    if "--missing-tasks" in sys.argv:
        print(" ".join(t for t, m in missing.items() if m))
        return
    tot = sum(len(v) for v in done.values())
    for t in TASK_KIND:
        m = missing[t]
        print(f"{t:26s} {len(done[t]):2d}/{len(encs)}" + (f"  missing: {' '.join(m)}" if m else ""))
    print(f"TOTAL {tot}/{len(encs) * len(TASK_KIND)}")


if __name__ == "__main__":
    main()
