"""Shuffled-3Di control, downstream contrasts (PRESPEC_shuf.md step 2).

Per task: S(saprot), S(saprot_shuf3di), S(saprot_seqonly); A = S(saprot) - S(saprot_shuf3di) and
B = S(saprot_shuf3di) - S(saprot_seqonly), each with the paired protein bootstrap of ab_bootstrap.py (same index matrix,
same boot_metric; ab_bootstrap.py is imported, not edited). Selection per source from its own score rows (lowest mean
validation loss over three seeds): score_parts/ab_long_<task>__{saprot,saprot_seqonly}.csv (read only) and
score_parts_shuf/ab_shuf_long_<task>__saprot_shuf3di.csv. Tasks whose inputs are not all present are listed as pending.

Writes ab_modality/downstream_deltas_shuf.csv and downstream_group_summary_shuf.csv.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AB = Path((PROTCOMPASS_ROOT + "/protcompass/extract/ab_modality"))
sys.path.insert(0, str(AB))
from ab_bootstrap import DRAWS, GROUP, ORDER, TASKS, boot_metric, evaluate   # noqa: E402

FULL, SHUF, SEQ = "saprot", "saprot_shuf3di", "saprot_seqonly"
SOURCES = (FULL, SHUF, SEQ)


def parts_file(task, s):
    return (AB / "score_parts_shuf" / f"ab_shuf_long_{task}__{s}.csv" if s == SHUF
            else AB / "score_parts" / f"ab_long_{task}__{s}.csv")


def state(lo, hi, pos, neg):
    return pos if lo > 0 else neg if hi < 0 else "unresolved"


def main():
    rows, draws, pending = [], {}, []
    for task in ORDER:
        need = [parts_file(task, s) for s in SOURCES] + [AB / "predictions" / f"{task}__{s}.npz" for s in SOURCES]
        miss = [f.name for f in need if not f.exists()]
        if miss:
            pending.append((task, miss))
            continue
        kind = TASKS[task][0]
        P = {s: dict(np.load(AB / "predictions" / f"{task}__{s}.npz", allow_pickle=False)) for s in SOURCES}
        for s in SOURCES:
            assert np.array_equal(P[s]["test_ids"], P[FULL]["test_ids"]), (task, s)
            assert np.array_equal(P[s]["y_test"], P[FULL]["y_test"]), (task, s)
        y = P[FULL]["y_test"]
        n = len(y)
        rng = np.random.default_rng(sum(map(ord, task)))                 # the index matrix of ab_bootstrap.py
        I = rng.integers(0, n, size=(DRAWS, n))
        C = np.stack([np.bincount(r, minlength=n) for r in I]).astype(float)
        S, B, probe, sd = {}, {}, {}, {}
        for s in SOURCES:
            d = pd.read_csv(parts_file(task, s))
            pv = d.groupby("probe").val_loss.mean()
            probe[s] = pv.idxmin()
            pts, bs = [], []
            for seed in (0, 1, 2):
                out = P[s][f"{probe[s]}_{seed}"]
                pts.append(evaluate(kind, out, y))
                bs.append(boot_metric(kind, y, out, I, C))
            S[s], B[s], sd[s] = float(np.mean(pts)), np.mean(bs, 0), float(np.std(pts, ddof=1))
            lg = d[d.probe == probe[s]].score.mean()
            assert abs(lg - S[s]) < 1e-6, (task, s, lg, S[s])
        dA, dB = B[FULL] - B[SHUF], B[SHUF] - B[SEQ]
        draws[task] = (dA, dB, B[FULL] - B[SEQ])
        r = dict(task=task, group=GROUP[task], metric=TASKS[task][1], n_test=n,
                 S_saprot=S[FULL], S_shuf3di=S[SHUF], S_seqonly=S[SEQ],
                 seed_sd_saprot=sd[FULL], seed_sd_shuf3di=sd[SHUF], seed_sd_seqonly=sd[SEQ],
                 probe_saprot=probe[FULL], probe_shuf3di=probe[SHUF], probe_seqonly=probe[SEQ])
        for name, dd, pos, neg, point in (("A", dA, "alignment adds", "alignment costs", S[FULL] - S[SHUF]),
                                          ("B", dB, "shuffled above #", "shuffled below #", S[SHUF] - S[SEQ])):
            lo, hi = np.percentile(dd, [2.5, 97.5])
            lo3, hi3 = np.percentile(dd, [0.125, 99.875])
            r.update({f"delta_{name}": point, f"lo_{name}": lo, f"hi_{name}": hi, f"lo9975_{name}": lo3,
                      f"hi9975_{name}": hi3, f"state_{name}": state(lo, hi, pos, neg),
                      f"state9975_{name}": state(lo3, hi3, pos, neg)})
        lo, hi = np.percentile(B[FULL] - B[SEQ], [2.5, 97.5])
        r.update(delta_full_minus_seqonly=S[FULL] - S[SEQ], lo_full_minus_seqonly=lo, hi_full_minus_seqonly=hi)
        rows.append(r)
        print(f"{task:26s} saprot {S[FULL]:.4f} shuf {S[SHUF]:.4f} seq {S[SEQ]:.4f} | "
              f"A {r['delta_A']:+.4f} [{r['lo_A']:+.4f}, {r['hi_A']:+.4f}] {r['state_A']} | "
              f"B {r['delta_B']:+.4f} [{r['lo_B']:+.4f}, {r['hi_B']:+.4f}] {r['state_B']}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(AB / "downstream_deltas_shuf.csv", index=False)
    G = []
    for gname, tasks in (("experimental", [t for t in ORDER if GROUP[t] == "experimental"]),
                         ("predicted", [t for t in ORDER if GROUP[t] != "experimental"])):
        have = [t for t in tasks if t in draws]
        if not have:
            continue
        sub = D[D.task.isin(have)]
        for k, name in ((0, "A"), (1, "B")):
            m = np.mean([draws[t][k] for t in have], 0)
            lo, hi = np.percentile(m, [2.5, 97.5])
            st = sub[f"state_{name}"]
            G.append(dict(contrast=name, group=gname, n_tasks=len(have), n_tasks_group=len(tasks),
                          mean_delta=sub[f"delta_{name}"].mean(), lo=lo, hi=hi,
                          above=int(st.str.contains("adds|above").sum()), below=int(st.str.contains("costs|below").sum()),
                          unresolved=int((st == "unresolved").sum())))
    G = pd.DataFrame(G)
    G.to_csv(AB / "downstream_group_summary_shuf.csv", index=False)
    pd.set_option("display.width", 250)
    print(G.round(4).to_string(index=False))
    for t, m in pending:
        print(f"pending {t}: missing {m}")


if __name__ == "__main__":
    main()
