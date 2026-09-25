"""Shuffled-3Di control, probing contrasts (PRESPEC_shuf.md step 3), with the machinery of scripts/symmetric_decide.py.

Per probe: A = S(saprot) - S(saprot_shuf3di), B = S(saprot_shuf3di) - S(saprot_seqonly). Readout family of each source
chosen on validation (symmetric_decide.main.choose); intervals from the protein-level paired bootstrap of
symmetric_decide.Boot (2,000 draws, one fixed draw per probe), as ab_probe_decide.py (imported, not edited).
saprot is read from the released symmetric/results_superfamily/by_source/ (read only); the other two from
ab_modality/symmetric/results_superfamily/by_source/. Writes ab_modality/probing_deltas_shuf.csv.
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
import ab_probe_decide as apd                                            # noqa: E402

sd = apd.sd
PAIRS = [("saprot", "saprot_shuf3di", "A", "alignment adds", "alignment costs"),
         ("saprot_shuf3di", "saprot_seqonly", "B", "shuffled above #", "shuffled below #")]


def main():
    for s in ("saprot_shuf3di", "saprot_seqonly"):
        assert (apd.NEW / f"{s}.csv").exists(), s
    S = apd.load_scores(["saprot", "saprot_shuf3di", "saprot_seqonly"])
    val = S.set_index(["source", "probe", "family"])["val"]
    test = S.set_index(["source", "probe", "family"])["test"]
    B = apd.Boot2(apd.DRAWS)

    def choose(src, probe):
        best = None
        for fam in ("linear", "nonlinear"):
            if (src, probe, fam) in val.index and np.isfinite(val[(src, probe, fam)]):
                if best is None or val[(src, probe, fam)] > best[1] + 1e-12:
                    best = (fam, val[(src, probe, fam)])
        return best[0]

    rows = []
    for p in apd.PROBES:
        fam = {s: choose(s, p) for s in ("saprot", "saprot_shuf3di", "saprot_seqonly")}
        r = dict(probe=p, determined=p in sd.DETERMINED,
                 S_saprot=test[("saprot", p, fam["saprot"])], S_shuf3di=test[("saprot_shuf3di", p, fam["saprot_shuf3di"])],
                 S_seqonly=test[("saprot_seqonly", p, fam["saprot_seqonly"])],
                 fam_saprot=fam["saprot"], fam_shuf3di=fam["saprot_shuf3di"], fam_seqonly=fam["saprot_seqonly"])
        for a, b, name, pos, neg in PAIRS:
            g = B.gap(p, a, fam[a], b, fam[b])
            lo, hi = np.percentile(g, [2.5, 97.5])
            lo3, hi3 = np.percentile(g, [0.05, 99.95])
            r.update({f"delta_{name}": test[(a, p, fam[a])] - test[(b, p, fam[b])], f"lo_{name}": lo, f"hi_{name}": hi,
                      f"lo999_{name}": lo3, f"hi999_{name}": hi3, f"state_{name}": apd.state(lo, hi, pos, neg),
                      f"state999_{name}": apd.state(lo3, hi3, pos, neg)})
        rows.append(r)
        print(f"{p:30s} saprot {r['S_saprot']:.3f} shuf {r['S_shuf3di']:.3f} seq {r['S_seqonly']:.3f} | "
              f"A {r['delta_A']:+.3f} [{r['lo_A']:+.3f}, {r['hi_A']:+.3f}] | B {r['delta_B']:+.3f} "
              f"[{r['lo_B']:+.3f}, {r['hi_B']:+.3f}]", flush=True)
    pd.DataFrame(rows).to_csv(AB / "probing_deltas_shuf.csv", index=False)


if __name__ == "__main__":
    main()
