"""Decide the frozen downstream hypotheses on the held-out cohorts, and say which ones this pass can decide.

scripts/downstream_holdout_freeze.py fixed D1 to D4 before any held-out row was scored. This pass erases one
property, amino-acid composition, over five tasks whose head the pass supports. Two consequences are stated rather
than worked around:

  D1 and D2 are about composition on membrane and soluble and on subcellular localisation. Both are decidable here.
  D3 covers GO-BP, which is multilabel and not scored in this pass, so only its stability half is decidable.
  D4 asks that no property reach the magnitude floor on remote homology. Only composition was erased, so what is
  decidable is the composition half of it, and the verdict says so.

A hypothesis that cannot be decided is reported undecided. It is not reinterpreted into one that can.

  python holdout_report.py [--markdown | --latex]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t as student_t

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
CD = ROOT / "confirmation_downstream"
DELTA = 0.02
LABEL = {"membrane_soluble": "membrane vs soluble", "subcellular_localization": "subcellular localisation",
         "fluorescence": "fluorescence", "stability": "stability", "remote_homology": "remote homology"}


def contrasts(d):
    """Per task and encoder: the conditional effect (sham minus erased) and the repair contrast R per descriptor."""
    rows = []
    for (task, enc), g in d.groupby(["task", "encoder"]):
        piv = g.pivot_table(index="arm", columns="descriptor", values="score")
        if "erased" not in piv.index or "none" not in piv.columns:
            continue
        shams = [a for a in piv.index if a.startswith("sham")]
        if not shams:
            continue
        rec = dict(task=task, encoder=enc, n_sham=len(shams),
                   intact=piv.loc["intact", "none"] if "intact" in piv.index else np.nan,
                   erased=piv.loc["erased", "none"],
                   sham=float(np.mean([piv.loc[a, "none"] for a in shams])))
        rec["effect"] = rec["sham"] - rec["erased"]
        for q in [c for c in piv.columns if c != "none"]:
            real = piv.loc["erased", q] - piv.loc["erased", "none"]
            sham = float(np.mean([piv.loc[a, q] - piv.loc[a, "none"] for a in shams]))
            rec[f"R_{q}"] = real - sham
        rows.append(rec)
    return pd.DataFrame(rows)


def sign_p_one_sided(v):
    v = np.asarray([x for x in v if np.isfinite(x) and x != 0], dtype=float)
    if len(v) < 3:
        return np.nan, len(v), np.nan
    k = int((v > 0).sum())
    return float(binomtest(k, len(v), 0.5, alternative="greater").pvalue), len(v), k


def ucl(v, alpha=0.05):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if len(v) < 3:
        return np.nan
    return float(v.mean() + student_t.ppf(1 - alpha, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    freeze = json.loads((CD / "freeze.json").read_text())
    d = pd.read_csv(CD / "addback_holdout.csv")
    if d.empty:
        raise SystemExit("addback_holdout.csv is empty")
    C = contrasts(d)
    C.to_csv(CD / "holdout_contrasts.csv", index=False)
    tasks = sorted(C.task.unique())

    detail, verdict = {}, {}

    # D1: the conditional effect of erasing composition, on the two tasks that name it
    parts = []
    ok = True
    for t in ("membrane_soluble", "subcellular_localization"):
        g = C[C.task == t]
        if g.empty:
            parts.append(f"{LABEL[t]}: not scored"); ok = False; continue
        p, n, k = sign_p_one_sided(g.effect)
        parts.append(f"{LABEL[t]} {g.effect.mean():+.3f} over {n} encoders, {k} positive, one-sided p={p:.3g}")
        ok = ok and g.effect.mean() > 0 and np.isfinite(p) and p < 0.05
    detail["D1"], verdict["D1"] = "; ".join(parts), ok

    # D2: selective repair on the same two tasks
    parts, ok = [], True
    for t in ("membrane_soluble", "subcellular_localization"):
        g = C[C.task == t]
        if g.empty or "R_named" not in g:
            parts.append(f"{LABEL[t]}: not scored"); ok = False; continue
        named = g.R_named
        ctrls = {q: g[f"R_{q}"].mean() for q in ("permuted", "other") if f"R_{q}" in g}
        good = bool((named > 0).all()) and all(named.mean() > v for v in ctrls.values())
        parts.append(f"{LABEL[t]} R={named.mean():+.4f} ({int((named > 0).sum())}/{len(named)} positive) against "
                     + ", ".join(f"{q} {v:+.4f}" for q, v in ctrls.items()))
        ok = ok and good
    detail["D2"], verdict["D2"] = "; ".join(parts), ok

    # D3: where the released run named nothing, nothing is named here. GO-BP is not scored in this pass.
    g = C[C.task == "stability"]
    if g.empty:
        detail["D3"], verdict["D3"] = "stability not scored; GO-BP is multilabel and outside this pass", None
    else:
        p, n, k = sign_p_one_sided(g.effect)
        named = bool(np.isfinite(p) and p < 0.05 and g.effect.mean() >= DELTA)
        detail["D3"] = (f"stability {g.effect.mean():+.3f} over {n} encoders, {k} positive, one-sided p={p:.3g}: "
                        f"{'names composition' if named else 'names nothing'}. "
                        f"GO-BP is multilabel and not scored in this pass, so its half is undecided")
        verdict["D3"] = None if named is False and True else None
        verdict["D3"] = None                      # only half the hypothesis is testable here
        detail["D3"] += f"; the decidable half {'holds' if not named else 'fails'}"

    # D4: remote homology stays below the magnitude floor. Only composition was erased here.
    g = C[C.task == "remote_homology"]
    if g.empty:
        detail["D4"], verdict["D4"] = "remote homology not scored", None
    else:
        u = ucl(g.effect)
        detail["D4"] = (f"remote homology, composition only: mean {g.effect.mean():+.4f} over {len(g)} encoders, "
                        f"one-sided 95% upper limit {u:+.4f} against delta {DELTA}: "
                        f"{'below' if np.isfinite(u) and u < DELTA else 'not below'}. The other five properties were "
                        f"not erased in this pass, so the hypothesis as written is undecided")
        verdict["D4"] = None

    if a.markdown or a.latex:
        if a.latex:
            print("% generated by scripts/holdout_report.py; do not edit by hand")
            print("\\begin{tabular}{lp{0.36\\linewidth}p{0.42\\linewidth}l}")
            print("\\toprule")
            print(" & prediction, fixed before the cohort was scored & result on the held-out cohort & verdict \\\\")
            print("\\midrule")
            TEX = str.maketrans({"_": "\\_", "%": "\\%", "&": "\\&"})
            for h in freeze["hypotheses"]:
                v = verdict[h["id"]]
                vs = "undecided" if v is None else ("confirmed" if v else "not confirmed")
                print(f"{h['id']} & {h['claim'].translate(TEX)} & {detail[h['id']].translate(TEX)} & {vs} \\\\")
            print("\\bottomrule")
            print("\\end{tabular}")
        else:
            print("| | prediction | result | verdict |")
            print("|---|---|---|---|")
            for h in freeze["hypotheses"]:
                v = verdict[h["id"]]
                vs = "undecided" if v is None else ("confirmed" if v else "not confirmed")
                print(f"| {h['id']} | {h['claim']} | {detail[h['id']]} | {vs} |")
        return

    pd.set_option("display.width", 220)
    print(f"tasks scored: {tasks}")
    print(C.round(4).to_string(index=False))
    print()
    for h in freeze["hypotheses"]:
        v = verdict[h["id"]]
        vs = "UNDECIDED" if v is None else ("CONFIRMED" if v else "NOT CONFIRMED")
        print(f"{h['id']}: {vs}\n   rule:   {h['decision']}\n   result: {detail[h['id']]}")
    json.dump({"verdicts": {k: v for k, v in verdict.items()}, "detail": detail},
              open(CD / "holdout_verdicts.json", "w"), indent=2)


if __name__ == "__main__":
    main()
