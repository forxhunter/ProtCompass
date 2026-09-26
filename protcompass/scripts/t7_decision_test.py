"""A prospective decision: which descriptor should repair a restricted representation on each task?

The add-back of Appendix B.14 answers a question about the erasure it inverts. This test asks for something else: a
decision the measurement makes in advance, tested on data that was not used to make it. This is that test.

The decision. A practitioner has a representation that is too small for the task and one descriptor's worth of extra
input to spend. Which descriptor? The released erasure results answer per task, and the answer is not the same
everywhere: amino-acid composition on membrane and soluble and on subcellular localisation, and burial and packing
on fluorescence, which Appendix B.10 records as the one task where a geometric property leads a compositional one.
That difference is what makes the prediction falsifiable rather than a restatement of "composition usually helps".

The protocol. The restricted representation is the encoder projected to 32 dimensions by a PCA fitted on the
released training split, identical for every task and every descriptor, so nothing about the restriction is chosen
per prediction. Each candidate descriptor block is concatenated to it, the head is refitted under the released
protocol, and the score is read once on the held-out cohort. The comparators are all present: no
descriptor, composition only, every descriptor at once, a descriptor predicted not to help, and the descriptor a
validation split would have chosen.

The prediction is written to a freeze file before any held-out score exists, and is decided as written.

  python t7_decision_test.py --freeze        # write the prediction, once
  python t7_decision_test.py                 # score it
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import common_struct                                                    # noqa: E402
from common_struct import INDEPENDENT_BLOCKS                            # noqa: E402
from addback_holdout import KIND, load_holdout, load_released, extend_concept_table   # noqa: E402

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
CD = ROOT / "confirmation_downstream"
OUT = ROOT / "confirmation_decision"
K_RESTRICT = 32
TASKS = ["membrane_soluble", "subcellular_localization", "fluorescence", "stability"]
PANEL = ["esm2_650m", "prottrans_t5_xl_u50", "ankh_base", "esm3", "esm_if1", "proteinmpnn"]

# Written before any held-out score exists. Each entry is the block the released erasure names on that task, and the
# block predicted not to help, which is the strongest undemanded one there.
PREDICTION = {
    "membrane_soluble": {"helps": "amino-acid composition", "does_not": "contact order",
                         "released": "composition +0.051; contact order +0.0003"},
    "subcellular_localization": {"helps": "amino-acid composition", "does_not": "contact order",
                                 "released": "composition +0.035; contact order -0.0009"},
    "fluorescence": {"helps": "burial and packing", "does_not": "contact order",
                     "released": "burial and packing +0.050 leads composition +0.036; contact order -0.0009"},
    "stability": {"helps": None, "does_not": None,
                  "released": "no property is named; the task is scored as a control and carries no prediction"},
}
DECISION = ("On each task carrying a prediction, the named block gives the largest gain over the restricted "
            "representation among the single blocks, and a strictly larger gain than the block predicted not to "
            "help. Confirmed on a task when both hold on a majority of the panel and on the panel mean.")


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "freeze.json"
    if f.exists():
        raise SystemExit(f"{f} exists; the prediction is written once")
    body = dict(written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                restriction=f"PCA to {K_RESTRICT} dimensions, fitted on the released training split only",
                tasks=TASKS, panel=PANEL, prediction=PREDICTION, decision_rule=DECISION,
                comparators=["none", "composition only", "all descriptors", "the block predicted not to help",
                             "the block a validation split selects"],
                evaluated_on="the frozen held-out cohorts of confirmation_downstream/freeze.json, used once",
                not_claimed="these proteins share their source datasets with the scored rows; this is a decision "
                            "tested on uninspected proteins, not on a new distribution")
    f.write_text(json.dumps(body, indent=2) + "\n")
    print(f"prediction frozen: {f}")
    for t, p in PREDICTION.items():
        print(f"  {t:26s} helps: {p['helps']}   does not: {p['does_not']}")
    print(f"\nsha256: {hashlib.sha256(f.read_bytes()).hexdigest()[:16]}")


def fit(Xtr, ytr, Xva, yva, Xte, yte, kind):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    A, B, C = ((Z - mu) / sd for Z in (Xtr, Xva, Xte))
    if kind == "reg":
        m = RidgeCV(alphas=(1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4)).fit(A, ytr)
        v = spearmanr(yva, m.predict(B))[0]
        return float(spearmanr(yte, m.predict(C))[0]), float(v)
    best = None
    for c in (1e-2, 1e-1, 1.0):
        m = LogisticRegression(C=c, max_iter=2000).fit(A, ytr)
        v = (roc_auc_score(yva, m.predict_proba(B)[:, 1]) if kind == "binary" else m.score(B, yva))
        if best is None or v > best[0]:
            best = (v, m)
    v, m = best
    te = (roc_auc_score(yte, m.predict_proba(C)[:, 1]) if kind == "binary" else m.score(C, yte))
    return float(te), float(v)


def run():
    fz = json.loads((OUT / "freeze.json").read_text())
    rows = []
    for task in fz["tasks"]:
        kind = KIND[task]
        common_struct.set_task(task)
        for enc in fz["panel"]:
            rel, hold = load_released(task, enc), load_holdout(task, enc)
            if rel is None or hold is None:
                print(f"  [skip] {task}/{enc}: missing extraction", flush=True)
                continue
            (Xtr, ytr, itr), (Xva, yva, iva) = rel["train"], rel["valid"]
            Xte, yte, ite = hold
            ok = ~np.isnan(ytr)
            Xtr, ytr, itr = Xtr[ok], ytr[ok], [i for i, k in zip(itr, ok) if k]
            extend_concept_table(task, list(itr) + list(iva) + list(ite))
            conc = common_struct.protein_descriptors({"train": itr, "valid": iva, "test": ite}, task=task)
            blocks = [b for b in INDEPENDENT_BLOCKS if b in conc]

            pca = PCA(n_components=min(K_RESTRICT, Xtr.shape[1]), random_state=0).fit(Xtr)
            R = {k: pca.transform(v) for k, v in (("train", Xtr), ("valid", Xva), ("test", Xte))}

            def score(extra):
                A = R["train"] if extra is None else np.hstack([R["train"], extra[0]])
                B = R["valid"] if extra is None else np.hstack([R["valid"], extra[1]])
                C = R["test"] if extra is None else np.hstack([R["test"], extra[2]])
                return fit(A, ytr, B, yva, C, yte, kind)

            base_te, base_va = score(None)
            rec = dict(task=task, encoder=enc, dim=Xtr.shape[1], k=K_RESTRICT, base_test=base_te, base_valid=base_va)
            per_block = {}
            for b in blocks:
                te, va = score((conc[b][0][ok], conc[b][1], conc[b][2]))
                per_block[b] = (te, va)
                rec[f"gain__{b}"] = te - base_te
                rec[f"valid__{b}"] = va - base_va
            allb = tuple(np.concatenate([conc[b][i] for b in blocks], 1) for i in (0, 1, 2))
            te, _ = score((allb[0][ok], allb[1], allb[2]))
            rec["gain__all"] = te - base_te
            best_valid = max(blocks, key=lambda b: per_block[b][1])
            rec["valid_selected"] = best_valid
            rec["gain__valid_selected"] = per_block[best_valid][0] - base_te
            rows.append(rec)
            print(f"  {task:26s} {enc:20s} base {base_te:.4f}  best block "
                  f"{max(blocks, key=lambda b: per_block[b][0])} "
                  f"{max(per_block[b][0] for b in blocks) - base_te:+.4f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "decision_test.csv", index=False)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "decision_test.csv", index=False)
    print(f"\nwritten: {OUT / 'decision_test.csv'} ({len(d)} rows)")
    return d


def report():
    fz = json.loads((OUT / "freeze.json").read_text())
    d = pd.read_csv(OUT / "decision_test.csv")
    blocks = [c[len("gain__"):] for c in d.columns if c.startswith("gain__")
              and c not in ("gain__all", "gain__valid_selected")]
    out = {}
    for task, g in d.groupby("task"):
        pred = fz["prediction"].get(task, {})
        if not pred.get("helps"):
            out[task] = dict(verdict=None, detail="no prediction; scored as a control")
            continue
        helps, nots = pred["helps"], pred["does_not"]
        wins = sum(1 for _, r in g.iterrows()
                   if r[f"gain__{helps}"] == max(r[f"gain__{b}"] for b in blocks))
        beats = sum(1 for _, r in g.iterrows() if r[f"gain__{helps}"] > r[f"gain__{nots}"])
        mean_ok = (g[f"gain__{helps}"].mean() == max(g[f"gain__{b}"].mean() for b in blocks)
                   and g[f"gain__{helps}"].mean() > g[f"gain__{nots}"].mean())
        ok = wins > len(g) / 2 and beats > len(g) / 2 and bool(mean_ok)
        out[task] = dict(verdict=bool(ok),
                         detail=(f"{helps} is the best single block on {wins}/{len(g)} encoders and beats {nots} on "
                                 f"{beats}/{len(g)}; panel mean gain {g[f'gain__{helps}'].mean():+.4f} against "
                                 f"{g[f'gain__{nots}'].mean():+.4f} for {nots}, {g['gain__all'].mean():+.4f} for all "
                                 f"blocks and {g['gain__valid_selected'].mean():+.4f} for the validation-selected "
                                 f"block"))
    json.dump(out, open(OUT / "decision_verdicts.json", "w"), indent=2)
    pd.set_option("display.width", 240)
    cols = ["task", "encoder", "base_test"] + [f"gain__{b}" for b in blocks] + ["gain__all", "gain__valid_selected"]
    print(d[cols].round(4).to_string(index=False))
    print(f"\ndecision rule: {fz['decision_rule']}\n")
    for t, v in out.items():
        state = "UNDECIDED (control)" if v["verdict"] is None else ("CONFIRMED" if v["verdict"] else "NOT CONFIRMED")
        print(f"{t}: {state}\n   {v['detail']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        freeze()
    elif a.report:
        report()
    else:
        run()
        report()
