"""Freeze held-out downstream cohorts before they are scored, for the confirmation of the demand results.

Every erasure, add-back and naming threshold in Section 4.3 was developed on the stage-1 subsample of each task's
manifest. The manifests hold more rows than that subsample: those rows were never extracted, never scored and never
looked at. They are the held-out cohorts here.

This is weaker than the CATH 4.4 construction used for the probing axis. These proteins come from the same source
datasets, so the claim it supports is that no descriptor, threshold or naming rule was developed against them. It is
not a claim that they are new to any encoder's pretraining, and it is not a fold-level or homology-level split.

Where a task has unused rows the source dataset already designates as test, those are preferred, because they were
held out twice. Where it has none, unused validation rows are used and the file says so. Solubility and PPI affinity
have no unused rows at all and cannot be confirmed this way; they are recorded as excluded rather than substituted.

Hypotheses are written here, before any held-out score exists, with the direction and the decision rule.

  python downstream_holdout_freeze.py [--n 2000] [--force]
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
MAN = ROOT / "extract" / "manifest"
OUT = ROOT / "confirmation_downstream"
SEED = 20260922
N_TARGET = 2000

TASKS = ["membrane_soluble", "subcellular_localization", "fluorescence", "stability",
         "ec_classification", "go_mf", "go_bp", "remote_homology"]
PANEL = ["esm2_650m", "prottrans_t5_xl_u50", "ankh_base", "esm3", "esm_if1", "proteinmpnn"]

HYPOTHESES = [
    {"id": "D1", "claim": "Erasing linear access to amino-acid composition costs the retrained head more than the "
                          "sham erasure does, on membrane and soluble and on subcellular localisation.",
     "released": "+0.051 and +0.035 over 33 encoders",
     "decision": "confirmed on a task if the mean conditional effect is positive and the sign test over the panel "
                 "is one-sided p < 0.05"},
    {"id": "D2", "claim": "The named descriptor repairs its own erasure more than it repairs the sham, and the "
                          "permuted copy and the other structural blocks do not.",
     "released": "R = +0.036, +0.015 for composition; within 0.003 of zero for both controls",
     "decision": "confirmed on a task if mean R for composition is positive on every panel encoder and exceeds the "
                 "mean R of both controls"},
    {"id": "D3", "claim": "Where the released run named nothing, nothing is named here either: GO-BP stays below "
                          "the magnitude floor and stability does not hold in sign across the panel.",
     "released": "GO-BP +0.014 below the floor of 0.02; stability +0.070 fails the sign test",
     "decision": "confirmed if neither task passes the released naming rule on the held-out cohort"},
    {"id": "D4", "claim": "Remote homology remains a bound: no property reaches the magnitude floor.",
     "released": "largest effect +0.005, every upper limit at or below +0.009",
     "decision": "confirmed if every Holm-adjusted one-sided 95% upper limit stays below 0.02"},
]


def sha256_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_TARGET)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fz = OUT / "freeze.json"
    if fz.exists() and not a.force:
        raise SystemExit(f"{fz} exists; a freeze is written once")

    rng = np.random.default_rng(SEED)
    cohorts, excluded = {}, {}
    for t in TASKS:
        f = MAN / f"{t}.csv"
        if not f.exists():
            excluded[t] = "no manifest"
            continue
        m = pd.read_csv(f)
        scored = set(m.loc[m.stage1 == 1, "id"])
        un = m[m.stage1 != 1].copy()
        if un.empty:
            excluded[t] = "no unused rows; the whole manifest was scored"
            continue
        # rows the source dataset already calls test are preferred: held out twice
        pool = un[un.split == "test"]
        origin = "unused rows of the source test split"
        if len(pool) < 200:
            pool = un[un.split == "valid"]
            origin = "unused rows of the source validation split"
        if len(pool) < 200:
            pool = un
            origin = "unused rows of any split"
        take = min(a.n, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        sel = pool.iloc[np.sort(idx)]
        assert not (set(sel.id) & scored), f"{t}: a held-out id was already scored"
        cols = [c for c in ("id", "split", "label", "length", "structure_path") if c in sel.columns]
        p = OUT / f"cohort_{t}.csv"
        sel[cols].to_csv(p, index=False)
        cohorts[t] = dict(n=int(take), pool=int(len(pool)), origin=origin,
                          manifest=str(p.relative_to(ROOT)), sha256=sha256_text(p.read_text()),
                          n_scored_in_stage1=int(len(scored)),
                          label_summary=(sel.label.value_counts().head(6).to_dict()
                                         if sel.label.dtype.kind in "iOb"
                                         else dict(mean=float(sel.label.mean()), sd=float(sel.label.std()))))
        print(f"{t:26s} pool {len(pool):6d} -> cohort {take:5d}  ({origin})")
    for t in ("solubility", "ppi_affinity"):
        excluded.setdefault(t, "no unused rows; the whole manifest was scored")

    freeze = dict(
        written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        purpose="confirmation of the Section 4.3 demand results on downstream proteins never scored by this project",
        rule="rows of the released manifest with stage1 != 1, preferring the source dataset's own test split",
        sampling=f"seed {SEED}, up to {a.n} per task",
        panel=PANEL,
        cohorts=cohorts,
        excluded=excluded,
        not_claimed=("these proteins come from the same source datasets as the scored rows, so this establishes "
                     "that no descriptor, threshold or naming rule was developed against them, and nothing about "
                     "pretraining exposure, fold novelty or homology separation"),
        protocol=dict(
            fit="the released stage-1 training rows and their heads, unchanged",
            evaluate="the cohorts above, used once",
            erasure="LEACE on the conditional composition residual, the released construction",
            controls="sham erasure of the permuted property; permuted and alternative descriptors for the add-back",
            no_refitting="no threshold, property or rule may change after this file is written",
        ),
        hypotheses=HYPOTHESES,
    )
    fz.write_text(json.dumps(freeze, indent=2) + "\n")
    print(f"\nexcluded: {excluded}")
    print(f"freeze written: {fz}")
    for h in HYPOTHESES:
        print(f"  {h['id']}: {h['decision']}")


if __name__ == "__main__":
    main()
