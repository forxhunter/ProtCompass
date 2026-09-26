"""The causal demand experiment with the controls a later check showed were necessary.

What changed, and why. The first version compared a LEACE erasure of a named property against erasing a random
subspace of the same rank. Those two arms are not the same class of transformation: LEACE removes the directions its
un-whitening step biases toward high variance, so it removed 6.7 to 23 times more of the representation's variance
than the rank-matched random control. A check then showed the consequence directly: a composition vector
row-permuted across proteins, carrying no relation to the protein it is assigned to, produced a mean effect of +0.029
on stability with 9 of 10 encoders positive and a sign test at p = 0.021 — it passed this experiment's own
significance rule. Fifteen of the eighteen effects reported as demands were at or below that sham level.

So this version measures three arms with the same rank r, and reports the effect against the sham:

  erased   LEACE fitted to the property.
  sham     LEACE fitted to the same property row-permuted across proteins, averaged over permutation seeds. The same
           map, the same rank, a target with the same marginal distribution and no relation to the protein. This is
           the null, and `effect = sham - erased`.
  pcs      Erase the top-r principal directions. A content-free intervention that removes the most variance any rank-r
           erasure can, so it is the ceiling a property effect is read against rather than a null.
  rotation The old random-rotation control, kept so the two versions can be compared.

Also fixed here:
  - Conditional erasure runs over the linearly independent blocks only. `hydrophobic fraction` and `charge` are exact
    linear functions of `amino-acid composition`, so conditioning them on the other blocks left float32 rounding error
    and the erasure was fitted to that. They are measured instead as contrasts inside composition, conditioned on
    everything except composition, and labelled as such.
  - The erasure check is applied to the conditional arm too. The first version verified it only on the marginal arm,
    so the reported table was never gated on the erasure having worked.
  - Every concept is additionally erased at a common rank, because the property ordering in the first version was
    partly the advantage of being erased at rank 15 rather than rank 1.

  python erasure_v2.py --tasks stability --encoders esm2_650m esm_if1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common_struct
from common_struct import DEPENDS_ON, INDEPENDENT_BLOCKS
from erasure import (EMB, OUT, TASK_KIND, concepts_for, fit_score, leace_eraser, load_task, probe_concept,
                     random_eraser)
from matched_null import matched_null_projector, property_share_removed, variance_removed

SHAM_SEEDS = (0, 1, 2)
NULL_SEEDS = (0, 1, 2, 3)
WITH_MATCHED_NULL = False
MATCHED_RANK = 2          # every concept also erased at this rank, so the ordering is not a rank artefact


def top_pc_eraser(X, r):
    """An orthonormal basis of the complement of the top-r principal directions: the variance ceiling."""
    Xc = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[r:].T


def pls_reduce(X, Z, k, Zte=None):
    """Z restricted to its top-k directions of covariance with X, so a concept can be erased at a chosen rank.

    The basis is fitted on train and applied to test as well, because the erasure check re-probes the target that was
    actually erased; probing the full-width target against a reduced one silently compares different quantities."""
    if Z.shape[1] <= k:
        return Z, Zte
    mu = Z.mean(0, keepdims=True)
    Xc, Zc = X - X.mean(0, keepdims=True), Z - mu
    S = Xc.T @ Xc / len(Xc) + 1e-6 * np.eye(X.shape[1])
    vals, vecs = np.linalg.eigh(S)
    W = vecs @ np.diag(np.clip(vals, 1e-6, None) ** -0.5) @ vecs.T
    _, _, Vt = np.linalg.svd(W @ (Xc.T @ Zc / len(Xc)), full_matrices=False)
    A = Vt[:k].T
    return Zc @ A, (None if Zte is None else (Zte - mu) @ A)


def arms(Xtr, ytr, Xva, yva, Xte, yte, kind, Ztr, Zte, cat):
    """Score the erased arm and its three controls, all at the rank the property occupies."""
    mu, P, r, B = leace_eraser(Xtr, Ztr)
    if r == 0 or B.shape[1] < 8:
        return None
    er = lambda Z: ((Z - mu) @ P.T) @ B
    out = dict(rank=r, dim=Xtr.shape[1])
    out["erased"] = fit_score(er(Xtr), ytr, er(Xva), yva, er(Xte), yte, kind)
    out["concept_before"] = probe_concept(Xtr, Ztr, Xte, Zte, cat)
    out["concept_after"] = probe_concept(er(Xtr), Ztr, er(Xte), Zte, cat)

    # the null: the same map fitted to the same property, permuted across proteins
    sham, sham_var = [], []
    for s in SHAM_SEEDS:
        rng = np.random.default_rng(s)
        Zs = Ztr[rng.permutation(len(Ztr))]
        mus, Ps, rs, Bs = leace_eraser(Xtr, Zs)
        if rs == 0 or Bs.shape[1] < 8:
            continue
        es = lambda Z, mus=mus, Ps=Ps, Bs=Bs: ((Z - mus) @ Ps.T) @ Bs
        sham.append(fit_score(es(Xtr), ytr, es(Xva), yva, es(Xte), yte, kind))
        sham_var.append(variance_removed(Xtr, es))
    out["sham"] = float(np.mean(sham)) if sham else np.nan
    out["sham_sd"] = float(np.std(sham)) if len(sham) > 1 else np.nan
    out["var_sham"] = float(np.mean(sham_var)) if sham_var else np.nan

    # the ceiling: erase the r highest-variance directions
    V = top_pc_eraser(Xtr, r)
    mu0 = Xtr.mean(0, keepdims=True)
    pc = lambda Z: (Z - mu0) @ V
    out["pcs"] = fit_score(pc(Xtr), ytr, pc(Xva), yva, pc(Xte), yte, kind)

    # the old control, for comparability with version 1
    rot = []
    for s in (0, 1):
        Q = random_eraser(Xtr.shape[1], r, s)
        cr = lambda Z: (Z - mu0) @ Q
        rot.append(fit_score(cr(Xtr), ytr, cr(Xva), yva, cr(Xte), yte, kind))
    out["rotation"] = float(np.mean(rot))

    if WITH_MATCHED_NULL:
        # The sham matches the rank of the real map but not the amount it removes: across the panel the real map
        # takes 9% to 24% of the representation variance and the sham 1.4% to 3.4%. This arm matches the amount as
        # well, in principal directions carrying as little of the property as the search finds, so the contrast
        # against it is a contrast about which directions were removed rather than about how many.
        target = variance_removed(Xtr, er)
        out["var_erased"] = target
        out["prop_erased"] = property_share_removed(Xtr, Ztr, er)
        vals, vvar, vprop = [], [], []
        for s_ in NULL_SEEDS:
            mun, Pn, _, Bn, _, _ = matched_null_projector(Xtr, Ztr, r, target, seed=300 + s_)
            en = lambda Z, mun=mun, Pn=Pn, Bn=Bn: ((Z - mun) @ Pn.T) @ Bn
            vals.append(fit_score(en(Xtr), ytr, en(Xva), yva, en(Xte), yte, kind))
            vvar.append(variance_removed(Xtr, en))
            vprop.append(property_share_removed(Xtr, Ztr, en))
        out["null"] = float(np.mean(vals))
        out["null_sd"] = float(np.std(vals))
        out["var_null"] = float(np.mean(vvar))
        out["prop_null"] = float(np.mean(vprop))
        out["effect_null"] = out["null"] - out["erased"]

    out["effect"] = out["sham"] - out["erased"]                    # the number to read
    out["effect_v1"] = out["rotation"] - out["erased"]             # what version 1 reported
    out["effect_vs_ceiling"] = out["effect"] / (out["sham"] - out["pcs"]) if out["sham"] > out["pcs"] else np.nan
    out["erasure_ok"] = bool(out["concept_after"] < 0.5 * out["concept_before"])
    return out


def safe_arms(*a, **kw):
    """`arms` with the numerical failures recorded instead of raised.

    A single arm that fails used to take the worker down with it, and every cell the worker had not yet written was
    lost. Four cells failed this way on a RidgeCV decomposition of the PC-erased matrix. A failed arm is recorded as
    a row with `failed` set and no scores, so the cell counts as attempted and the analysis, which reads only rows
    with `erasure_ok`, ignores it."""
    where = a[-1]
    try:
        return arms(*a[:-1], **kw)
    except Exception as e:
        print(f"  [failed]  {where}: {type(e).__name__}: {e}", flush=True)
        return {"rank": 0, "erasure_ok": False, "failed": f"{type(e).__name__}: {e}"[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASK_KIND))
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--matched-rank", type=int, default=MATCHED_RANK)
    ap.add_argument("--concepts", nargs="+", default=None,
                    help="restrict to these property blocks; the default runs all eight")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="restrict to these arms, e.g. conditional; the default runs all of them")
    ap.add_argument("--matched-null", action="store_true",
                    help="also score the variance-matched null of scripts/matched_null.py")
    ap.add_argument("--parts", default="parts", help="subdirectory of the erasure output to write and resume from")
    a = ap.parse_args()
    global WITH_MATCHED_NULL
    WITH_MATCHED_NULL = a.matched_null
    import os
    # Every process writes its own file, keyed by pid as well as by the shard suffix. An earlier version reused the
    # shard name across restarts, so a restart read the previous run's rows into `done`, skipped them, and then
    # overwrote the file with only its own rows: the skipped cells were deleted from disk and not recomputed.
    # Membrane/soluble lost 9 of 14 encoders that way. With a unique file per process the resume glob still sees
    # every finished cell and nothing is ever truncated.
    suffix = (os.environ.get("OUT_SUFFIX") or "") + f"_pid{os.getpid()}"
    (OUT / a.parts).mkdir(parents=True, exist_ok=True)
    dest = OUT / a.parts / f"erasure_v2{suffix}.csv"
    encoders = a.encoders or sorted(d.name for d in (EMB / "stability").iterdir() if (d / "meta.json").exists())

    # Resume: a (task, encoder) already present in any part file is skipped, so the sweep can be stopped and
    # restarted at a different worker count without losing or repeating finished cells.
    done = set()
    for f in (OUT / a.parts).glob("erasure_v2*.csv"):
        try:
            prev = pd.read_csv(f, usecols=["task", "encoder"])
        except Exception:
            continue
        done |= set(map(tuple, prev.drop_duplicates().values))

    rows = []
    for task in a.tasks:
        kind = TASK_KIND[task]
        common_struct.set_task(task)
        for enc in encoders:
            if (task, enc) in done:
                continue
            data = load_task(task, enc)
            if data is None:
                continue
            (Xtr, ytr, itr), (Xva, yva, iva), (Xte, yte, ite) = (data[s] for s in ("train", "valid", "test"))
            ok = np.ones(len(ytr), bool) if ytr.ndim > 1 else ~np.isnan(ytr)
            if ok.sum() < 200:
                continue
            conc = concepts_for({"train": itr, "valid": iva, "test": ite})
            names = [c for c in conc if a.concepts is None or c in a.concepts]
            if not names:
                continue
            base = fit_score(Xtr[ok], ytr[ok], Xva, yva, Xte, yte, kind)

            for name in names:
                Ztr, Zva, Zte, cat = conc[name]
                common = dict(task=task, encoder=enc, concept=name, base=base)

                targets = [("marginal", Ztr[ok], Zte)]
                Zm, Zmte = pls_reduce(Xtr[ok], Ztr[ok], a.matched_rank, Zte)
                targets.append((f"matched rank {a.matched_rank}", Zm, Zmte))
                targets = [t for t in targets if a.arms is None or t[0] in a.arms]
                for arm_name, Zt, Zt_te in targets:
                    r = safe_arms(Xtr[ok], ytr[ok], Xva, yva, Xte, yte, kind, Zt, Zt_te, cat,
                                  f"{task}/{enc}/{name}/{arm_name}")
                    if r:
                        rows.append({**common, "arm": arm_name, **r})

                # conditional: only over blocks that are not exact functions of one another
                if name in DEPENDS_ON:
                    others = [c for c in names if c != name and c != DEPENDS_ON[name] and c not in DEPENDS_ON]
                    arm_name = f"contrast within {DEPENDS_ON[name]}"
                else:
                    others = [c for c in INDEPENDENT_BLOCKS if c != name and c in conc]
                    arm_name = "conditional"
                if not others or (a.arms is not None and arm_name not in a.arms):
                    continue
                O = np.concatenate([conc[c][0] for c in others], 1)[ok]
                coef = np.linalg.lstsq(O, Ztr[ok], rcond=None)[0]
                Zc = Ztr[ok] - O @ coef
                Ote = np.concatenate([conc[c][2] for c in others], 1)
                Zc_te = Zte - Ote @ coef                    # the same projection, so the check probes what was erased
                rel = np.linalg.norm(Zc) / max(np.linalg.norm(Ztr[ok] - Ztr[ok].mean(0)), 1e-12)
                if rel < 1e-3:
                    # the residual is rounding error, not a property; record the abstention rather than erasing noise
                    rows.append({**common, "arm": arm_name, "rank": 0, "residual_frac": rel, "erasure_ok": False})
                    print(f"  [skip]  {task:12s} {enc:18s} {name:28s} residual {rel:.2e} of signal, abstaining",
                          flush=True)
                    continue
                r = safe_arms(Xtr[ok], ytr[ok], Xva, yva, Xte, yte, kind, Zc, Zc_te, cat,
                              f"{task}/{enc}/{name}/{arm_name}")
                if r and "failed" not in r:
                    rows.append({**common, "arm": arm_name, "residual_frac": rel, **r})
                    print(f"  [{arm_name[:9]:9s}] {task:12s} {enc:18s} {name:28s} rank {r['rank']:3d}  "
                          f"erased {r['erased']:.3f}  sham {r['sham']:.3f}  pcs {r['pcs']:.3f}  "
                          f"effect {r['effect']:+.3f} (v1 said {r['effect_v1']:+.3f})", flush=True)
                pd.DataFrame(rows).to_csv(dest, index=False)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"\nwritten: {dest}")


if __name__ == "__main__":
    main()
