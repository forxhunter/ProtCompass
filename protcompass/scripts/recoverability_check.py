"""Is the property still recoverable after erasure by a probe that LEACE does not constrain?

The erasure check in the sweep is a linear probe, and LEACE removes linear predictability by construction, so an
obvious objection is that the check cannot fail and therefore verifies nothing. That is right about the linear
probe and it is why this exists: the same before-and-after measurement is repeated with a two-layer MLP, which
LEACE makes no promise about. If the property is still recoverable nonlinearly, the erasure removed access rather
than information, and every effect in the sweep has to be read as the cost of removing linear access.

Run over a spanning subset rather than the whole sweep, because the point is the size of the residual, not another
33-encoder table.

  python recoverability_check.py --tasks stability solubility --encoders esm2_650m esm_if1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import common_struct
from common_struct import INDEPENDENT_BLOCKS
from erasure import OUT, TASK_KIND, concepts_for, leace_eraser, load_task, probe_concept

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASKS = ["stability", "solubility", "membrane_soluble", "subcellular_localization"]
ENCODERS = ["esm2_650m", "esm1b", "prottrans_t5_xl_u50", "saprot", "esm_if1", "proteinmpnn", "gvp", "gearnet_pretrained"]


def mlp_probe(Xtr, Ztr, Xte, Zte, categorical, seed=0, hidden=256, epochs=40):
    """The same quantity `probe_concept` measures, with a two-layer head instead of a linear one."""
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    A = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=DEV)
    C = torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=DEV)
    if categorical:
        y = torch.tensor(Ztr.argmax(1), dtype=torch.long, device=DEV)
        yte = Zte.argmax(1)
        n_out = int(max(y.max().item(), yte.max()) + 1)
        loss_fn = nn.CrossEntropyLoss()
    else:
        y = torch.tensor(np.atleast_2d(Ztr.T).T, dtype=torch.float32, device=DEV)
        yte = np.atleast_2d(Zte.T).T
        n_out = y.shape[1]
        loss_fn = nn.MSELoss()
    net = nn.Sequential(nn.Linear(A.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, n_out)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        for i in range(0, len(A), 512):
            opt.zero_grad()
            loss_fn(net(A[i:i + 512]), y[i:i + 512]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        p = net(C).cpu().numpy()
    if categorical:
        return float((p.argmax(1) == yte).mean())
    r = [float(np.corrcoef(p[:, j], yte[:, j])[0, 1]) if np.std(p[:, j]) > 0 and np.std(yte[:, j]) > 0 else 0.0
         for j in range(yte.shape[1])]
    return float(np.mean(r))


def conditional_target(name, conc, ok):
    others = [c for c in INDEPENDENT_BLOCKS if c != name and c in conc]
    Ztr = conc[name][0][ok]
    if not others:
        return Ztr, conc[name][2]
    O = np.concatenate([conc[c][0] for c in others], 1)[ok]
    coef = np.linalg.lstsq(O, Ztr, rcond=None)[0]
    Ote = np.concatenate([conc[c][2] for c in others], 1)
    return Ztr - O @ coef, conc[name][2] - Ote @ coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--encoders", nargs="+", default=ENCODERS)
    a = ap.parse_args()
    rows = []
    for task in a.tasks:
        common_struct.set_task(task)
        for enc in a.encoders:
            data = load_task(task, enc)
            if data is None:
                continue
            (Xtr, ytr, itr), (Xva, yva, iva), (Xte, yte, ite) = (data[s] for s in ("train", "valid", "test"))
            conc = concepts_for({"train": itr, "valid": iva, "test": ite})
            ok = np.ones(len(Xtr), bool)
            for name in conc:
                Zt, Zt_te = conditional_target(name, conc, ok)
                mu, P, r, B = leace_eraser(Xtr, Zt)
                if r == 0 or B.shape[1] < 8:
                    continue
                er = lambda Z: ((Z - mu) @ P.T) @ B
                cat = conc[name][3]
                lin_b = probe_concept(Xtr, Zt, Xte, Zt_te, cat)
                lin_a = probe_concept(er(Xtr), Zt, er(Xte), Zt_te, cat)
                mlp_b = mlp_probe(Xtr, Zt, Xte, Zt_te, cat)
                mlp_a = mlp_probe(er(Xtr), Zt, er(Xte), Zt_te, cat)
                rows.append(dict(task=task, encoder=enc, concept=name, rank=r,
                                 linear_before=lin_b, linear_after=lin_a,
                                 mlp_before=mlp_b, mlp_after=mlp_a,
                                 linear_kept=lin_a / lin_b if lin_b > 1e-9 else np.nan,
                                 mlp_kept=mlp_a / mlp_b if mlp_b > 1e-9 else np.nan))
                print(f"  {task:24s} {enc:20s} {name:28s} linear {lin_b:.3f}->{lin_a:.3f}  "
                      f"mlp {mlp_b:.3f}->{mlp_a:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "recoverability_check.csv", index=False)
    d = pd.DataFrame(rows)
    print("\n=== fraction of the readout that survives the erasure ===")
    print(d.groupby("concept")[["linear_kept", "mlp_kept"]].median().round(3).to_string())
    print(f"\nover {len(d)} cells: median linear {d.linear_kept.median():.3f}, median MLP {d.mlp_kept.median():.3f}")


if __name__ == "__main__":
    main()
