"""Is the difference between the project's map and exact LEACE the absolute eigenvalue floor (ridge = 1e-6)?
For three encoders and two fluorescence properties: the covariance spectrum against 1e-6, and the project's map
recomputed with a floor relative to the spectrum (1e-6 x largest eigenvalue) against the official exact map."""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, (PROTCOMPASS_ROOT + "/protcompass/scripts"))
import common_struct
from addback_named_pairs import conditional
from erasure import concepts_for, leace_eraser, load_task
from leace_check import official, metrics
common_struct.set_task("fluorescence")
rows = []
for enc in ("esm2_650m", "proteinmpnn", "gcpnet_pretrained"):
    data = load_task("fluorescence", enc)
    (Xtr, ytr, itr), (Xva, yva, iva), (Xte, yte, ite) = (data[s] for s in ("train", "valid", "test"))
    ok = ~np.isnan(ytr)
    conc = concepts_for({"train": itr, "valid": iva, "test": ite})
    X = Xtr[ok]
    ev = np.linalg.eigvalsh(np.cov(X, rowvar=False))
    for name in ("amino-acid composition", "burial and packing"):
        Z = conditional(conc, name, ok)
        m_o, P_o, r_o = official(X, Z, True)
        Eo = (X - m_o) @ P_o.T
        for label, ridge in (("absolute 1e-6 (released)", 1e-6), ("relative 1e-6 x max eigenvalue", 1e-6 * ev.max()),
                             ("relative 1e-9 x max eigenvalue", 1e-9 * ev.max())):
            mu, P, r, B = leace_eraser(X, Z, ridge=ridge)
            E = (X - mu) @ P.T
            met = metrics(X, Z, X, Z, mu, P)
            rows.append(dict(encoder=enc, concept=name, floor=label, eig_max=ev.max(), eig_share_below_1e6=float((ev < 1e-6).mean()),
                             rank=r, rel_diff_vs_official_exact=float(np.linalg.norm(E - Eo) / np.linalg.norm(X - X.mean(0))),
                             V_loss=met["V_loss"], D_edit=met["D_edit"], residual_xcov=met["residual_xcov"]))
        mo = metrics(X, Z, X, Z, m_o, P_o)
        rows.append(dict(encoder=enc, concept=name, floor="official exact", eig_max=ev.max(), eig_share_below_1e6=float((ev < 1e-6).mean()),
                         rank=r_o, rel_diff_vs_official_exact=0.0, V_loss=mo["V_loss"], D_edit=mo["D_edit"], residual_xcov=mo["residual_xcov"]))
        print(enc, name, flush=True)
pd.DataFrame(rows).to_csv("results_floor.csv", index=False)
print(pd.DataFrame(rows).round(5).to_string(index=False))
