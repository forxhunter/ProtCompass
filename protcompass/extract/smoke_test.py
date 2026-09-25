"""Load each encoder once and run it on a few real proteins: shape, determinism, speed, errors."""
import argparse
import json
import time
import traceback

import numpy as np
import pandas as pd

from common import HERE, SPECS, Encoder, load_backbone

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--encoders", nargs="+", default=None); ap.add_argument("--env", default=None)
    a = ap.parse_args()
    man = pd.read_csv(HERE / "manifest" / "stability.csv").query("split == 'test'").head(3)
    prots = [(r.sequence, *load_backbone(r.structure_path, r.sequence)[:1], r.structure_path) for r in man.itertuples()]
    keys = a.encoders or [s.key for s in SPECS if a.env is None or s.env == a.env]
    rows = []
    for key in keys:
        t0 = time.time()
        try:
            enc = Encoder(key)
            load_s = time.time() - t0
            t1 = time.time()
            outs = [enc.residue(s, coords=c, pdb_path=p) for s, c, p in prots]
            per = (time.time() - t1) / len(prots)
            again = enc.residue(prots[0][0], coords=prots[0][1], pdb_path=prots[0][2])
            det = float(np.abs(again - outs[0]).max())
            rows.append({"encoder": key, "ok": True, "dim": outs[0].shape[1], "load_s": round(load_s, 1), "s_per_protein": round(per, 3),
                         "max_abs_diff_repeat": det, "weights_sha256": enc.meta.get("weights_sha256", enc.meta.get("checkpoint"))})
            del enc
        except Exception as e:
            rows.append({"encoder": key, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"})
            traceback.print_exc()
        print(json.dumps(rows[-1]), flush=True)
    out = HERE / "audit" / f"smoke_{a.env or 'custom'}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
