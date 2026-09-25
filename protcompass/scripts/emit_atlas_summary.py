"""Per-layer summary of the ten-task atlas: generated/atlas_summary.tex and atlas/data_v4/summary.json.

Encoder class is the paper's declared class (make_atlas_figures_paper.cls), not the atlas metadata.
  python emit_atlas_summary.py
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import json
from pathlib import Path

import numpy as np

import make_atlas_figures_paper as F

OUT_TEX = Path((PROTCOMPASS_ROOT + "/protcompass/paper/generated/atlas_summary.tex"))
OUT_JSON = F.ATLAS.parent / "summary.json"


def stats(layer):
    encs = layer["encoders"]; n = len(encs)
    C = np.array(layer["cka"]); K = np.array(layer["knn_overlap"]); iu = np.triu_indices(n, 1)
    classes = [F.cls(e["key"]) for e in encs]
    same = np.array([classes[i] == classes[j] for i, j in zip(*iu)])
    struct = [i for i in range(n) if classes[i] == "struct"]
    rnd = [i for i in struct if encs[i]["weights"].startswith("random")]; pre = [i for i in struct if i not in rnd]
    def mean_pairs(A, B, same_set):
        v = [C[i, j] for i in A for j in B if (i < j if same_set else True)]
        return float(np.mean(v)) if v else float("nan")
    pur = [(e["metrics"]["knn_purity"], F.disp(e)) for e in encs]
    prb = [(e["metrics"]["linear_probe_acc_cv5"], F.disp(e)) for e in encs]
    return {"n_proteins": layer["n_proteins"], "n_classes": len(layer["label_names"]),
            "cka_mean": float(C[iu].mean()), "cka_min": float(C[iu].min()), "cka_max": float(C[iu].max()),
            "knn_overlap_mean": float(K[iu].mean()),
            "cka_within_class": float(C[iu][same].mean()), "cka_between_class": float(C[iu][~same].mean()),
            "cka_rand_rand": mean_pairs(rnd, rnd, True), "cka_pre_pre": mean_pairs(pre, pre, True),
            "cka_rand_pre": mean_pairs(rnd, pre, False),
            "purity_min": min(pur), "purity_max": max(pur), "purity_median": float(np.median([p for p, _ in pur])),
            "purity_chance": encs[0]["metrics"]["knn_purity_chance"],
            "probe_min": min(prb), "probe_max": max(prb), "majority": encs[0]["metrics"]["majority_class_acc"],
            "distinct_rows_min": min(e["n_distinct_rows"] for e in encs), "n_encoders": n}


def main():
    d = json.load(open(F.ATLAS))
    rows = {name: stats(layer) for name, layer in d["layers"].items()}
    OUT_JSON.write_text(json.dumps(rows, indent=1))
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{\textbf{Latent-space atlas, one layer per scored downstream task.} Each layer is a stratified sample of the "
             r"proteins the task was scored on, labelled by the task's own target (Appendix~\ref{sec:appendix_atlas}). CKA is debiased "
             r"linear CKA between encoders on the full embeddings, averaged over the off-diagonal pairs; within/between is by declared "
             r"class; rand/pre are the randomly initialised and pretrained structure encoders. $k$-NN purity ($k=10$) and the five-fold "
             r"linear probe are given as the range over the encoders with the encoder at each end, against chance and the majority-class rate.}",
             r"\label{tab:atlas_summary}",
             r"\resizebox{\linewidth}{!}{\begin{tabular}{lrrccccccc}", r"\toprule",
             r"layer & $n$ & classes & CKA mean (range) & within / between & rand--rand / pre--pre / rand--pre & purity min & purity max & chance & probe min--max (majority) \\",
             r"\midrule"]
    for name, s in rows.items():
        lines.append(f"{F.LAYER_TITLE.get(name, name)} & {s['n_proteins']:,} & {s['n_classes']} & "
                     f"{s['cka_mean']:.3f} ({s['cka_min']:.2f}--{s['cka_max']:.2f}) & {s['cka_within_class']:.3f} / {s['cka_between_class']:.3f} & "
                     f"{s['cka_rand_rand']:.3f} / {s['cka_pre_pre']:.3f} / {s['cka_rand_pre']:.3f} & "
                     f"{s['purity_min'][0]:.3f} ({s['purity_min'][1]}) & {s['purity_max'][0]:.3f} ({s['purity_max'][1]}) & {s['purity_chance']:.3f} & "
                     f"{s['probe_min'][0]:.3f}--{s['probe_max'][0]:.3f} ({s['majority']:.3f}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    OUT_TEX.write_text("\n".join(lines).replace("\\\\", "\\\\") + "\n")
    print("wrote", OUT_TEX, "and", OUT_JSON)
    for name, s in rows.items():
        print(f"{name:13s} n={s['n_proteins']:5d} k={s['n_classes']:2d} cka={s['cka_mean']:.3f} within/between={s['cka_within_class']:.3f}/{s['cka_between_class']:.3f} "
              f"rr/pp/rp={s['cka_rand_rand']:.3f}/{s['cka_pre_pre']:.3f}/{s['cka_rand_pre']:.3f} purity={s['purity_min'][0]:.3f}({s['purity_min'][1]})..{s['purity_max'][0]:.3f}({s['purity_max'][1]}) chance={s['purity_chance']:.3f} probe={s['probe_min'][0]:.3f}..{s['probe_max'][0]:.3f} maj={s['majority']:.3f} rows>={s['distinct_rows_min']}")


if __name__ == "__main__":
    main()
