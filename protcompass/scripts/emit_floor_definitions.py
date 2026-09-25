"""The eight non-learned floors, with what each contains and how wide it is.

The review could not tell what the floors are: the paper named them and gave one dimension. A floor comparison is
only interpretable if the reader can see what the descriptor holds and how many features the probe was given, since
a wider descriptor is easier to read with a fixed head. Both are printed here from the code that builds them
(`scripts/run_probing_suite.py`) and the released runs.

  python emit_floor_definitions.py
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
from pathlib import Path

import pandas as pd

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))

# atoms and window are part of the definition, not commentary: a reviewer could not tell from the paper whether a
# floor carries a precomputed dihedral or recovers one from a window, which is the difference between an upper
# bound on what the input determines and a partial answer key.
FLOORS = [
    ("ref\\_length", "--", "--", "log length and length/600", "trivial baseline"),
    ("ref\\_position", "--", "--",
     "residue index, distance to the nearer terminus and its log, both scaled by length, log length",
     "trivial baseline"),
    ("ref\\_aa", "--", "own residue", "the residue's amino-acid one-hot; at protein level, the composition",
     "an encoder handed residue identity gets no credit for recovering it"),
    ("ref\\_seq\\_window", "--", "$\\pm 8$",
     "amino-acid one-hot per offset with an out-of-chain flag; no property scale",
     "context-matches the sequence side to the coordinate floors, which see a neighbourhood"),
    ("ref\\_seq\\_expert", "--", "$\\pm 8$, smoothed $3/7/15$",
     "the one-hot window, seven published residue scales over the same window, their means over three window "
     "widths, chain composition, position",
     "the sequence side needed a descriptor of the effort ref\\_global received"),
    ("ref\\_seq\\_global", "--", "$\\pm 8$, smoothed to $61$",
     "ref\\_seq\\_expert plus the chain's 400 dipeptide frequencies and two long-range scale averages",
     "the non-local sequence analogue of ref\\_global"),
    ("ref\\_coords", "CA", "16 nearest, five shells",
     "sorted 16 nearest CA distances, chain-neighbour distances, contact counts at 5, 7, 9, 11, 13~\\AA{}",
     "first coordinate floor; CA only, which understates the input"),
    ("ref\\_backbone", "N, CA, C, O, CB", "$\\pm 2$",
     "all pairwise distances among the $5 \\times 5$ atoms of the window, one chirality sign per offset, plus the "
     "ref\\_coords kNN and shell block. No angle is precomputed",
     "encoders receive N, CA, C, O and CB, so a CA-only floor is too weak"),
    ("ref\\_global", "N, CA, C, O, CB", "$\\pm 2$ plus whole chain",
     "ref\\_backbone plus the residue's distance and rank from the centroid and the leading eigenvector "
     "components of the 8~\\AA{} contact map",
     "ref\\_backbone is local, so non-local probes still favoured the encoders"),
    ("ref\\_pairhist", "CA", "whole chain",
     "the joint histogram of (sequence separation, CA--CA distance) over all pairs, contact-map spectra at two "
     "cutoffs, and the residue's own row of the histogram",
     "the non-local and fold-level probes needed the pair-distance distribution of the whole chain"),
]


def _open(cols, header):
    print("% generated file: the whole tabular, so the caller never inputs rows into an open tabular")
    print("\\begin{tabular}{" + cols + "}")
    print("\\toprule")
    print(header + " \\\\")
    print("\\midrule")


def _close():
    print("\\bottomrule")
    print("\\end{tabular}")


def widths():
    """The width recorded with every probe run of that source. The released runs store one width per source, so
    this is the residue-level width; the protein-level twin of each coordinate floor is narrower and is described
    in the content column rather than given a second number it cannot support."""
    out = {}
    for f in sorted((ROOT / "probing_results" / "by_source").glob("ref_*.csv")):
        d = pd.read_csv(f, usecols=["source", "dim"])
        out[f.stem] = int(d.dim.iloc[0])
    return out


def main():
    w = widths()
    _open("lrllp{0.30\\linewidth}p{0.20\\linewidth}",
          "floor & width & atoms & window & what it contains & why it was added")
    for name, atoms, window, content, why in FLOORS:
        key = name.replace("\\_", "_")
        print(f"\\texttt{{{name}}} & {w.get(key, '--')} & {atoms} & {window} & {content} & {why} \\\\")
    _close()
    print("% width is the number of features the probe receives at residue level; all coordinates enter as "
          "distances and one chirality sign, never as an angle or a frame")


if __name__ == "__main__":
    main()
