"""Name the ProteinWorkshop ECReaction class indices with EC numbers.

The ECReaction label file only stores an integer class per PDB chain. Swiss-Prot links PDB entries to EC
numbers, so each class index is named by a majority vote over the EC numbers of its member PDB entries.
Output: data/ec_class_names.json with, per class index, the voted EC number, its top-level class (1-7),
the number of votes and the share of votes that agree with the winner at the top level.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import collections
import gzip
import json
import re
from pathlib import Path

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
SPROT = ROOT / "structure_encoder_data/ec/uniprot_sprot.dat.gz"
CHAINS = ROOT / "structure_encoder_data/proteinworkshop/ECReaction/chain_functions.txt"
OUT = ROOT / "./atlas/data/ec_class_names.json"
EC_RE = re.compile(r"EC=(\d+\.[\d\-n]+\.[\d\-n]+\.[\d\-n]+)")


def pdb_to_ec():
    mapping = collections.defaultdict(set)
    ecs, pdbs = set(), []
    with gzip.open(SPROT, "rt") as fh:
        for line in fh:
            if line.startswith("DE"):
                ecs.update(EC_RE.findall(line))
            elif line.startswith("DR   PDB;"):
                pdbs.append(line.split(";")[1].strip().lower())
            elif line.startswith("//"):
                # only entries with a single EC number give an unambiguous vote
                if len(ecs) == 1:
                    for p in pdbs:
                        mapping[p].update(ecs)
                ecs, pdbs = set(), []
    return mapping


def main():
    mapping = pdb_to_ec()
    print("PDB entries with an EC number in Swiss-Prot:", len(mapping))
    votes = collections.defaultdict(collections.Counter)
    n_chain = 0
    for line in open(CHAINS):
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        n_chain += 1
        pdb = parts[0].split(".")[0].lower()
        ec = mapping.get(pdb)
        if ec and len(ec) == 1:
            votes[parts[1].strip()][next(iter(ec))] += 1
    out = {}
    for cls, counter in votes.items():
        winner, _ = counter.most_common(1)[0]
        total = sum(counter.values())
        top = winner.split(".")[0]
        agree = sum(c for ec, c in counter.items() if ec.split(".")[0] == top)
        out[cls] = {"ec": winner, "top_level": int(top), "votes": total, "top_level_agreement": round(agree / total, 3)}
    OUT.write_text(json.dumps(out, indent=0))
    agreements = [v["top_level_agreement"] for v in out.values()]
    print(f"chains: {n_chain}; classes named: {len(out)}; median top-level agreement: {sorted(agreements)[len(agreements) // 2]:.3f}; "
          f"classes with agreement < 0.8: {sum(a < 0.8 for a in agreements)}")


if __name__ == "__main__":
    main()
