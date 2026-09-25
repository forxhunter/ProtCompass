#!/usr/bin/env python3
"""Collect real structures for the ProtCompass downstream tasks.

Subcommands
  proteinworkshop   EC and GO: convert the local ProteinWorkshop processed graphs (atom37 coordinates) to PDB files
  afdb              DeepLoc 2.0: download AlphaFold DB models by UniProt accession
  scope             TAPE remote homology: extract SCOPe PDB-style domain files from the official bulk archives
  proteingym        Mutation effect: collect the ProteinGym wild-type AF2 structures that are already local
  coverage          Write COVERAGE.md from the manifests

Every step is resumable: existing outputs are kept, downloads go to a .part file first.
Network use is limited to 8 concurrent connections per host, with retries and exponential backoff.
"""
import os as _os  # release: repository root, override with the PROTCOMPASS_ROOT environment variable
PROTCOMPASS_ROOT = _os.environ.get("PROTCOMPASS_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import csv
import io
import json
import os
import pickle
import re
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path((PROTCOMPASS_ROOT + "/protcompass"))
OUT = ROOT / "." / "structures"
PW = ROOT / "structure_encoder_data" / "proteinworkshop"
EMB = ROOT / "ProtCompass_Embeddings" / "embeddings"
COMMON = ROOT / "." / "downstream_fix" / "common_ids"
UA = "Mozilla/5.0 (X11; Linux x86_64) ProtCompass-structure-fetch (academic research)"

# Atom order of the ProteinWorkshop / graphein atom37 tensors. Verified on 300 ECReaction chains (87,231 residues):
# no residue carries an atom outside the set expected for its type, and all covalent bond lengths are in range.
ATOM37 = ["N", "CA", "C", "O", "CB", "OG", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OD1", "ND2", "CG1", "CG2", "CD", "CE",
          "NZ", "OD2", "OE1", "NE2", "OE2", "OH", "NE", "NH1", "NH2", "OG1", "SD", "ND1", "SG", "NE1", "CE3", "CZ2", "CZ3",
          "CH2", "OXT"]
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
       "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
MODIFIED = {"MSE": "MET", "SEP": "SER", "TPO": "THR", "PTR": "TYR", "CSO": "CYS", "CME": "CYS", "MLY": "LYS", "HYP": "PRO",
            "KCX": "LYS", "LLP": "LYS", "CSD": "CYS", "OCS": "CYS", "PCA": "GLU", "M3L": "LYS", "CAS": "CYS", "CSX": "CYS"}
MANIFEST_COLS = ["id", "split", "source", "source_id", "source_version", "task_seq_source", "len_task_seq", "len_struct_seq",
                 "seq_identity", "task_coverage", "exact_match", "frac_full_backbone", "n_nonstandard", "n_chains",
                 "mean_plddt", "renumbered", "status", "reason", "pdb_path"]


# ------------------------------------------------------------------ shared helpers
def atom_line(serial, name, resname, chain, resseq, icode, xyz, occ=1.0, bfac=0.0):
    nm = name if len(name) == 4 else " " + name.ljust(3)
    return ("ATOM  %5d %4s %3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n"
            % (serial % 100000, nm, resname[:3], chain[:1] or "A", resseq, icode[:1] or " ", xyz[0], xyz[1], xyz[2], occ, bfac, name[0]))


def align(task_seq, struct_seq):
    """Global alignment. Returns identity and coverage relative to the task sequence, and task->structure index map."""
    if task_seq == struct_seq:
        return 1.0, 1.0, None
    from Bio import Align
    al = Align.PairwiseAligner()
    al.mode = "global"; al.match_score = 2; al.mismatch_score = -1
    al.open_gap_score = -6; al.extend_gap_score = -0.5
    al.target_end_gap_score = 0; al.query_end_gap_score = 0
    a = al.align(task_seq, struct_seq)[0]
    mapping = [-1] * len(task_seq); ident = 0; cov = 0
    for (t0, t1), (s0, s1) in zip(*a.aligned):
        t0, t1, s0 = int(t0), int(t1), int(s0)
        for k in range(t1 - t0):
            mapping[t0 + k] = s0 + k; cov += 1
            if task_seq[t0 + k] == struct_seq[s0 + k]:
                ident += 1
    n = max(1, len(task_seq))
    return ident / n, cov / n, mapping


def default_status(identity, coverage, frac_bb, L):
    """Default flag. Identity and coverage are relative to the task sequence; unobserved residues lower both,
    so wrong residues are judged on the aligned positions only (identity / coverage)."""
    if L == 0:
        return "unusable", "no residues with coordinates"
    reasons = []
    if coverage > 0 and identity / coverage < 0.95:
        reasons.append("residues differ from the task sequence at aligned positions (aligned identity %.2f)" % (identity / coverage))
    if coverage < 0.70:
        reasons.append("coordinates for %.2f of the task sequence" % coverage)
    if frac_bb < 0.95:
        reasons.append("full backbone for %.2f of residues" % frac_bb)
    return ("check", "; ".join(reasons)) if reasons else ("ok", "")


def restatus():
    """Re-apply default_status to existing manifests and add the aligned_identity column."""
    import pandas as pd
    for f in sorted(OUT.glob("*/manifest.csv")):
        m = pd.read_csv(f, low_memory=False, keep_default_na=False)
        ident, cov = pd.to_numeric(m.seq_identity, errors="coerce"), pd.to_numeric(m.task_coverage, errors="coerce")
        m["aligned_identity"] = (ident / cov).round(4)
        for i in m.index[(m.status != "unusable") & ident.notna()]:
            keep = [r for r in str(m.at[i, "reason"]).split("; ") if r.startswith(("structure covers", "mean pLDDT"))]
            st, rs = default_status(ident[i], cov[i], float(m.at[i, "frac_full_backbone"] or 1), int(m.at[i, "len_struct_seq"] or 0))
            if keep and f.parent.name == "mutation_effect" and ident[i] / max(cov[i], 1e-9) > 0.98:
                st, rs = "ok", keep[0]
            m.at[i, "status"], m.at[i, "reason"] = st, "; ".join([rs] + [k for k in keep if k not in rs]).strip("; ")
        m.to_csv(f, index=False)
        print(f"{f.parent.name:26s} {m.status.value_counts().to_dict()}")



def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = sorted({k for r in rows for k in r} - set(MANIFEST_COLS))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS + extra, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_alignment(adir, pid, task_seq, struct_seq, mapping):
    if mapping is None:
        return
    adir.mkdir(parents=True, exist_ok=True)
    with open(adir / f"{pid}.json", "w") as f:
        json.dump({"id": pid, "task_seq": task_seq, "struct_seq": struct_seq, "task_to_struct": mapping}, f)


def pdb_residues(lines):
    """Parse ATOM/HETATM lines -> ordered residues [(chain, resseq, icode, resname, {atom: (xyz, bfac)})]."""
    res, index = [], {}
    for ln in lines:
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            continue
        if ln[16] not in " A1":
            continue
        key = (ln[21], ln[22:26], ln[26])
        if key not in index:
            index[key] = len(res); res.append([ln[21], int(ln[22:26]), ln[26].strip(), ln[17:20].strip(), {}, ln.startswith("HETATM")])
        try:
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54])); b = float(ln[60:66] or 0)
        except ValueError:
            continue
        res[index[key]][4].setdefault(ln[12:16].strip(), (xyz, b))
    return res


def load_cache_ids(task):
    union = set()
    for p in (EMB / task).glob("*/*_ids.txt"):
        union.update(l.strip() for l in open(p) if l.strip())
    return union


def load_common_ids(task):
    p = COMMON / f"{task}.json"
    if not p.exists():
        return set()
    d = json.load(open(p))
    return set(d.get("train", [])) | set(d.get("test", []))


# ------------------------------------------------------------------ ProteinWorkshop (EC, GO)
def _convert_pw(args):
    pt_path, out_pdb, task_seq = args
    import torch
    try:
        d = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as ex:                                             # unreadable file in the local archive copy
        return dict(id=Path(out_pdb).stem, status="unusable", reason="processed graph unreadable: " + type(ex).__name__,
                    len_task_seq=len(task_seq), len_struct_seq=0, pdb_path="", _task_seq=task_seq, _struct_seq="", _mapping=None)
    if not hasattr(d, "coords"):                                        # PyG bookkeeping files such as pre_filter.pt
        return None
    coords = d.coords.numpy().astype(np.float64)
    present = np.abs(coords - 1e-5).max(-1) > 1e-4                      # graphein fills missing atoms with 1e-5
    rows, seq, n_nonstd, bb_full, numbering = [], [], 0, 0, []
    for i, rid in enumerate(d.residue_id):
        parts = rid.split(":")
        chain, resname, num = parts[0], parts[1], parts[2]
        icode = parts[3] if len(parts) > 3 else ""
        resname = MODIFIED.get(resname, resname)
        if resname not in AA3:
            n_nonstd += 1
        seq.append(AA3.get(resname, "X"))
        bb_full += int(present[i, [0, 1, 2, 3]].all())               # N, CA, C, O
        numbering.append((chain, int(num), icode, resname))
    nums = [(c, n, ic) for c, n, ic, _ in numbering]
    renumber = len(set(nums)) != len(nums) or any(n < -999 or n > 9999 for _, n, _ in nums) or any(len(c) != 1 for c, _, _ in nums)
    serial = 1
    for i, (chain, num, icode, resname) in enumerate(numbering):
        for a, name in enumerate(ATOM37):
            if not present[i, a] or (resname == "GLY" and name == "CB"):
                continue
            rows.append(atom_line(serial, name, resname if resname in AA3 else "UNK", "A" if renumber else chain,
                                  i + 1 if renumber else num, "" if renumber else icode, coords[i, a]))
            serial += 1
    out_pdb = Path(out_pdb)
    with open(out_pdb, "w") as f:
        f.write("REMARK   1 SOURCE ProteinWorkshop processed graph %s (atom37 coordinates of RCSB chain)\n" % Path(pt_path).name)
        f.writelines(rows); f.write("TER\nEND\n")
    if renumber:
        with open(out_pdb.with_suffix(".numbering.tsv"), "w") as f:
            f.write("new_resseq\torig_chain\torig_resseq\torig_icode\tresname\n")
            for i, (c, n, ic, rn) in enumerate(numbering):
                f.write(f"{i + 1}\t{c}\t{n}\t{ic}\t{rn}\n")
    struct_seq = "".join(seq); L = len(seq)
    ts = task_seq if task_seq else struct_seq
    identity, coverage, mapping = align(ts, struct_seq)
    frac_bb = bb_full / max(1, L)
    status, reason = default_status(identity, coverage, frac_bb, L)
    return dict(id=out_pdb.stem, len_task_seq=len(ts), len_struct_seq=L, seq_identity=round(identity, 4), task_coverage=round(coverage, 4),
                exact_match=int(mapping is None), frac_full_backbone=round(frac_bb, 4), n_nonstandard=n_nonstd, n_chains=len({c for c, _, _ in nums}),
                mean_plddt="", renumbered=int(renumber), status=status, reason=reason, pdb_path=str(out_pdb.relative_to(OUT)),
                _task_seq=ts, _struct_seq=struct_seq, _mapping=mapping)


def run_proteinworkshop(workers, only=None):
    norm = lambda s: (s.split(".")[0].lower() + "_" + s.split(".")[1]) if "." in s else (s.split("-")[0].lower() + "_" + s.split("-")[1])
    # ---- EC
    ec_split = {}
    for split in ["training", "validation", "testing"]:
        for l in open(PW / "ECReaction" / f"{split}.txt"):
            if l.strip():
                ec_split[norm(l.strip())] = split
    # ---- GO
    go_split, go_seq = {}, {}
    for split, fn in [("train", "nrPDB-GO_train.txt"), ("valid", "nrPDB-GO_valid.txt"), ("test", "nrPDB-GO_test.txt")]:
        for l in open(PW / "GeneOntology" / fn):
            if l.strip():
                go_split[norm(l.strip())] = split
    from Bio import SeqIO
    for fa in ["nrPDB-GO_sequences.fasta", "nrPDB-GO_test_sequences.fasta"]:
        for rec in SeqIO.parse(str(PW / "GeneOntology" / fa), "fasta"):
            go_seq.setdefault(norm(rec.id), str(rec.seq))
    for name, sub, split_map, seqs, tasks in [("ec_classification", "ECReaction", ec_split, {}, ["ec_classification"]),
                                              ("go", "GeneOntology", go_split, go_seq, ["go_mf", "go_bp"])]:
        if only and name not in only:
            continue
        pdir = OUT / name / "pdb"; pdir.mkdir(parents=True, exist_ok=True)
        files = sorted((PW / sub / "processed").glob("*.pt"))
        jobs = [(str(f), str(pdir / (f.stem + ".pdb")), seqs.get(f.stem, "")) for f in files]
        print(f"[{name}] converting {len(jobs)} processed graphs with {workers} workers", flush=True)
        rows, t0 = [], time.time()
        with ProcessPoolExecutor(workers) as ex:
            for k, r in enumerate(ex.map(_convert_pw, jobs, chunksize=64)):
                if r is None:
                    continue
                save_alignment(OUT / name / "alignment", r["id"], r.pop("_task_seq"), r.pop("_struct_seq"), r.pop("_mapping"))
                r.update(split=split_map.get(r["id"], "not_in_split_files"), source="RCSB chain via ProteinWorkshop archive",
                         source_id=r["id"], source_version=f"{sub}.tar.gz (ProteinWorkshop Zenodo, local copy dated 2026-04-21)",
                         task_seq_source="nrPDB-GO_sequences.fasta (SEQRES)" if seqs.get(r["id"]) else "structure-derived (dataset ships no sequence file)")
                rows.append(r)
                if (k + 1) % 5000 == 0:
                    print(f"   {k + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
        for task in tasks:
            cache, common = load_cache_ids(task), load_common_ids(task)
            trows = [dict(r, in_cache_union=int(r["id"] in cache), in_common_ids=int(r["id"] in common)) for r in rows]
            if task != name:
                tdir = OUT / task; tdir.mkdir(exist_ok=True)
                for link in ["pdb", "alignment"]:
                    if not (tdir / link).exists():
                        os.symlink(f"../{name}/{link}", tdir / link)
            write_manifest(OUT / task / "manifest.csv", trows)
            print(f"[{task}] manifest rows {len(trows)}, in cache union {sum(r['in_cache_union'] for r in trows)}", flush=True)


# ------------------------------------------------------------------ AlphaFold DB (DeepLoc)
def _http_get(session, url, tries=5):
    for k in range(tries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code in (200, 404):
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** k * 2)); continue
            return r
        except Exception:
            time.sleep(min(60, 2 ** k * 2))
    return None


def _fetch_afdb(args):
    acc, out_pdb, version = args
    import requests
    out_pdb = Path(out_pdb)
    if out_pdb.exists() and out_pdb.stat().st_size > 0:
        return acc, "exists", version, 1
    s = _fetch_afdb.session = getattr(_fetch_afdb, "session", None) or requests.Session()
    s.headers["User-Agent"] = UA
    base = acc.split("-")[0]
    if base != acc:
        return acc, "isoform accession; AFDB holds canonical sequences only", "", 0
    r = _http_get(s, f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{version}.pdb")
    nfrag, ver = 1, version
    if r is None:
        return acc, "network error after retries", "", 0
    if r.status_code == 404:
        a = _http_get(s, f"https://alphafold.ebi.ac.uk/api/prediction/{acc}")
        if a is None or a.status_code != 200 or not a.json():
            return acc, "no AFDB model for this accession", "", 0
        entries = a.json(); nfrag = len(entries)
        if nfrag > 1:
            return acc, f"AFDB model is split into {nfrag} fragments (protein longer than 2700 residues)", str(entries[0].get("latestVersion", "")), nfrag
        ver = entries[0].get("latestVersion", version)
        r = _http_get(s, entries[0]["pdbUrl"])
        if r is None or r.status_code != 200:
            return acc, "model listed by the API but file download failed", str(ver), nfrag
    elif r.status_code != 200:
        return acc, f"HTTP {r.status_code}", "", 0
    tmp = out_pdb.with_suffix(".part"); tmp.write_bytes(r.content); tmp.rename(out_pdb)
    time.sleep(0.05)
    return acc, "downloaded", ver, nfrag


def run_afdb(version, limit=None):
    import pandas as pd
    tr = pd.read_csv(ROOT / "structure_encoder_data" / "deeploc" / "train_partitions.csv")
    te = pd.read_csv(ROOT / "structure_encoder_data" / "deeploc" / "test.csv")
    recs = {}
    for _, r in tr.iterrows():
        recs.setdefault(str(r["ACC"]), dict(split=f"train_partition_{int(r['Partition'])}", seq=str(r["Sequence"])))
    n_dup = 0
    for _, r in te.iterrows():
        k = str(r["sid"])
        if k in recs:
            n_dup += 1; recs[k]["split"] += "+test_hpa"
        else:
            recs[k] = dict(split="test_hpa", seq=str(r["fasta"]))
    print(f"[deeploc] {len(recs)} unique accessions ({n_dup} appear in both the train file and the HPA test file)", flush=True)
    pdir = OUT / "deeploc" / "pdb"; pdir.mkdir(parents=True, exist_ok=True)
    accs = list(recs)[:limit] if limit else list(recs)
    fetch_log = {}
    with ThreadPoolExecutor(8) as ex:
        for k, (acc, msg, ver, nfrag) in enumerate(ex.map(_fetch_afdb, [(a, str(pdir / f"{a}.pdb"), version) for a in accs])):
            fetch_log[acc] = (msg, ver, nfrag)
            if (k + 1) % 2000 == 0:
                ok = sum(1 for m, _, _ in fetch_log.values() if m in ("downloaded", "exists"))
                print(f"   {k + 1}/{len(accs)} fetched, {ok} with a model", flush=True)
    json.dump(fetch_log, open(OUT / "deeploc" / "fetch_log.json", "w"))
    rows = []
    for acc in accs:
        msg, ver, nfrag = fetch_log[acc]; rec = recs[acc]; p = pdir / f"{acc}.pdb"
        row = dict(id=acc, split=rec["split"], source="AFDB", source_id=f"AF-{acc}-F1", source_version=f"model_v{ver}" if ver else "",
                   task_seq_source="DeepLoc 2.0 CSV", len_task_seq=len(rec["seq"]), n_chains=1, renumbered=0, n_nonstandard=0)
        if msg not in ("downloaded", "exists"):
            row.update(status="unusable", reason=msg, len_struct_seq=0, seq_identity="", task_coverage="", exact_match="", frac_full_backbone="", mean_plddt="", pdb_path="")
        else:
            res = [r for r in pdb_residues(open(p)) if not r[5]]
            sseq = "".join(AA3.get(r[3], "X") for r in res)
            ident, cov, mapping = align(rec["seq"], sseq)
            save_alignment(OUT / "deeploc" / "alignment", acc, rec["seq"], sseq, mapping)
            bb = np.mean([all(a in r[4] for a in ("N", "CA", "C", "O")) for r in res]) if res else 0.0
            pl = np.mean([r[4]["CA"][1] for r in res if "CA" in r[4]]) if res else float("nan")
            status, reason = default_status(ident, cov, bb, len(res))
            if status == "ok" and pl < 50:
                reason = "mean pLDDT below 50 (kept; filter downstream if wanted)"
            row.update(len_struct_seq=len(res), seq_identity=round(ident, 4), task_coverage=round(cov, 4), exact_match=int(mapping is None),
                       frac_full_backbone=round(float(bb), 4), mean_plddt=round(float(pl), 2), status=status, reason=reason, pdb_path=str(p.relative_to(OUT)))
        rows.append(row)
    for task in ["subcellular_localization", "membrane_soluble"]:
        cache, common = load_cache_ids(task), load_common_ids(task)
        tdir = OUT / task; tdir.mkdir(exist_ok=True)
        for link in ["pdb", "alignment"]:
            if not (tdir / link).exists():
                os.symlink(f"../deeploc/{link}", tdir / link)
        write_manifest(tdir / "manifest.csv", [dict(r, in_cache_union=int(r["id"] in cache), in_common_ids=int(r["id"] in common)) for r in rows])
    write_manifest(OUT / "deeploc" / "manifest.csv", rows)


# ------------------------------------------------------------------ SCOPe (TAPE remote homology)
def tape_remote_homology():
    import lmdb
    base = ROOT / "downstream_tasks" / "data" / "tape" / "remote_homology"
    recs = {}
    for split in ["train", "valid", "test_fold_holdout", "test_superfamily_holdout", "test_family_holdout"]:
        env = lmdb.open(str(base / f"remote_homology_{split}.lmdb"), readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            for i in range(pickle.loads(txn.get(b"num_examples"))):
                it = pickle.loads(txn.get(str(i).encode()))
                sid = it["id"].decode() if isinstance(it["id"], bytes) else str(it["id"])
                recs[sid] = dict(split=split, seq=it["primary"], fold_label=int(it["fold_label"]))
    return recs


def clean_scope_entry(text):
    """Keep protein ATOM records; turn common modified residues (HETATM) into their parent residue; drop other HETATM."""
    out, n_mod = [], 0
    for ln in text.splitlines(True):
        if ln.startswith("ATOM"):
            out.append(ln)
        elif ln.startswith("HETATM") and ln[17:20] in MODIFIED:
            name = ln[12:16]
            if ln[17:20] == "MSE" and name.strip() == "SE":
                name = " SD "; ln = ln[:76] + " S" + ln[78:]
            out.append("ATOM  " + ln[6:12] + name + ln[16:17] + MODIFIED[ln[17:20]] + ln[20:]); n_mod += 1
        elif ln.startswith(("REMARK  99", "TER", "END")):
            out.append(ln)
    return "".join(out), n_mod


def run_scope(archives):
    recs = tape_remote_homology(); want = set(recs)
    rdir = OUT / "remote_homology"; pdir = rdir / "pdb"; pdir.mkdir(parents=True, exist_ok=True)
    found = {p.stem: "existing" for p in pdir.glob("*.pdb")}
    prev = rdir / "found_in_archive.json"
    if prev.exists():
        found.update({k: v for k, v in json.load(open(prev)).items() if k in found})
    raw = rdir / "source" / "extracted"                      # loose .ent files extracted with GNU tar (fast path)
    for ent in (raw.rglob("*.ent") if raw.exists() else []):
        sid = ent.stem
        if sid in want and found.get(sid) in (None, "existing"):
            if sid not in found:
                text, _ = clean_scope_entry(ent.read_text(encoding="latin-1")); (pdir / f"{sid}.pdb").write_text(text)
            found[sid] = "pdbstyle-2.08 full archive, " + ent.relative_to(raw).parts[0]
    for arc in archives:
        arc = Path(arc)
        if not arc.exists():
            print(f"[scope] archive missing: {arc}", flush=True); continue
        print(f"[scope] scanning {arc.name}; still need {len(want - set(found))}", flush=True)
        with tarfile.open(arc, "r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                sid = Path(m.name).stem
                if sid in want and sid not in found:
                    text, n_mod = clean_scope_entry(tf.extractfile(m).read().decode("latin-1"))
                    (pdir / f"{sid}.pdb").write_text(text); found[sid] = arc.name
    json.dump(found, open(rdir / "found_in_archive.json", "w"))
    rows, missing = [], []
    for sid, rec in recs.items():
        p = pdir / f"{sid}.pdb"
        row = dict(id=sid, split=rec["split"], source="SCOPe PDB-style domain file", source_id=sid, source_version="SCOPe 2.08 bulk archive",
                   task_seq_source="TAPE remote_homology LMDB (primary)", len_task_seq=len(rec["seq"]), renumbered=0, mean_plddt="",
                   cache_id=f"b'{sid}'", fold_label=rec["fold_label"])
        if not p.exists():
            row.update(status="unusable", reason="sid not present in the SCOPe 2.08 archives scanned", len_struct_seq=0, pdb_path=""); missing.append(sid)
        else:
            res = pdb_residues(open(p)); sseq = "".join(AA3.get(r[3], "X") for r in res)
            ident, cov, mapping = align(rec["seq"], sseq)
            save_alignment(rdir / "alignment", sid, rec["seq"], sseq, mapping)
            bb = float(np.mean([all(a in r[4] for a in ("N", "CA", "C", "O")) for r in res])) if res else 0.0
            status, reason = default_status(ident, cov, bb, len(res))
            row.update(len_struct_seq=len(res), seq_identity=round(ident, 4), task_coverage=round(cov, 4), exact_match=int(mapping is None),
                       frac_full_backbone=round(bb, 4), n_nonstandard=sum(r[3] not in AA3 for r in res), n_chains=len({r[0] for r in res}),
                       status=status, reason=reason, pdb_path=str(p.relative_to(OUT)), source_version=f"SCOPe 2.08 ({found.get(sid, '')})")
        rows.append(row)
    cache, common = {c.strip("b'\"") for c in load_cache_ids("remote_homology")}, load_common_ids("remote_homology")
    write_manifest(rdir / "manifest.csv", [dict(r, in_cache_union=int(r["id"] in cache), in_common_ids=int(r["id"] in common)) for r in rows])
    with open(rdir / "missing_sids.fasta", "w") as f:
        for sid in missing:
            f.write(f">{sid} split={recs[sid]['split']}\n{recs[sid]['seq']}\n")
    print(f"[scope] {len(recs) - len(missing)} of {len(recs)} sids have a structure; {len(missing)} listed in missing_sids.fasta", flush=True)


# ------------------------------------------------------------------ ProteinGym wild-type structures
def run_proteingym():
    import pandas as pd
    import shutil
    pg = ROOT / "downstream_tasks" / "data" / "proteingym"
    mapping = json.load(open(pg / "structure_mapping.json"))
    sdir = pg / "ProteinGym_AF2_structures"; pdir = OUT / "mutation_effect" / "pdb"; pdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for assay, remote_path in sorted(mapping.items()):
        name = Path(remote_path).name if remote_path else ""
        src = sdir / name if name else None
        row = dict(id=assay, split="assay", source="ProteinGym AF2 wild-type structure", source_id=name, source_version="ProteinGym_AF2_structures.zip (local copy dated 2026-04-22)",
                   task_seq_source="wild type reconstructed from the DMS file", n_chains=1, renumbered=0, n_nonstandard=0)
        csvf = pg / "DMS_ProteinGym_substitutions" / f"{assay}.csv"
        wt = ""
        if csvf.exists():
            first = pd.read_csv(csvf, nrows=1).iloc[0]; s = list(str(first["mutated_sequence"]))
            for mut in str(first["mutant"]).split(":"):
                m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mut)
                if m and 0 < int(m.group(2)) <= len(s):
                    s[int(m.group(2)) - 1] = m.group(1)
            wt = "".join(s)
        row["len_task_seq"] = len(wt)
        if not src or not src.exists():
            row.update(status="unusable", reason="no wild-type structure file for this assay in the local ProteinGym folder", len_struct_seq=0, pdb_path="")
        else:
            dst = pdir / name
            if not dst.exists():
                shutil.copyfile(src, dst)
            res = [r for r in pdb_residues(open(dst)) if not r[5]]; sseq = "".join(AA3.get(r[3], "X") for r in res)
            ident, cov, mp = align(wt, sseq) if wt else (float("nan"), float("nan"), None)
            save_alignment(OUT / "mutation_effect" / "alignment", assay, wt, sseq, mp)
            bb = float(np.mean([all(a in r[4] for a in ("N", "CA", "C", "O")) for r in res])) if res else 0.0
            pl = float(np.mean([r[4]["CA"][1] for r in res if "CA" in r[4]])) if res else float("nan")
            status, reason = default_status(ident if wt else 1.0, cov if wt else 1.0, bb, len(res))
            if wt and cov < 0.999 and ident / max(cov, 1e-9) > 0.98:
                status, reason = "ok", "structure covers %.0f%% of the assayed sequence (ProteinGym structures can be sub-ranges)" % (100 * cov)
            row.update(len_struct_seq=len(res), seq_identity=round(ident, 4), task_coverage=round(cov, 4), exact_match=int(mp is None and bool(wt)),
                       frac_full_backbone=round(bb, 4), mean_plddt=round(pl, 2), status=status, reason=reason, pdb_path=str(dst.relative_to(OUT)))
        rows.append(row)
    write_manifest(OUT / "mutation_effect" / "manifest.csv", rows)
    print(f"[proteingym] {len(rows)} assays, {sum(r['status'] != 'unusable' for r in rows)} with a local wild-type structure, {len(set(r['source_id'] for r in rows if r['pdb_path']))} distinct files", flush=True)


# ------------------------------------------------------------------ coverage report
def run_coverage():
    import pandas as pd
    tasks = ["ec_classification", "go_mf", "go_bp", "subcellular_localization", "membrane_soluble", "remote_homology", "mutation_effect"]
    out = ["# Structure coverage", "", "Generated by `fetch_structures.py coverage`. One row per task; details follow.", "",
           "| Task | Source | ids | usable (ok) | usable with a flag (check) | unusable | exact sequence match | median identity | ids in the embedding-cache union | of those with a structure | common-ID set | of those with a structure |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    detail = []
    for t in tasks:
        f = OUT / t / "manifest.csv"
        if not f.exists():
            out.append(f"| {t} | manifest not written yet | | | | | | | | | | |"); continue
        m = pd.read_csv(f, low_memory=False)
        has = m.status.isin(["ok", "check"])
        cu = m.in_cache_union == 1 if "in_cache_union" in m else pd.Series(False, index=m.index)
        ci = m.in_common_ids == 1 if "in_common_ids" in m else pd.Series(False, index=m.index)
        ident = pd.to_numeric(m.seq_identity, errors="coerce")
        out.append(f"| {t} | {m.source.iloc[0]} | {len(m)} | {(m.status == 'ok').sum()} | {(m.status == 'check').sum()} | {(m.status == 'unusable').sum()} | "
                   f"{int(pd.to_numeric(m.exact_match, errors='coerce').fillna(0).sum())} | {ident[has].median():.3f} | {cu.sum()} | {(cu & has).sum()} | {ci.sum()} | {(ci & has).sum()} |")
        detail += ["", f"## {t}", "", f"Splits: {m.split.value_counts().to_dict()}", ""]
        rs = m.loc[m.status != "ok", "reason"].fillna("").str.replace(r"[0-9.]+", "#", regex=True).value_counts().head(8)
        detail += ["Reasons for `check` or `unusable` (numbers replaced by #):", ""] + [f"- {k}: {v}" for k, v in rs.items()]
        detail += ["", f"Identity to the task sequence among structures: min {ident[has].min():.3f}, 5th percentile {ident[has].quantile(0.05):.3f}, median {ident[has].median():.3f}; "
                       f"below 0.90: {(ident[has] < 0.90).sum()}; below 0.50: {(ident[has] < 0.50).sum()}."]
        cvg = pd.to_numeric(m.task_coverage, errors="coerce")
        detail += [f"Share of task-sequence residues that have coordinates: median {cvg[has].median():.3f}, 5th percentile {cvg[has].quantile(0.05):.3f}."]
        bb = pd.to_numeric(m.frac_full_backbone, errors="coerce")
        detail += [f"Residues with full backbone (N, CA, C, O): median {bb[has].median():.3f}; structures below 0.95: {(bb[has] < 0.95).sum()}."]
        pl = pd.to_numeric(m.mean_plddt, errors="coerce")
        if pl.notna().any():
            detail += [f"Mean pLDDT: median {pl.median():.1f}; below 70: {(pl < 70).sum()}; below 50: {(pl < 50).sum()}."]
        if "renumbered" in m and pd.to_numeric(m.renumbered, errors="coerce").fillna(0).sum() > 0:
            detail += [f"Renumbered files (original numbering in `<id>.numbering.tsv`): {int(pd.to_numeric(m.renumbered, errors='coerce').fillna(0).sum())}."]
    notes = OUT / "COVERAGE_NOTES.md"
    (OUT / "COVERAGE.md").write_text("\n".join(out + detail) + "\n\n" + (notes.read_text() if notes.exists() else ""))
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["proteinworkshop", "afdb", "scope", "proteingym", "restatus", "coverage"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--afdb-version", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--archives", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None, help="proteinworkshop: ec_classification and/or go")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "proteinworkshop":
        run_proteinworkshop(a.workers, a.only)
    elif a.step == "afdb":
        run_afdb(a.afdb_version, a.limit)
    elif a.step == "scope":
        run_scope(a.archives or sorted(str(p) for p in (OUT / "remote_homology" / "source").glob("*.tgz")))
    elif a.step == "proteingym":
        run_proteingym()
    elif a.step == "restatus":
        restatus()
    elif a.step == "coverage":
        run_coverage()


if __name__ == "__main__":
    sys.exit(main())
