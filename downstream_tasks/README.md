# Downstream Task Benchmark

**Goal:** Measure how well each protein encoder's structural invariants predict real biological task performance — and which invariant families (I1–I8) gateway to which application domains.

---

## Motivation

InvariantsBench characterizes encoders by how much of each structural invariant family (local geometry, packing, topology, etc.) survives in the embedding. But *does capturing those invariants actually help with biology?*

This benchmark answers that question with a second-order correlation: across 30 encoders, does an encoder's I-family score predict its performance on a downstream biological task? A high Spearman ρ between I4 (topology) and GO function prediction, for example, would mean that encoders which preserve contact order tend to be better functional annotators — giving an interpretable mechanistic link from structure representation to biological utility.

---

## Task Selection

Twelve tasks spanning five biological application domains, drawn from three established benchmarks:

### Structure & fold (TAPE, NeurIPS 2019)

| ID | Task | Metric | Dataset | Source |
|----|------|--------|---------|--------|
| T1 | Secondary structure | Q3 accuracy | CB513 (513 test) | TAPE |
| T2 | Contact prediction | top L/5 | ProteinNet (40 test) | TAPE |
| T3 | Remote homology | Fold accuracy | SCOP fold holdout (718 test) | TAPE |

### Fitness & stability (TAPE)

| ID | Task | Metric | Dataset | Source |
|----|------|--------|---------|--------|
| T7 | Fluorescence | Spearman ρ | GFP variants (27K test) | TAPE |
| T8 | Stability | Spearman ρ | Rocklin designed proteins (12K test) | TAPE |

*Structure encoders skip T7/T8: these datasets have no PDB coordinates (sequence variants only).*

### Function annotation (ProteinWorkshop, ICLR 2024)

| ID | Task | Metric | Dataset | Source |
|----|------|--------|---------|--------|
| T4 | EC reaction class | Accuracy | 384 classes, 5.6K test | ProteinWorkshop |
| T5 | GO molecular function | $F_{max}$ | 489 terms, 3.4K test | ProteinWorkshop |
| T6 | GO biological process | $F_{max}$ | 1943 terms, 3.4K test | ProteinWorkshop |

### Interaction & binding (ProteinWorkshop)

| ID | Task | Metric | Dataset | Source |
|----|------|--------|---------|--------|
| T9 | PPI interface site | AUPRC | MaSIF-site (359 test) | ProteinWorkshop |
| T10 | Metal binding site | AUPRC | Metal3D (59 test) | ProteinWorkshop |

### Localization (DeepLoc 2.0, NAR 2022)

| ID | Task | Metric | Dataset | Source |
|----|------|--------|---------|--------|
| T11 | Subcellular localization | Accuracy | 10 classes, 5.5K test | DeepLoc 2.0 |
| T12 | Membrane vs. soluble | AUROC | Binary, 5.5K test | DeepLoc 2.0 |

---

## Encoders

30 encoders across two regimes:

**Random-init structure encoders (15):** SchNet, DimeNet, EGNN, GVP, GCPNet, CDConv, MACE, SE3-Transformer, TFN, IPA, DualStream, GearNet, Foldseek, GeoBPE, RandomToken

**Pretrained PLMs (15):** ESM-2-35M, ESM-2-150M, ESM-2-650M, ESM-1b, ESM-IF, ESM-3, ProteinMPNN-pre, GearNet-pre, ProstT5, ProtTrans, SaProt-650M, Ankh-base, ProteinBERT-BFD

---

## Pipeline

### Step 1 — Embedding extraction (GPU)

For each encoder, every protein in every task's train and test split is passed through the frozen encoder. Per-residue or global embeddings are saved to disk as NumPy arrays.

```
embeddings/{task}/{encoder}/
    train_embeddings.npy    [N_train, D]  or ragged object array for residue-level
    train_labels.npy
    test_embeddings.npy     [N_test, D]
    test_labels.npy
    meta.json               {D, n_train, n_test, elapsed_sec, ...}
```

This step runs **once per encoder**. Embeddings are deterministic — no seed needed.

**Residue-level tasks** (T1 secondary structure, T2 contact, T9 PPI site, T10 metal binding) store per-protein arrays as ragged numpy object arrays. **Contact prediction** additionally saves CA coordinates from TAPE's tertiary field so that precision@L/5 can be computed properly.

### Step 2 — Probe training (CPU, 5 seeds)

On the saved embeddings, two probes are trained per task per encoder:

- **Linear:** SGDClassifier (clf tasks) or Ridge (regression / multilabel). Stochastic gradient, fast for any number of classes.
- **MLP-2:** Two hidden layers of size `max(128, min(512, D//2))`, early stopping at 15 patience.

Five seeds control the train/val split shuffle and weight initialization. Per-seed results are saved as `results/{encoder}/{task}/seed_{0-4}.json` and aggregated to `aggregated.json` with mean ± std. `best_mean` (across linear and MLP-2) is the primary score used in Step 3.

**Caps applied for speed:**
- Residue-level train: max 80,000 residues (randomly sampled across proteins)
- Protein-level train: max 20,000 proteins
- Contact prediction pair features: max 200,000 pairs (early-stop sampling)

### Step 3 — I-family × task correlation

For each (I-family, task) pair across all 30 encoders:

1. Load the encoder's MLP-2 composite I-family score from `outputs/sweep_v4_full/{encoder}/aggregated_results.json`
2. Load the encoder's best downstream score from `results/{encoder}/{task}/aggregated.json`
3. Compute Spearman ρ with 2000-sample bootstrap 95% CI

Three matrices are produced: all encoders, pretrained only, random-init only.

**I5 (amino acid identity) is excluded** from primary metrics — sequence leakage.

---

## How to run

### On a single machine

```bash
# From repo root, with env activated:
bash downstream_tasks/run_pipeline.sh

# Or step by step:
PYTHONPATH=. python downstream_tasks/embed_proteins.py --encoder all --task all
PYTHONPATH=. python downstream_tasks/evaluate_tasks.py --encoder all --task all --seeds 0 1 2 3 4
PYTHONPATH=. python downstream_tasks/analyze_ifamily_correlation.py --n-bootstrap 2000
```

### On a SLURM cluster (single GPU)

See [CLUSTER_SETUP.md](CLUSTER_SETUP.md) for the full walkthrough. Short version:

```bash
# Local machine: transfer everything (~85 GB, no internet needed on cluster after)
bash downstream_tasks/transfer.sh user@cluster.edu:/scratch/user/structembed

# Cluster: create env + submit single job
conda env create -f downstream_tasks/environment.yml
conda activate structembed_downstream
sbatch downstream_tasks/slurm_run.sh
```

The SLURM job runs all three steps sequentially in one submission. If killed, resubmit — skip logic means no work is repeated.

---

## Outputs

```
downstream_tasks/results/
├── {encoder}/
│   └── {task}/
│       ├── seed_0.json … seed_4.json     per-seed probe scores
│       └── aggregated.json               mean ± std + best_mean
├── ifamily_task_correlation.json         full Spearman ρ matrix + CI + p-values
├── heatmap_all.png                       7 × 12 heatmap, all 30 encoders
├── heatmap_pretrained.png                pretrained PLMs only
├── heatmap_random_init.png               random-init structure encoders only
└── top_correlations.txt                  pairs with |ρ| ≥ 0.4, ranked
```

### Reading the heatmap

Rows are I-families (I1–I8, excluding I5). Columns are the 12 downstream tasks grouped by domain. Cell values are Spearman ρ across 30 encoders between I-family score and downstream task `best_mean`. Significance stars: * p<0.05, ** p<0.01, *** p<0.001.

A strong positive ρ in cell (I4, GO-MF) means: encoders that preserve contact order tend to be better GO molecular function predictors. A near-zero ρ means that invariant has no predictive value for that task across the encoder space.

---

## File structure

```
downstream_tasks/
├── config.py                   encoder list, task definitions, I-family mappings
├── download_data.py            download TAPE / DeepLoc / ProteinWorkshop (run locally)
├── embed_proteins.py           Step 1: extract embeddings (GPU)
├── evaluate_tasks.py           Step 2: train probes, 5 seeds (CPU)
├── analyze_ifamily_correlation.py  Step 3: Spearman ρ matrix + heatmaps
├── tasks/
│   ├── __init__.py
│   └── load_proteinworkshop.py  loaders for EC, GO, MaSIF-site, Metal3D
├── data/                       benchmark datasets (17 GB after download)
├── embeddings/                 cached embeddings (generated by Step 1)
├── results/                    probe results (generated by Steps 2–3)
├── environment.yml             conda environment
├── transfer.sh                 rsync script for cluster transfer
├── slurm_run.sh                single-GPU SLURM job (all 3 steps)
└── CLUSTER_SETUP.md            step-by-step cluster instructions
```

---

## Implementation notes

**Contact prediction (T2).** TAPE's ProteinNet stores backbone CA coordinates as tertiary fields. Embeddings are saved as per-residue arrays; CA coordinates are saved alongside as `_labels_0.npy`. At probe training time, long-range pairs (|i−j| ≥ 6) are constructed on the fly, with pair feature `|emb_i − emb_j|` (symmetric, equivariant to permutation). Up to 200K pairs are sampled at random for training; precision@L/5 is evaluated per test protein.

**GO multilabel (T5/T6).** Linear probe uses multi-output Ridge regression (closed-form, much faster than 1943 separate binary classifiers). Continuous Ridge scores are used directly in the Fmax threshold sweep.

**Large PLM memory.** ProstT5 (3B, ~22 GB weights) and ESM-3 (1.4B, ~30 GB VRAM peak) are scheduled last in the embedding loop so that earlier encoders are not blocked. With 96 GB VRAM on the Blackwell Pro 6000, all models fit.

**Offline weight loading.** `embed_proteins.py` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` at runtime, pointing `HF_HOME` and `TORCH_HOME` to `weights/` inside the repo. ProteinMPNN and GearNet call `load_pretrained()` from `checkpoints/`. ESM-3 reads from `esm3-sm-open-v1/data/weights/`. No internet connection is needed on the cluster.
