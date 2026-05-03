#!/usr/bin/env bash
#SBATCH --job-name=structembed
#SBATCH --output=logs/run_%j.out
#SBATCH --error=logs/run_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # CPU cores for probe training (sklearn uses them)
#SBATCH --mem=128G                   # ProstT5 needs ~60 GB; ESM-3 ~30 GB — budget 128 GB
#SBATCH --gres=gpu:1                 # 1× Blackwell Pro 6000 (96 GB VRAM)
#SBATCH --time=72:00:00             # worst case ~3 days; checkpoint skipping means safe to resubmit
#SBATCH --partition=gpu             # ← change to your cluster's partition name

# ══════════════════════════════════════════════════════════════════════════════
# InvariantsBench — Downstream Task Benchmark
# Single-GPU sequential run (designed for 1× NVIDIA Blackwell Pro 6000, 96 GB)
#
# WHAT THIS JOB DOES:
#   Step 1 (GPU)  — Extract frozen embeddings for every encoder × every task.
#                   Each encoder is loaded once, run on all tasks, then unloaded.
#                   Embeddings are saved to downstream_tasks/embeddings/ and
#                   reused in Step 2 — they are never re-extracted.
#
#   Step 2 (CPU)  — Train linear + MLP-2 probes, 5 random seeds per
#                   (encoder, task) pair.  All probe training is CPU-only
#                   (scikit-learn); the GPU is idle during this step.
#                   Results → downstream_tasks/results/{encoder}/{task}/
#
#   Step 3 (CPU)  — Compute Spearman ρ between I-family probe scores
#                   (from outputs/sweep_v4_full/) and downstream task scores.
#                   Outputs heatmaps + top-correlation report to
#                   downstream_tasks/results/
#
# RESUME SAFETY:
#   Each embedding and each probe seed is saved immediately after completion.
#   If the job is killed, resubmit — it will skip already-done work.
#
# ENCODER ORDER:
#   Structure/random-init encoders first (fast, no weights needed).
#   Large PLMs last (slow embedding, large VRAM).
#   ProstT5 is the largest (~22 GB weights, ~60 GB peak VRAM) — scheduled last.
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

CONDA_ENV="structembed_downstream"
REPO_ROOT="${SLURM_SUBMIT_DIR}"

# ── Activate environment ───────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

echo "=============================================="
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : $(hostname)"
echo "GPU      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "CPUs     : ${SLURM_CPUS_PER_TASK}"
echo "Start    : $(date)"
echo "=============================================="

# Point to local weight caches — no internet needed
export HF_HOME="${REPO_ROOT}/weights/huggingface"
export TRANSFORMERS_CACHE="${REPO_ROOT}/weights/huggingface/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="${REPO_ROOT}/weights/torch_hub"

# Encoder order: fast structure encoders → medium PLMs → large PLMs → ProstT5 last
STRUCTURE_ENCODERS=(
    schnet dimenet egnn gvp gcpnet cdconv mace
    se3_transformer tfn ipa gearnet
    foldseek geobpe random_token
)
PLM_ENCODERS=(
    proteinbert_bfd          # 420M  ~3 GB
    esm2_35m                 # 35M   ~0.1 GB
    esm2_150m                # 150M  ~0.6 GB
    ankh                     # 700M  ~5.5 GB
    saprot_pretrained        # 650M  ~4.9 GB
    esm1b                    # 650M  ~4.9 GB
    esm2                     # 650M  ~2.5 GB   (esm2 = ESM-2-650M in registry)
    proteinmpnn_pretrained   # 6 MB  (structure-based)
    gearnet_pretrained       # 77 MB (structure-based)
    esm_if                   # 142M  ~1.6 GB
    esm3                     # 1.4B  ~30 GB
    prottrans                # 3B    ~4.6 GB   (half-precision enc only)
    prostt5                  # 3B    ~22 GB    — run last, largest VRAM
)
ALL_ENCODERS=("${STRUCTURE_ENCODERS[@]}" "${PLM_ENCODERS[@]}")

N_ENCODERS=${#ALL_ENCODERS[@]}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: EMBEDDING EXTRACTION  (GPU-bound, ~1–40 min per encoder depending on size)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════"
echo "STEP 1: Embedding extraction (${N_ENCODERS} encoders)"
echo "══════════════════════════════════════════════"

for enc in "${ALL_ENCODERS[@]}"; do
    echo ""
    echo "  [$(date +%H:%M:%S)] Embedding: ${enc}"
    PYTHONPATH=. python downstream_tasks/embed_proteins.py \
        --encoder "${enc}" \
        --task all \
        || echo "  [WARN] ${enc} embedding failed — check logs and resubmit"
done

echo ""
echo "  [$(date +%H:%M:%S)] Step 1 done."

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: PROBE TRAINING  (CPU-bound, ~30 min–4 h per encoder)
# GPU is idle here; all ${SLURM_CPUS_PER_TASK} CPU cores used by sklearn.
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════"
echo "STEP 2: Probe training — 5 seeds × all tasks (${N_ENCODERS} encoders)"
echo "══════════════════════════════════════════════"

for enc in "${ALL_ENCODERS[@]}"; do
    # Skip if no embeddings were produced (e.g. encoder failed in Step 1)
    EMB_COUNT=$(find downstream_tasks/embeddings -maxdepth 3 -name "meta.json" \
                -path "*/${enc}/*" 2>/dev/null | wc -l)
    if [ "${EMB_COUNT}" -eq 0 ]; then
        echo "  [SKIP] ${enc} — no embeddings found"
        continue
    fi

    echo ""
    echo "  [$(date +%H:%M:%S)] Probing: ${enc}  (${EMB_COUNT} tasks)"
    PYTHONPATH=. python downstream_tasks/evaluate_tasks.py \
        --encoder "${enc}" \
        --task all \
        --seeds 0 1 2 3 4 \
        || echo "  [WARN] ${enc} probing failed — partial results saved"
done

echo ""
echo "  [$(date +%H:%M:%S)] Step 2 done."

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: I-FAMILY × TASK CORRELATION  (CPU, ~5 min)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════"
echo "STEP 3: I-family × downstream task correlation"
echo "══════════════════════════════════════════════"

PYTHONPATH=. python downstream_tasks/analyze_ifamily_correlation.py \
    --n-bootstrap 2000 \
    --min-rho 0.4

echo ""
echo "=============================================="
echo "All done: $(date)"
echo "Results → downstream_tasks/results/"
echo "  ifamily_task_correlation.json"
echo "  heatmap_{all,pretrained,random_init}.png"
echo "  top_correlations.txt"
echo "=============================================="
