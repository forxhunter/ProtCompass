#!/usr/bin/env bash
# transfer.sh — ship the full project (code + data + weights) to the cluster.
# After this runs, the cluster job needs NO internet access whatsoever.
#
# Usage (from your local repo root):
#   bash downstream_tasks/transfer.sh user@cluster.edu:/scratch/user/structembed
#
# Total size ≈ 80–90 GB:
#   17 GB  downstream_tasks/data/   (TAPE, ProteinWorkshop, DeepLoc)
#   47 GB  HuggingFace weight cache  (ESM-2, ESM-1b, ProtTrans, ProstT5, Ankh, SaProt, ProteinBERT)
#    5 GB  esm3-sm-open-v1/          (ESM-3 weights)
#    1.6GB torch hub cache            (ESM-IF)
#   84 MB  checkpoints/              (ProteinMPNN, GearNet)
#   ~1 GB  src/, configs/, Geo_BPE/  (code + GeoBPE ckpts)
#   14 MB  outputs/sweep_v4_full/    (I-family scores, needed for final correlation step)

set -euo pipefail

REMOTE="${1:?Usage: $0 user@host:/path/to/dest}"
REPO="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

echo "=========================================="
echo "Transferring project to: ${REMOTE}"
echo "Source root:             ${REPO}"
echo "=========================================="

rsync_to() {
    local src="$1"; local dest="$2"; shift 2
    echo ""
    echo "--- $src → $dest ---"
    rsync -avzP --mkpath "$@" "${src}" "${REMOTE}/${dest}"
}

# ── Code ────────────────────────────────────────────────────────────────────
rsync_to "${REPO}/src/"           "src/" \
  --exclude='**/__pycache__' --exclude='**/*.pyc'

rsync_to "${REPO}/configs/"       "configs/" \
  --exclude='**/__pycache__'

# ── Downstream tasks (code + 17GB data, NOT embeddings/results) ─────────────
rsync_to "${REPO}/downstream_tasks/" "downstream_tasks/" \
  --exclude='**/__pycache__' --exclude='**/*.pyc' \
  --exclude='downstream_tasks/embeddings/' \
  --exclude='downstream_tasks/results/'

# ── I-family probe scores (needed by analyze_ifamily_correlation.py) ─────────
rsync_to "${REPO}/outputs/sweep_v4_full/" "outputs/sweep_v4_full/"

# ── Project-local checkpoints (ProteinMPNN 6MB, GearNet 77MB) ───────────────
rsync_to "${REPO}/checkpoints/" "checkpoints/"

# ── ESM-3 local weights (5.2 GB) ────────────────────────────────────────────
rsync_to "${REPO}/esm3-sm-open-v1/" "esm3-sm-open-v1/"

# ── GeoBPE checkpoint ───────────────────────────────────────────────────────
rsync_to "${REPO}/Geo_BPE/" "Geo_BPE/" \
  --exclude='**/__pycache__' --exclude='**/*.pyc' \
  --exclude='Geo_BPE/**/*.lmdb' --exclude='Geo_BPE/**/data/'

# ── HuggingFace weight cache → weights/huggingface/ ─────────────────────────
# embed_proteins.py sets HF_HOME=ROOT/weights/huggingface at runtime.
HF_CACHE="${HOME}/.cache/huggingface"
if [ -d "${HF_CACHE}/hub" ]; then
    echo ""
    echo "--- HuggingFace hub cache (${HF_CACHE}/hub) ---"
    rsync -avzP --mkpath \
      --exclude='*.lock' \
      --exclude='tmp_*' \
      "${HF_CACHE}/hub/" \
      "${REMOTE}/weights/huggingface/hub/"
else
    echo "WARNING: ${HF_CACHE}/hub not found — HuggingFace weights will not be transferred."
fi

# ── PyTorch hub cache (ESM-IF, 1.6 GB) → weights/torch_hub/ ────────────────
TH_CACHE="${HOME}/.cache/torch/hub"
if [ -d "${TH_CACHE}" ]; then
    echo ""
    echo "--- PyTorch hub cache (${TH_CACHE}) ---"
    rsync -avzP --mkpath \
      "${TH_CACHE}/" \
      "${REMOTE}/weights/torch_hub/hub/"
else
    echo "WARNING: ${TH_CACHE} not found — ESM-IF weights will not be transferred."
fi

echo ""
echo "=========================================="
echo "Transfer complete."
echo ""
echo "Next steps on the cluster:"
echo "  1. cd /scratch/user/structembed"
echo "  2. conda env create -f downstream_tasks/environment.yml"
echo "  3. conda activate structembed_downstream"
echo "  4. sbatch downstream_tasks/slurm_run.sh"
echo "=========================================="
