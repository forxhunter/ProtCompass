# Structure-Aware Protein Embedding Benchmark - Reproduction Package

**Date**: 2026-04-21  
**Project**: Comprehensive evaluation of 35 protein encoders across 15 downstream tasks

This package contains everything needed to reproduce the downstream task evaluation pipeline.

---

## Overview

### Scope
- **35 encoders**: 8 sequence-only + 27 structure-aware (excluding ESM3)
- **15 downstream tasks**: 12 structure tasks + 3 sequence-only tasks
- **Total jobs**: 456 encoder×task combinations
- **Current progress**: 40/456 complete (9%)

### Task Categories

**Structure Tasks (12)** — all 35 encoders:
1. Secondary Structure (TAPE CB513)
2. Contact Prediction (TAPE ProteinNet)
3. Remote Homology (TAPE SCOPe)
4. Stability (TAPE)
5. EC Classification
6. GO Molecular Function
7. GO Biological Process
8. PPI Site Prediction
9. Metal Binding Site
10. Subcellular Localization
11. Membrane vs Soluble
12. Mutation Effect (ProteinGym)

**Sequence-Only Tasks (3)** — 8 sequence encoders only:
13. Fluorescence (GFP)
14. Solubility
15. PPI Affinity

---

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment from YAML
conda env create -f protein_embeddings_env.yml
conda activate protein_embeddings

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
```

### 2. Data Preparation

All data is now unified in a single directory:

```
$PROTCOMPASS_ROOT/structure_encoder_data/
├── tape/                                  # TAPE benchmark datasets
│   ├── secondary_structure/
│   ├── contact_prediction/
│   ├── remote_homology/
│   └── stability/
├── ppi/PFMBench/                          # PPI tasks
├── deeploc2/                              # Localization
├── ec_classification/                     # EC prediction
├── go_annotations/                        # GO terms
├── proteingym/ProteinGym_AF2_structures/  # Mutation effect structures
├── fluorescence/                          # GFP fluorescence
├── solubility/                            # Solubility prediction
└── processed/                             # Processed HDF5 files

$PROTCOMPASS_ROOT/downstream_tasks/embeddings/
└── {task}/{encoder}/                      # Output embeddings (37GB+)
```

**Note**: All raw datasets are consolidated in `structure_encoder_data/` for easier management.

### 3. Single-Machine Execution

```bash
# cd ~

# Run single encoder×task
python downstream_tasks/embed_proteins.py \
    --encoder esm2 \
    --task secondary_structure

# Run all tasks for one encoder
for task in secondary_structure contact_prediction stability; do
    python downstream_tasks/embed_proteins.py --encoder esm2 --task $task
done
```

### 4. Multi-Machine Execution (Recommended)

Due to disk space constraints and time requirements, we split work across 4 machines:

```bash
# Machine 1 (107 jobs)
python machine1_downstream.py > machine1.log 2>&1 &

# Machine 2 (95 jobs)
python machine2_downstream.py > machine2.log 2>&1 &

# Machine 3 (109 jobs)
python machine3_downstream.py > machine3.log 2>&1 &

# Machine 4 (105 jobs)
python machine4_downstream.py > machine4.log 2>&1 &
```

See `MULTI_MACHINE_GUIDE.md` for detailed instructions.

---

## File Structure

```
reproduction_package/
├── README.md                          # This file
├── protein_embeddings_env.yml         # Conda environment
├── PIPELINE_GUIDE.md                  # Detailed pipeline documentation
├── MULTI_MACHINE_GUIDE.md             # Multi-machine execution guide
├── DATA_DOWNLOAD.md                   # Data acquisition instructions
├── ENCODER_REGISTRY.md                # All 35 encoders with details
├── TASK_SPECIFICATIONS.md             # Task definitions and metrics
├── machine1_downstream.py             # Machine 1 script (107 jobs)
├── machine2_downstream.py             # Machine 2 script (95 jobs)
├── machine3_downstream.py             # Machine 3 script (109 jobs)
├── machine4_downstream.py             # Machine 4 script (105 jobs)
├── check_progress.sh                  # Progress monitoring script
└── TROUBLESHOOTING.md                 # Common issues and solutions
```

---

## Key Scripts

### Core Pipeline Scripts

1. **`downstream_tasks/embed_proteins.py`**
   - Main embedding generation script
   - Handles all 35 encoders and 15 tasks
   - Auto-detects sequence vs structure encoders
   - Saves embeddings + metadata to `embeddings/{task}/{encoder}/`

2. **`downstream_tasks/evaluate_tasks.py`**
   - Evaluation script (run after embeddings complete)
   - Ridge regression for regression tasks
   - Logistic regression for classification
   - Multi-seed evaluation (default: 5 seeds)

3. **`downstream_tasks/config.py`**
   - Task definitions and encoder registry
   - Data paths and hyperparameters

### Monitoring Scripts

- **`check_progress.sh`**: Real-time progress across all tasks
- **`machine{1-4}_downstream.py`**: Self-contained execution with progress reporting

---

## Expected Runtime

**Per encoder×task** (approximate):
- Sequence encoders: 5-30 min (depends on model size)
- Structure encoders: 10-120 min (depends on complexity)
- Large datasets (mutation_effect): up to 2h per encoder

**Total estimated time**:
- Single machine: ~200-300 hours
- 4 machines (parallel): ~50-75 hours

---

## Output Format

Each completed job produces:

```
downstream_tasks/embeddings/{task}/{encoder}/
├── meta.json                    # Metadata (encoder, task, dataset size, embedding dim)
├── train_embeddings.npy         # Training embeddings [N_train, D]
├── train_labels.npy             # Training labels
├── train_ids.npy                # Training IDs
├── test_embeddings.npy          # Test embeddings [N_test, D]
├── test_labels.npy              # Test labels
└── test_ids.npy                 # Test IDs
```

**Verification**: Check `meta.json` exists to confirm completion.

---

## Troubleshooting

### Disk Space Issues
- **Problem**: /mnt/f at 100% capacity
- **Solution**: Use multi-machine split or mount additional storage

### CUDA Out of Memory
- **Problem**: GPU memory exhausted during encoding
- **Solution**: Reduce batch size in encoder config or use CPU fallback

### Missing Dependencies
- **Problem**: Import errors for specific encoders
- **Solution**: Check `TROUBLESHOOTING.md` for encoder-specific requirements

### Tokenizer Truncation (ESM2 variants)
- **Known issue**: ESM2 models truncate at 1024 tokens
- **Impact**: 11.4% of mutation_effect sequences dropped
- **Status**: Fixed in `src/models/external.py` (defensive copy_len)

---

## Citation

If you use this benchmark or reproduction package, please cite:

```bibtex
@article{yourpaper2026,
  title={Structure-Aware Protein Embedding Benchmark},
  author={Your Name},
  journal={TBD},
  year={2026}
}
```

---

## Support

- **Issues**: Check `TROUBLESHOOTING.md` first
- **Questions**: See `PIPELINE_GUIDE.md` for detailed documentation
- **Data**: See `DATA_DOWNLOAD.md` for dataset acquisition

---

## Changelog

- **2026-04-21**: Initial reproduction package created
  - 40/456 jobs complete
  - Multi-machine split implemented
  - ESM2 truncation bug fixed
  - All 15 tasks validated
