# <img src="figures/icon.png" alt="ProtEncEval Icon" width="100" align="left" style="margin-right: 10px" /> ProtEncEval: What Protein Structure Encoders Know, What Biological Tasks Require, and How to Choose the Right Model

<p align="center">
  <img src="figures/main.png" alt="ProtEncEval Framework" width="800">
</p>

## Overview

ProtEncEval is a comprehensive benchmark designed to evaluate **30 diverse protein encoders** across **19 probing tasks** and **9 downstream biological tasks**. Existing benchmarks emphasize aggregate downstream performance, leaving encoder representations opaque. ProtEncEval addresses this by introducing an *invariant-family framework* that quantifies the structural and biochemical information encapsulated in embeddings and aligns them with downstream biological requirements. 

Our benchmark provides:
1. **Systematic Information Profiling:** Measures encoder preservation of geometry, packing, topology, flexibility, compactness, chemistry, and sequence-structure consistency.
2. **The Information Alignment Hypothesis:** Demonstrates that downstream task performance is governed by the alignment between an encoder's information footprint and task demands.
3. **ESP & ESP++ Selectors:** Predictors that estimate an encoder's downstream utility based on probing profiles, enabling principled model selection prior to executing tasks.

## Repository Structure

The current codebase is organized into key modules to facilitate experimentation, replication, and exploration:

- **`src/`**
  Core logic and implementations for the ProtEncEval framework.
  - `/models/`: Interfaces and wrappers for the 30 tested structure- and sequence-based protein encoders.
  - `/probes/`: Implementation of linear and complex probes for the 19 biological invariant families.
  - `/tasks/`: Execution and validation protocols.
  - `/analysis/` & `/interpretation/`: Logic for the ESP and ESP++ encoder selection predictors.
  - `/visualization/`: Scripts for plotting and generating figures.

- **`downstream_tasks/`**
  Contains the dataset processing, definitions, and inference scripts for the 9 selected biological downstream tasks (e.g., function annotation, fitness prediction).

- **`reproduction_package/`**
  Contains the necessary scripts, configurations, and instructions dedicated to accurately replicating the experimental results and extracting the model invariant profiles from the paper.

- **`figures/`**
  Contains the main paper graphics, including the `main.png` framework overview.

- **`external_repos/`**
  Third-party encoder implementations. See [EXTERNAL_REPOS.md](EXTERNAL_REPOS.md) for installation instructions and links to original repositories.

**Note:** Large datasets and pre-computed embeddings are hosted on HuggingFace (see Data Download section below).

---

## Data Download

This repository contains only the source code. Large datasets and pre-computed embeddings are hosted on HuggingFace:

### Datasets (ProtEnv)
Download raw datasets and predicted structures from HuggingFace:

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Download datasets (117GB)
huggingface-cli download Anonymoususer2223/ProtEnv \
    --repo-type dataset \
    --local-dir structure_encoder_data
```

**Contents:**
- Raw sequences, labels, and LMDB files for all tasks
- PDB structure cache (46GB)
- TAPE benchmark datasets
- ProteinWorkshop tasks
- ProteinGym mutation data
- And more (see [ProtEnv README](https://huggingface.co/datasets/Anonymoususer2223/ProtEnv))

### Pre-computed Embeddings (ProtCompass_Embeddings)
Download pre-computed embeddings and evaluation results:

```bash
# Download embeddings and results (~150GB compressed)
huggingface-cli download Anonymoususer2223/ProtCompass_Embeddings \
    --repo-type dataset \
    --local-dir .
```

**Contents:**
- Pre-computed embeddings from 70+ encoders across 13 tasks (compressed)
- Probing task embeddings and results
- Evaluation results for all encoder-task combinations
- Analysis outputs and paper figures

**Decompression:**
```bash
# Decompress all tasks
cd embeddings/
for f in *.tar.gz; do tar -xzf "$f"; done

# Or decompress specific task
tar -xzf secondary_structure.tar.gz

# Or decompress specific encoder (for large tasks)
cd contact_prediction/
tar -xzf esm2.tar.gz
```

### External Encoder Repositories
Clone third-party encoder implementations:

```bash
# See EXTERNAL_REPOS.md for detailed instructions
cd external_repos
# Follow installation instructions in EXTERNAL_REPOS.md
```

---
