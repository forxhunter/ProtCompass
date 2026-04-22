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

- **`structure_encoder_data/` & `esm3-sm-open-v1/`**
  Required structural data samples, generated embeddings, and caches for executing the individual evaluations (like ESM-3).

---
