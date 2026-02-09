<p align="center">
  <img src="https://img.shields.io/badge/MACROS-Multi-Agent Closed-Loop Reasoning for Organic Structure-6A1B9A?style=for-the-badge&logo=chemistry&logoColor=white" alt="MACROS Badge">
</p>

<h1 align="center">MACROS</h1>

<p align="center">
  <strong>Multi-Agent Closed-Loop Reasoning for Organic Structure Elucidation</strong><br>
  <em>Autonomous structure determination from routine multimodal spectra: ¹H NMR, ¹³C NMR, HSQC, and IR</em>
</p>

<p align="center">
  <a href="https://github.com/streamflowmaster/MACROS/stargazers">
    <img src="https://img.shields.io/github/stars/streamflowmaster/MACROS?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/streamflowmaster/MACROS/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square&logo=apache" alt="License">
  </a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-orange?style=flat-square" alt="Status">
</p>

<p align="center">
  <img src="https://via.placeholder.com/1200x500/1a0033/00ffaa?text=MACROS+-+Agentic+Reasoning+for+Spectroscopic+Structure+Elucidation" alt="MACROS Banner">
  <!-- Recommended: replace with a generated image showing multi-agent loop + spectra + molecule -->
</p>

## Overview

MACROS (**Multi-Agent Closed-Loop Reasoning for Organic Structure**) is a purpose-built multi-agent system designed for **automated structure elucidation** of organic molecules from routinely acquired multimodal spectroscopic data: **¹H NMR**, **¹³C NMR**, **HSQC**, and **IR**.

Unlike approaches that adapt general foundation models, MACROS is engineered from the ground up with modality-specific components and an **explicit multi-agent closed-loop reasoning mechanism** that emulates the iterative, expert human process for resolving complex spectra.

### Core Innovations

- **Modality-Specific Pretrained Agents** — Each agent is self-supervised on large-scale simulated data tailored to its spectral type.
- **Hierarchical Integration & Joint Training** — Agents are unified in a hierarchical framework and jointly trained on ~105 million simulated spectra–structure pairs.
- **Fine-Tuning on Real Data** — Final adaptation to experimental variability using unassigned real-world spectra.
- **Explicit Closed-Loop Reasoning** (core of MACROS):
  1. One agent **drafts candidate structures**.
  2. Another **simulates spectra** and ranks candidates using multi-criterion similarity.
  3. A third **refines** candidates over multiple inference cycles, using high-ranking outputs as guidance.
  → Enables **test-time scaling**: performance improves progressively during inference.
- **Emergent Chemical Intuition** — The system develops textbook-consistent behaviors, such as functional group–chemical shift relationships and preference for ring-first skeleton determination.
- **Interpretability** — Produces transparent reasoning trajectories with **atom-level reliability estimates**, supporting expert review and confident evaluation.

### Performance & Impact

MACROS demonstrates strong **zero-shot generalization** across diverse real-world datasets — including synthetic reaction products, complex natural products (>800 Da), and human metabolites — even when mostly pretrained on simulated data.

Fine-tuning on experimental spectra yields further gains, confirming robust adaptation to real-world variability.

In collaborative tests with experienced chemists, MACROS reduced structure elucidation time by ~**10×** while improving accuracy by at least **40%**, establishing it as a practical partner in research workflows.

The **modular, agent-based design** naturally supports incorporation of additional constraints (e.g., reaction substrates, molecular formulae from MS/MS), paving the way for extensions to 2D/3D NMR and other modalities.

## Key Features

- Multimodal input handling (¹H, ¹³C, HSQC, IR)
- Closed-loop iterative refinement during inference
- Interpretable outputs with atom-level confidence
- Strong generalization from simulation to experiment
- Modular architecture for future modality / constraint integration

## Project Structure (Current)

```text
MACROS/
├── agents/                  # Modality-specific agents (pretraining & inference)
├── closed_loop/             # Core multi-agent reasoning loop & refinement cycles
├── data/                    # Simulated & experimental data handling (docs only)
├── pretrain/                # Self-supervised pretraining pipelines per modality
├── joint_training/          # Hierarchical joint training on unified corpus
├── finetune/                # Adaptation scripts for real experimental spectra
├── simulation/              # Spectrum simulation & multi-criterion ranking
├── utils/                   # Evaluation, visualization, reliability estimation
└── examples/                # Inference demos & reasoning trajectory outputs
