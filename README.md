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

### Checkpoints
General Pretrain checkpoint — Broad simulated pretraining
Human Metabolism checkpoint — Fine-tuned for human metabolites
Natural Product checkpoint — Adapted for complex natural products (>800 Da)
Organic Chemistry checkpoint — General-purpose for synthetic/reaction products
