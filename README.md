# Strategic-LLM-IPD

**Evaluating Strategic Intelligence in Large Language Models via Evolutionary Iterated Prisoner’s Dilemma**

This repository implements the full experimental framework, agent implementations, and analysis pipeline described in our study *“Do LLMs Possess Strategic Intelligence? Testing LLMs in Iterated Prisoner’s Dilemmas.”*

---

## Overview

- Implements a **multi-phase evolutionary tournament** among a diverse population of agents:  
  - **12 LLM-based agents** (from major providers)  
  - **13 canonical & synthetic rule-based strategies** (Tit-for-Tat, Grim Trigger, Prober, etc.)  
  - **3 adaptive learning agents** (Q-Learning, Thompson Sampling, Gradient Meta-Learner)  
- Supports varying conditions:  
  - **Shadow-of-the-future (termination probability)**: δ ∈ {0.02, 0.05, 0.10, 0.25, 0.75}  
  - **Memory regimes**: Anonymous Memory vs. Opponent Tracking  
  - **LLM temperature settings** (where applicable)  
- Provides automated logging of: moves, payoffs, per-round histories, and LLM rationales  
- Includes an analysis module computing: cooperation rates, strategic fingerprints, extended history behaviour, and rationales categorisation  

---

## Getting Started

### Prerequisites

- Python ≥ 3.9  
- Required packages (see `requirements.txt`): e.g., `numpy`, `pandas`, `matplotlib`, `seaborn`, etc.  
- Access credentials / API keys for LLM providers (if enabling LLM agents)

### Installation

```bash
git clone https://github.com/HCSS-Data-Lab/Strategic-LLM-IPD.git
cd Strategic-LLM-IPD
pip install -r requirements.txt
