# LLM Strategic Behavior in Iterated Prisoner's Dilemma

This repository contains the code and data for research on strategic behavior of Large Language Models (LLMs) in evolutionary Iterated Prisoner's Dilemma tournaments.

## Repository Contents

### Core Files

- **`evolutionary_PD_expanded.py`** - Main tournament simulation code that runs evolutionary Prisoner's Dilemma games with LLM agents
- **`labeling_sample.csv`** - Hand-coded sample of LLM rationales analyzed for horizon awareness and opponent modeling
- **`machine_coder.py`** - LLM-based coding system for analyzing strategic reasoning in rationales
- **`create_labeling_sample.py`** - Script to create representative samples from tournament data for manual coding
- **`requirements.txt`** - Python dependencies needed to run the code

### Data

- **`Consolidated results for evo PD/`** - Complete tournament results and strategic fingerprint data
  - Tournament logs with move-by-move data and LLM reasoning
  - Strategic fingerprint analyses showing behavioral patterns
  - Multiple experimental conditions (different termination probabilities)

## Research Overview

This research investigates whether Large Language Models exhibit genuine strategic thinking in game-theoretic contexts by:

1. **Running evolutionary tournaments** where LLMs compete against classic strategies
2. **Analyzing strategic reasoning** through detailed examination of LLM rationales
3. **Measuring strategic behaviors** including horizon awareness and opponent modeling
4. **Comparing AI models** (OpenAI GPT, Google Gemini, Anthropic Claude) across different strategic dimensions

## Key Findings

- LLMs demonstrate distinct "strategic fingerprints" with measurable differences in cooperation, adaptation, and strategic reasoning
- Models show varying sensitivity to time horizons and opponent behavior
- Evidence for genuine strategic thinking rather than simple pattern matching

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Set up API keys in `Axelrod.env` file (see code for required format)
3. Run tournaments: `python evolutionary_PD_expanded.py`
4. Analyze results using the provided analysis scripts

## Citation

If you use this code or data in your research, please cite our paper:

[Paper citation will be added upon publication]

## License

This research code is provided for academic use. See individual files for specific licensing terms. 