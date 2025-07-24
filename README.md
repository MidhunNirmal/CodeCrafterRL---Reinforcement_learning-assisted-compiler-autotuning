# CodeCrafterRL: Reinforcement Learning-Assisted Compiler Autotuning

Welcome to **CodeCrafterRL**, an open-source research project designed to explore **Reinforcement Learning (RL)** for **automatic compiler flag tuning** across benchmarks in the [PolyBench suite](https://github.com/stefanocereda/polybench_data). This project leverages Deep Q-Networks (DQN) to learn optimal compiler configurations that minimize execution time for a given benchmark.

---

## 🚀 Features

- 🧠 **Reinforcement Learning environment** using OpenAI Gym API
- ⚙️ RL agent (DQN) trained to select compiler flags
- 📊 Execution time prediction using Nearest Neighbors and heuristic fallback
- 📈 Reward curves plotted and saved for performance analysis


# Alternate RL Model & LOOCV (Leave-One-Out Cross Validation)

This directory contains advanced reinforcement learning (RL) models for compiler optimization, with a focus on the expanded features DQN agent, which has demonstrated state-of-the-art performance.

## Overview

- **alt_rl_model/**: Implements RL agents (notably DQN) for optimizing compiler flags using benchmark data. Includes environments, data preprocessors, training scripts, and analysis tools.
- **loocv/**: Implements Leave-One-Out Cross Validation (LOOCV) for the expanded features DQN agent, testing generalization by excluding a benchmark from training and evaluating on it.

---

## Why Expanded Features DQN?

**The expanded features DQN agent is the best model.**

- **100% Success Rate:** All 24 benchmarks optimized successfully!
- **Average Improvement:** 21.25% (vs previous ~16%)
- **Best Performance:** Up to 47.98% improvement on `network_patricia`
- **Top Performers:**
  - `network_patricia`: 47.98% improvement
  - `security_sha`: 42.21% improvement
  - `office_stringsearch1`: 35.11% improvement
  - `automotive_susan_s`: 32.62% improvement
  - `telecom_adpcm_c`: 31.38% improvement
- **Key:** The 279 additional static features (code characteristics, memory access, basic block info, branch prediction) allow the model to make much more informed optimization decisions.

---

## Directory Structure

- **alt_rl_model/**
  - `alt_comp_env.py`, `alt_comp_env_unified.py`: RL environments for standard and unified (multi-benchmark) training
  - `alt_data_preprocessor.py`, `alt_data_preprocessor_correct.py`: Data preprocessing scripts
  - `dqn_agent.py`: DQN agent implementation
  - `train_alt_rl.py`, `train_unified_alt_rl.py`, `train_expanded_features.py`: Training scripts for various models
  - `test_model.py`: Model evaluation and analysis
  - `analyze_alt_dataset.py`: Data analysis and visualization
  - `processed/`: Preprocessed benchmark data
  - `results/`, `plots/`: Output directories for results and visualizations

- **loocv/**
  - `loocv_data_preprocessor.py`: Data preprocessor for LOOCV
  - `loocv_environment.py`: RL environment for LOOCV
  - `loocv_trainer.py`: Main LOOCV training script
  - `loocv_tester.py`: LOOCV testing script
  - Output files: trained models, results, and plots

---

## How to Run

### 1. Train the Expanded Features DQN Agent

```bash
cd alt_rl_model
python train_unified_alt_rl.py
```
- Trains the DQN agent using all benchmarks and expanded features.
- Model weights saved as `unified_alt_dqn_agent.pth`.

### 2. Test/Evaluate the Trained Model

```bash
python test_model.py
```
- Evaluates the trained model on all benchmarks.
- Prints summary statistics and saves results/plots.

### 3. Leave-One-Out Cross Validation (LOOCV)

#### a. Train with LOOCV (excluding `security_sha`):
```bash
cd loocv
python loocv_trainer.py
```
- Trains the expanded features DQN agent on all benchmarks except `security_sha`.
- Evaluates on the excluded benchmark.
- Saves model and results in the `loocv/` directory.

#### b. Additional Testing (Optional):
```bash
python loocv_tester.py
```
- Loads the LOOCV-trained model and runs further evaluation on `security_sha`.

---

## Dependencies

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Gym

Install dependencies with:
```bash
pip install -r ../requirements.txt
```

---

## Results Summary

- **Expanded Features DQN is the best model:**
  - 100% success rate, highest improvements, and best generalization (see `alt_rl_model/trainres.txt` for details).
- **LOOCV** demonstrates strong generalization to unseen benchmarks.

---

## Citation
If you use this code or results, please cite the project or contact the authors for more information. 
