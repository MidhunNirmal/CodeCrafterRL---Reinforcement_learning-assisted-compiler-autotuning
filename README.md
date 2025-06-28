# CodeCrafterRL: Reinforcement Learning-Assisted Compiler Autotuning

Welcome to **CodeCrafterRL**, an open-source research project designed to explore **Reinforcement Learning (RL)** for **automatic compiler flag tuning** across benchmarks in the [PolyBench suite](https://github.com/stefanocereda/polybench_data). This project leverages Deep Q-Networks (DQN) to learn optimal compiler configurations that minimize execution time for a given benchmark.

---

## 🚀 Features

- 🧠 **Reinforcement Learning environment** using OpenAI Gym API
- ⚙️ RL agent (DQN) trained to select compiler flags
- 📊 Execution time prediction using Nearest Neighbors and heuristic fallback
- 📈 Reward curves plotted and saved for performance analysis
- 🧹 Fully modular: `data_preprocessing.py`, `compiler_environment.py`, `dqn_agent.py`, `trainer.py`, `main.py`

---

## 📁 Project Structure

```
CodeCrafterRL/
├── data/                           # Raw input CSVs (exec_times.csv)
├── checkpoints/                   # Saved model checkpoints
├── reward_curves/                 # Saved plots per benchmark
├── data_preprocessing.py          # Dataset loader and cleaner
├── compiler_environment.py        # Gym environment definition
├── dqn_agent.py                   # Deep Q-Learning agent
├── trainer.py                     # Training loop
├── main.py                        # Entry point to run the pipeline
├── test_reward_plot.py            # Test plot generator
└── README.md                      # Project documentation
```

---

## 📦 Dataset

The dataset used for this project is sourced from:

🔗 [https://github.com/stefanocereda/polybench\_data](https://github.com/stefanocereda/polybench_data)

- Pre-collected execution times of PolyBench benchmarks under various compiler flag settings
- Save the dataset CSV (`exec_times.csv`) in the `data/` directory

---

## ✅ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/MidhunNirmal/CodeCrafterRL---Reinforcement_learning-assisted-compiler-autotuning.git
cd CodeCrafterRL---Reinforcement_learning-assisted-compiler-autotuning
```

### 2. Create a Virtual Environment (Optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, manually install:

```bash
pip install numpy pandas matplotlib seaborn torch scikit-learn gym
```

---

## 🚦 Running the Project

### Step 1: Prepare the dataset

- Download or generate the `exec_times.csv` from the PolyBench data repo.
- Place it inside the `data/` directory.

### Step 2: Start Training

```bash
python main.py
```

### Outputs:

- Checkpoints: `checkpoints/<benchmark>/`
- Reward Curves: `reward_curves/<benchmark>_reward_curve.png`

---

## 📊 Visualization

To test plotting rewards independently:

```bash
python test_reward_plot.py
```

This will simulate a reward curve and save it under `reward_curves/`.

---



Great! Here's a refined and concise **`TESTING.md`** in Markdown, ready for direct use in your project:

```markdown
# 🧪 Testing Guide for CodeCrafterRL

Run tests in multiple modes depending on your goals:

---

## 🔹 1. Quick Test (Recommended First Step)

```bash
python quick_test_examples.py
```

Runs:
- Agent vs dataset performance
- Configuration analysis
- Decision pattern evaluation
- Benchmark comparisons

---

## 🔹 2. Interactive CLI

```bash
python interactive_test.py
```

Explore with a menu-driven interface:
- Choose benchmarks
- Test custom configurations
- Compare with best/worst known settings
- Run targeted evaluations

---

## 🔹 3. Full Test Suite

```bash
python run_tests.py
```

Executes all tests and saves:
- 📊 Plots to `test_plots/`
- 📝 JSON reports to `test_results/`

---

## ⚙️ Prerequisites

### Required Files:
- `data/exec_times.csv`
- Trained models in `models/`

### Required Scripts:
- `data_preprocessing.py`
- `compiler_environment.py`
- `dqn_agent.py`
- `trainer.py`

### Dependencies:

```bash
pip install pandas numpy matplotlib seaborn torch
```

Optional (PyTorch):

```bash
# For GPU
pip install torch torchvision torchaudio

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 🧪 Run Custom Tests

```python
from test_system import CompilerOptimizationTester

tester = CompilerOptimizationTester("data/exec_times.csv")
tester.setup()
results = tester.test_agent_vs_dataset("2mm", n_tests=50)
print(f"Win rate: {results['summary']['success_rate']:.2%}")
```

---

## 🛠️ Troubleshooting

- ❗ **No trained models?**
  - Run: `python trainer.py`

- 📂 **Missing dataset?**
  - Ensure `data/exec_times.csv` exists or update the path in code

- 🐍 **Import errors?**
  - Install dependencies with `pip install -r requirements.txt` or manually

---

✅ **Recommended Start:**

```bash
python quick_test_examples.py
```

This verifies your setup and shows meaningful test output!
```






## 🤝 Contributing

We welcome contributions of all kinds:

- 📈 Improve model architecture
- 🚀 Add support for PPO, A2C or other RL agents
- 🧪 Write unit tests or refactor training loop
- 📖 Improve documentation or Jupyter notebooks for explainability

### To contribute:

1. Fork this repo
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add your message'`
4. Push to your fork: `git push origin feature/your-feature`
5. Open a Pull Request ❤️

---

## 📬 Contact

For questions or collaborations:

- GitHub: [@MidhunNirmal](https://github.com/MidhunNirmal)
- Issues: Feel free to open one!

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

Made with ❤️ for compiler autotuning and AI research.

