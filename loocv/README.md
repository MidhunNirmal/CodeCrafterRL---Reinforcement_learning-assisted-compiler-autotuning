# LOOCV (Leave-One-Out Cross Validation) Implementation

This folder contains a complete implementation of Leave-One-Out Cross Validation (LOOCV) for the expanded features DQN agent, specifically excluding the "security_sha" benchmark from training and then testing on it.

## Overview

The LOOCV implementation replicates the expanded features DQN agent training process from the `alt_rl_model` folder, but with a key difference: it excludes one benchmark ("security_sha") from the training data and then evaluates the trained model specifically on that excluded benchmark.

## Files

### Core Implementation Files

1. **`loocv_data_preprocessor.py`** - Modified data preprocessor that excludes a specified benchmark from training data
2. **`loocv_environment.py`** - Modified RL environment that handles separate training and test data
3. **`loocv_trainer.py`** - Main training script that implements LOOCV training
4. **`loocv_tester.py`** - Separate testing script for evaluating the trained model

### Output Files (Generated)

- `loocv_expanded_features_dqn_agent.pth` - Trained model weights
- `loocv_training_results.csv` - Training performance results
- `loocv_training_progress.png` - Training progress visualization
- `loocv_test_results.csv` - Detailed test results
- `loocv_detailed_test_results.csv` - Episode-by-episode test results
- `loocv_test_results.png` - Comprehensive test results visualization

## How It Works

### 1. Data Preprocessing
- Loads the complete dataset from `../alternatedata/raw_data/data.csv`
- Excludes "security_sha" from training data
- Keeps "security_sha" as test data only
- Applies the same feature preprocessing as the original expanded features model

### 2. Environment Setup
- Creates a unified environment for all training benchmarks
- Uses the same 7 compiler flags + 279 static features + benchmark encoding
- Maintains separate performance models for training and test benchmarks

### 3. Training Process
- Trains the DQN agent on all benchmarks except "security_sha"
- Uses the same hyperparameters as the original expanded features model:
  - 5000 episodes
  - Learning rate: 0.0002
  - Epsilon decay: 0.9998
  - Memory size: 150,000
  - Batch size: 256

### 4. Testing Process
- Evaluates the trained model specifically on "security_sha"
- Runs multiple episodes to assess generalization performance
- Provides detailed analysis of improvement rates and success metrics

## Usage

### Step 1: Training
```bash
cd loocv
python loocv_trainer.py
```

This will:
- Load and preprocess data, excluding "security_sha"
- Train the DQN agent on remaining benchmarks
- Save the trained model and training results
- Evaluate the model on "security_sha"

### Step 2: Testing (Optional)
If you want to run additional testing:

```bash
python loocv_tester.py
```

This will:
- Load the trained model
- Run additional evaluation episodes on "security_sha"
- Generate detailed visualizations and analysis

## Key Differences from Original

1. **Data Split**: Excludes "security_sha" from training, uses it only for testing
2. **Environment**: Modified to handle separate training and test datasets
3. **Evaluation**: Focused evaluation on the excluded benchmark
4. **Analysis**: Detailed analysis of generalization performance

## Expected Results

The LOOCV approach tests the model's ability to generalize to unseen benchmarks. Results will show:

- How well the model trained on other benchmarks performs on "security_sha"
- Generalization capability of the expanded features approach
- Comparison with the original model's performance on "security_sha"

## Files Structure

```
loocv/
├── loocv_data_preprocessor.py    # Data preprocessing with LOOCV
├── loocv_environment.py          # Modified RL environment
├── loocv_trainer.py              # Main training script
├── loocv_tester.py               # Testing script
├── README.md                     # This file
├── loocv_expanded_features_dqn_agent.pth  # Trained model (generated)
├── loocv_training_results.csv    # Training results (generated)
├── loocv_training_progress.png   # Training visualization (generated)
├── loocv_test_results.csv        # Test results (generated)
├── loocv_detailed_test_results.csv # Detailed test results (generated)
└── loocv_test_results.png        # Test visualization (generated)
```

## Dependencies

The implementation uses the same dependencies as the original expanded features model:
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Gym

Make sure you have the `alt_rl_model` folder available for importing the DQN agent implementation. 