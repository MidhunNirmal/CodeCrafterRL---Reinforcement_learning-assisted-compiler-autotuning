import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Load the PolyBench dataset for 2mm benchmark only
def load_dataset(file_path="exec_times.csv"):
    """Load and preprocess exec_times.csv, filtering for 2mm benchmark."""
    data = pd.read_csv(file_path)
    # Filter for 2mm benchmark
    data = data[data["benchmark"] == "2mm"]
    if data.empty:
        raise ValueError("No data found for benchmark '2mm' in exec_times.csv")
    
    # Assumed flag columns (update based on data.columns if different)
    flag_columns = [
        "floop-interchange",
        "floop-strip-mine",
        "floop-block",
        "funroll-loops",
        "ftree-vectorize",
        "fprefetch-loop-arrays",
        "fopenmp"
    ]
    # Verify columns exist
    missing_cols = [col for col in flag_columns if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in CSV: {missing_cols}. Run `print(data.columns)` to check.")
    
    # Convert flag columns to binary (X -> 0, otherwise -> 1)
    for col in flag_columns:
        data[col] = data[col].apply(lambda x: 0 if x == "X" else 1)
    return data

# Custom Gym environment for compiler autotuning (2mm only)
class CompilerEnv(gym.Env):
    def __init__(self, benchmark="2mm", dataset=None, dataset_size="STANDARD_DATASET"):
        """Initialize environment for 2mm benchmark."""
        super(CompilerEnv, self).__init__()
        self.benchmark = benchmark
        self.dataset = dataset
        self.dataset_size = dataset_size
        self.action_space = gym.spaces.Discrete(128)  # 2^7 flag combinations
        # State: code features (50-dim placeholder), 7 flags, 1 runtime
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(50 + 7 + 1,), dtype=np.float32)
        self.current_flags = np.zeros(7, dtype=np.float32)
        self.code_features = self.extract_code_features()
        self.baseline_time = self.get_runtime(np.zeros(7))  # Baseline: all flags off
        if self.baseline_time == float("inf"):
            raise ValueError("Baseline runtime (all flags off) not found for 2mm")

    def extract_code_features(self):
        """Placeholder: Extract static code features for 2mm (e.g., via CNN)."""
        # Replace with your CNN or LLVM-based feature extraction for 2mm
        return np.random.rand(50).astype(np.float32)  # Placeholder

    def action_to_flags(self, action):
        """Convert action (0–127) to 7-bit binary flag vector."""
        return np.array([int(x) for x in format(action, "07b")], dtype=np.float32)

    def get_runtime(self, flags):
        """Look up execution time for 2mm with given flags."""
        flag_columns = [
            "floop-interchange",
            "floop-strip-mine",
            "floop-block",
            "funroll-loops",
            "ftree-vectorize",
            "fprefetch-loop-arrays",
            "fopenmp"
        ]
        row = self.dataset[
            (self.dataset["benchmark"] == self.benchmark) &
            (self.dataset[flag_columns].eq(flags).all(axis=1))
        ]
        if not row.empty:
            return row[self.dataset_size].values[0]
        return float("inf")  # Penalize invalid configurations

    def step(self, action):
        """Take an action (flag combination), return next state, reward, done."""
        self.current_flags = self.action_to_flags(action)
        runtime = self.get_runtime(self.current_flags)
        reward = -runtime / self.baseline_time  # Normalize to baseline
        state = np.concatenate([self.code_features, self.current_flags, [runtime / self.baseline_time]])
        done = False  # Episodes run for fixed steps
        return state, reward, done, {}

    def reset(self):
        """Reset to initial state (all flags off)."""
        self.current_flags = np.zeros(7, dtype=np.float32)
        runtime = self.get_runtime(self.current_flags)
        return np.concatenate([self.code_features, self.current_flags, [runtime / self.baseline_time]])

# Custom DQN policy with feature extractor
class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        """Initialize feature extractor for state processing."""
        super().__init__(observation_space, features_dim=128)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

    def forward(self, observations):
        """Process state (code features + flags + runtime)."""
        return self.net(observations)

# Main function to train and test the RL model for 2mm
def main():
    # Load dataset (filtered for 2mm)
    dataset = load_dataset("exec_times.csv")
    
    # Initialize environment for 2mm
    env = CompilerEnv(benchmark="2mm", dataset=dataset, dataset_size="STANDARD_DATASET")
    
    # Define DQN policy
    policy_kwargs = {"features_extractor_class": CustomExtractor}
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1
    )
    
    # Train the model
    model.learn(total_timesteps=10000, log_interval=100)
    model.save("dqn_autotune_2mm")
    
    # Test the model
    obs = env.reset()
    total_reward = 0
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        flags = env.action_to_flags(action)
        print(f"Action: {action}, Flags: {flags}, Reward: {reward:.4f}, Runtime: {-reward * env.baseline_time:.4f}")
        if done:
            obs = env.reset()
    print(f"Average reward for 2mm: {total_reward / 100:.4f}")

if __name__ == "__main__":
    main()