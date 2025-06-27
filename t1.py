import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Load the PolyBench dataset
def load_dataset(file_path="exec_times.csv"):
    """Load and preprocess exec_times.csv from the PolyBench dataset."""
    data = pd.read_csv(file_path)
    # Define flag columns based on CSV
    flag_columns = [
        '-funsafe-math-optimizations ',
        '-fno-guess-branch-probability ',
        '-fno-ivopts ',
        '-fno-tree-loop-optimize ',
        '-fno-inline-functions ',
        '-funroll-all-loops ',
        '-O2 '
    ]
    # Verify columns exist
    missing_cols = [col for col in flag_columns if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in CSV: {missing_cols}. Check CSV column names with `data.columns`.")
    
    # Convert flag columns to binary (X -> 0, otherwise -> 1)
    for col in flag_columns:
        data[col] = data[col].apply(lambda x: 0 if pd.isna(x) or x == "X" else 1)
    
    # Compute average execution time across runs
    time_columns = ['execution_time_1', 'execution_time_2', 'execution_time_3',
                    'execution_time_4', 'execution_time_5']
    data['avg_execution_time'] = data[time_columns].mean(axis=1, skipna=True)
    if data['avg_execution_time'].isna().all():
        raise ValueError("All execution times are NaN. Check data integrity.")
    elif data['avg_execution_time'].isna().any():
        print("Warning: Some rows have NaN execution times. Using available data.")
    
    return data, flag_columns

# Custom Gym environment for compiler autotuning
class CompilerEnv(gym.Env):
    def __init__(self, benchmark, dataset, flag_columns):
        """Initialize environment for a specific benchmark."""
        super(CompilerEnv, self).__init__()
        self.benchmark = benchmark
        self.dataset = dataset
        self.flag_columns = flag_columns
        self.action_space = gym.spaces.Discrete(128)  # 2^7 flag combinations
        # State: code features (50-dim placeholder), 7 flags, 1 avg runtime
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(50 + 7 + 1,), dtype=np.float32)
        self.current_flags = np.zeros(7, dtype=np.float32)
        self.code_features = self.extract_code_features()
        
        # Validate and set benchmark
        available_benchmarks = dataset['APP_NAME'].dropna().unique()
        if benchmark not in available_benchmarks:
            print(f"Warning: Benchmark '{benchmark}' not found. Using first available: {available_benchmarks[0]}")
            self.benchmark = available_benchmarks[0]
        
        # Set baseline time
        self.baseline_time = self.get_runtime(np.zeros(7))  # Baseline: all flags off
        if np.isnan(self.baseline_time) or self.baseline_time == float("inf"):
            print(f"Warning: Invalid baseline time for {self.benchmark}: {self.baseline_time}. Using 1.0 as fallback.")
            self.baseline_time = 1.0  # Fallback to avoid NaN rewards

    def extract_code_features(self):
        """Placeholder: Extract static code features (e.g., via LLVM or CNN)."""
        # Replace with your CNN or LLVM-based feature extraction
        return np.random.rand(50).astype(np.float32)  # Placeholder

    def action_to_flags(self, action):
        """Convert action (0–127) to 7-bit binary flag vector."""
        return np.array([int(x) for x in format(action, "07b")], dtype=np.float32)

    def get_runtime(self, flags):
        """Look up average execution time from dataset for given benchmark and flags."""
        row = self.dataset[
            (self.dataset['APP_NAME'] == self.benchmark) &
            (self.dataset[self.flag_columns].eq(flags).all(axis=1))
        ]
        if not row.empty:
            runtime = row['avg_execution_time'].values[0]
            print(f"Debug: Benchmark {self.benchmark}, Flags {flags}, Runtime {runtime}")
            if pd.isna(runtime) or runtime <= 0:
                print(f"Warning: Invalid runtime ({runtime}) for {self.benchmark} with flags {flags}")
                return float("inf")
            return runtime
        print(f"Warning: No data for {self.benchmark} with flags {flags}")
        return float("inf")  # Penalize missing configurations

    def step(self, action):
        """Take an action (flag combination), return next state, reward, done."""
        self.current_flags = self.action_to_flags(action)
        runtime = self.get_runtime(self.current_flags)
        reward = -1000 * (runtime / self.baseline_time - 1) if self.baseline_time > 0 else -1000 * runtime
        if runtime == self.baseline_time:
            print(f"Warning: Runtime equals baseline for action {action}, reward set to 0")
            reward = 0.0
        state = np.concatenate([self.code_features, self.current_flags, [min(runtime / self.baseline_time if self.baseline_time > 0 else runtime, 1.0)]])
        done = False  # Episodes run for fixed steps
        return state, reward, done, {}

    def reset(self):
        """Reset to initial state (all flags off)."""
        self.current_flags = np.zeros(7, dtype=np.float32)
        runtime = self.get_runtime(self.current_flags)
        return np.concatenate([self.code_features, self.current_flags, [min(runtime / self.baseline_time if self.baseline_time > 0 else runtime, 1.0)]])

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

# Main function to train and test the RL model
def main():
    # Load dataset
    try:
        dataset, flag_columns = load_dataset("exec_times.csv")
    except (KeyError, ValueError) as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Initialize environment for a specific benchmark
    env = CompilerEnv(benchmark="linear-algebra/kernels/2mm", dataset=dataset, flag_columns=flag_columns)
    
    # Define DQN policy
    policy_kwargs = {"features_extractor_class": CustomExtractor}
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=20000,  # Increased buffer size
        learning_starts=2000,  # Increased to allow more initial exploration
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.3,  # Further increased for more exploration
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1
    )
    
    # Train the model
    print("Starting training for 100,000 timesteps...")
    model.learn(total_timesteps=100000, log_interval=100)
    model.save("dqn_autotune")
    
    # Test the model
    obs = env.reset()
    total_reward = 0
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        flags = env.action_to_flags(action)
        print(f"Action: {action}, Flags: {flags}, Reward: {reward:.4f}, Runtime: {-reward / 1000 + 1:.4f} * baseline")
        if done:
            obs = env.reset()
    print(f"Average reward: {total_reward / 100:.4f}")

if __name__ == "__main__":
    main()