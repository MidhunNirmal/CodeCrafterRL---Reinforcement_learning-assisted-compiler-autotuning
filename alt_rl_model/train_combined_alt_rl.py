import os
import numpy as np
import torch
import pandas as pd
from alt_comp_env import AltCompilerEnvironment
from dqn_agent import DQNAgent
import random

EPISODES = 2000
SAVE_PATH = "combined_dqn_agent.pth"
PROCESSED_DIR = "processed"

class CombinedAltRLTrainer:
    def __init__(self):
        self.benchmark_data = {}
        self.environments = {}
        self.agent = None
        self.benchmark_names = []
        
    def load_processed_data(self):
        """Load all preprocessed benchmark data"""
        processed_dir = os.path.join(os.path.dirname(__file__), PROCESSED_DIR)
        for filename in os.listdir(processed_dir):
            if filename.endswith('.csv'):
                benchmark_name = filename.replace('.csv', '')
                filepath = os.path.join(processed_dir, filename)
                df = pd.read_csv(filepath)
                self.benchmark_data[benchmark_name] = df
                self.benchmark_names.append(benchmark_name)
        
        print(f"Loaded {len(self.benchmark_names)} benchmarks")
        
    def setup_environments(self):
        """Create environments for all benchmarks"""
        for benchmark_name in self.benchmark_names:
            env = AltCompilerEnvironment(self.benchmark_data, benchmark_name)
            self.environments[benchmark_name] = env
            
        # Use the first environment to determine state/action size
        first_env = list(self.environments.values())[0]
        state_size = first_env.observation_space.shape[0]
        action_size = first_env.action_space.n if hasattr(first_env.action_space, 'n') else first_env.action_space.nvec[0]
        
        print(f"State size: {state_size}, Action size: {action_size}")
        
        # Create unified agent
        self.agent = DQNAgent(
            state_size=state_size, 
            action_size=action_size,
            lr=0.0005,  # Lower learning rate for stability
            epsilon_decay=0.9995,  # Slower decay
            memory_size=50000,  # Larger memory
            batch_size=64  # Larger batch size
        )
        
    def train_combined_model(self):
        """Train the agent on all benchmarks"""
        best_improvements = {name: -np.inf for name in self.benchmark_names}
        
        for episode in range(EPISODES):
            # Select benchmark (round-robin)
            benchmark_name = self.benchmark_names[episode % len(self.benchmark_names)]
            env = self.environments[benchmark_name]
            
            state = env.reset()
            total_reward = 0
            
            while True:
                action = self.agent.act(state, training=True)
                next_state, reward, done, info = env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.replay()
                state = next_state
                total_reward += reward
                
                if done:
                    if info['improvement'] > best_improvements[benchmark_name]:
                        best_improvements[benchmark_name] = info['improvement']
                    break
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_improvement = np.mean(list(best_improvements.values()))
                print(f"Episode {episode+1}/{EPISODES} | "
                      f"Benchmark: {benchmark_name} | "
                      f"Reward: {total_reward:.4f} | "
                      f"Avg Best Improvement: {avg_improvement:.4f}")
        
        # Save combined model
        save_path = os.path.join(os.path.dirname(__file__), SAVE_PATH)
        self.agent.save(save_path)
        print(f"Combined model saved to: {save_path}")
        
        # Print final results
        print("\nFinal Best Improvements:")
        for name, improvement in best_improvements.items():
            print(f"  {name}: {improvement:.4f}")

def main():
    trainer = CombinedAltRLTrainer()
    trainer.load_processed_data()
    trainer.setup_environments()
    trainer.train_combined_model()

if __name__ == "__main__":
    main() 