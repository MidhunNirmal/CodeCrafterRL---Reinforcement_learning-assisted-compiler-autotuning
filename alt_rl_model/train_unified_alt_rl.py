import os
import numpy as np
import torch
import pandas as pd
from alt_data_preprocessor_correct import AltDataPreprocessorCorrect
from alt_comp_env_unified import AltUnifiedCompilerEnvironment
from dqn_agent import DQNAgent
import random

EPISODES = 3000
SAVE_PATH = "unified_alt_dqn_agent.pth"

class UnifiedAltRLTrainer:
    def __init__(self):
        self.benchmark_data = {}
        self.unified_env = None
        self.agent = None
        self.benchmark_names = []
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data using the corrected preprocessor"""
        pre = AltDataPreprocessorCorrect()
        pre.load_data()
        self.benchmark_data = pre.preprocess()
        self.benchmark_names = list(self.benchmark_data.keys())
        print(f"Loaded {len(self.benchmark_names)} benchmarks")
        
    def setup_unified_environment(self):
        """Create unified environment for all benchmarks"""
        self.unified_env = AltUnifiedCompilerEnvironment(self.benchmark_data)
        
        # Get state and action sizes
        state_size = self.unified_env.observation_space.shape[0]
        action_size = 7  # 7 compiler flags
        
        print(f"State size: {state_size} (7 flags + {self.unified_env.num_static_features} static features + {len(self.benchmark_names)} benchmarks + 3 context)")
        print(f"Action size: {action_size} (7 binary flags)")
        
        # Create unified agent with larger network for the expanded state
        self.agent = DQNAgent(
            state_size=state_size, 
            action_size=action_size,
            lr=0.0003,  # Lower learning rate for larger network
            epsilon_decay=0.9997,  # Slower decay for more exploration
            memory_size=100000,  # Larger memory for more complex state
            batch_size=128  # Larger batch size for stability
        )
        
    def train_unified_model(self):
        """Train the unified model on all benchmarks"""
        best_improvements = {name: -np.inf for name in self.benchmark_names}
        episode_rewards = []
        episode_benchmarks = []
        
        print(f"Starting unified training for {EPISODES} episodes...")
        
        for episode in range(EPISODES):
            # Select benchmark (round-robin)
            benchmark_name = self.benchmark_names[episode % len(self.benchmark_names)]
            
            # Reset environment with specific benchmark
            state = self.unified_env.reset(benchmark_name)
            total_reward = 0
            
            while True:
                action = self.agent.act(state, training=True)
                next_state, reward, done, info = self.unified_env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.replay()
                state = next_state
                total_reward += reward
                
                if done:
                    if info['improvement'] > best_improvements[benchmark_name]:
                        best_improvements[benchmark_name] = info['improvement']
                    break
            
            # Track statistics
            episode_rewards.append(total_reward)
            episode_benchmarks.append(benchmark_name)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_improvement = np.mean(list(best_improvements.values()))
                print(f"Episode {episode+1}/{EPISODES} | "
                      f"Benchmark: {benchmark_name} | "
                      f"Reward: {total_reward:.4f} | "
                      f"Avg Reward: {avg_reward:.4f} | "
                      f"Avg Best Improvement: {avg_improvement:.4f}")
        
        # Save unified model
        save_path = os.path.join(os.path.dirname(__file__), SAVE_PATH)
        self.agent.save(save_path)
        print(f"Unified model saved to: {save_path}")
        
        # Print final results
        print("\n" + "="*80)
        print("FINAL TRAINING RESULTS")
        print("="*80)
        print(f"Total episodes: {EPISODES}")
        print(f"Average reward: {np.mean(episode_rewards):.4f}")
        print(f"Average best improvement: {np.mean(list(best_improvements.values())):.4f}")
        
        print(f"\nBest improvements by benchmark:")
        for name, improvement in sorted(best_improvements.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {improvement:.4f}")
        
        return best_improvements, episode_rewards

def main():
    trainer = UnifiedAltRLTrainer()
    
    print("Loading and preprocessing data...")
    trainer.load_and_preprocess_data()
    
    print("Setting up unified environment...")
    trainer.setup_unified_environment()
    
    print("Starting unified training...")
    best_improvements, episode_rewards = trainer.train_unified_model()
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main() 