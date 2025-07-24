import os
import numpy as np
import torch
import pandas as pd
from alt_data_preprocessor_correct import AltDataPreprocessorCorrect
from alt_comp_env_unified import AltUnifiedCompilerEnvironment
from dqn_agent import DQNAgent
import random

EPISODES = 5000  # More episodes for the larger model
SAVE_PATH = "expanded_features_dqn_agent1.pth"

class ExpandedFeaturesRLTrainer:
    def __init__(self):
        self.benchmark_data = {}
        self.unified_env = None
        self.agent = None
        self.benchmark_names = []
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data using the expanded preprocessor"""
        pre = AltDataPreprocessorCorrect()
        pre.load_data()
        self.benchmark_data = pre.preprocess()
        self.benchmark_names = list(self.benchmark_data.keys())
        print(f"Loaded {len(self.benchmark_names)} benchmarks with expanded features")
        
    def setup_unified_environment(self):
        """Create unified environment for all benchmarks with expanded features"""
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
            lr=0.0002,  # Lower learning rate for larger network
            epsilon_decay=0.9998,  # Slower decay for more exploration
            memory_size=150000,  # Larger memory for more complex state
            batch_size=256  # Larger batch size for stability
        )
        
    def train_expanded_model(self):
        """Train the expanded model on all benchmarks"""
        best_improvements = {name: -np.inf for name in self.benchmark_names}
        episode_rewards = []
        episode_benchmarks = []
        
        print(f"Starting expanded features training for {EPISODES} episodes...")
        print("This model uses 7 flags + 279 static features for better performance prediction")
        
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
            if (episode + 1) % 200 == 0:
                avg_reward = np.mean(episode_rewards[-200:])
                avg_improvement = np.mean(list(best_improvements.values()))
                successful_benchmarks = len([imp for imp in best_improvements.values() if imp > 0])
                print(f"Episode {episode+1}/{EPISODES} | "
                      f"Benchmark: {benchmark_name} | "
                      f"Reward: {total_reward:.4f} | "
                      f"Avg Reward: {avg_reward:.4f} | "
                      f"Avg Best Improvement: {avg_improvement:.4f} | "
                      f"Successful: {successful_benchmarks}/{len(self.benchmark_names)}")
        
        # Save expanded model
        save_path = os.path.join(os.path.dirname(__file__), SAVE_PATH)
        self.agent.save(save_path)
        print(f"Expanded features model saved to: {save_path}")
        
        # Print final results
        print("\n" + "="*80)
        print("EXPANDED FEATURES TRAINING RESULTS")
        print("="*80)
        print(f"Total episodes: {EPISODES}")
        print(f"Average reward: {np.mean(episode_rewards):.4f}")
        print(f"Average best improvement: {np.mean(list(best_improvements.values())):.4f}")
        
        successful_benchmarks = len([imp for imp in best_improvements.values() if imp > 0])
        print(f"Successful optimizations: {successful_benchmarks}/{len(self.benchmark_names)} ({successful_benchmarks/len(self.benchmark_names)*100:.1f}%)")
        
        print(f"\nBest improvements by benchmark:")
        for name, improvement in sorted(best_improvements.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {improvement:.4f}")
        
        return best_improvements, episode_rewards

def main():
    trainer = ExpandedFeaturesRLTrainer()
    
    print("Loading and preprocessing data with expanded features...")
    trainer.load_and_preprocess_data()
    
    print("Setting up unified environment with expanded features...")
    trainer.setup_unified_environment()
    
    print("Starting expanded features training...")
    best_improvements, episode_rewards = trainer.train_expanded_model()
    
    print("\nExpanded features training completed successfully!")

if __name__ == "__main__":
    main() 