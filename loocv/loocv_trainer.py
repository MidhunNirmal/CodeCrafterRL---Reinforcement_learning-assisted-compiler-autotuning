import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from loocv_data_preprocessor import LOOCVDataPreprocessor
from loocv_environment import LOOCVCompilerEnvironment
import sys
sys.path.append('../alt_rl_model')
from dqn_agent import DQNAgent
import random

EPISODES = 5000  # Same as original expanded features training
SAVE_PATH = "loocv_expanded_features_dqn_agent.pth"
RESULTS_CSV = "loocv_training_results.csv"
PLOT_PNG = "loocv_training_progress.png"

class LOOCVExpandedFeaturesTrainer:
    def __init__(self, exclude_benchmark: str = "security_sha"):
        self.exclude_benchmark = exclude_benchmark
        self.benchmark_data = {}
        self.training_data = {}
        self.test_data = {}
        self.loocv_env = None
        self.agent = None
        self.training_benchmarks = []
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data using LOOCV preprocessor"""
        print(f"Loading data and excluding '{self.exclude_benchmark}' from training...")
        pre = LOOCVDataPreprocessor(exclude_benchmark=self.exclude_benchmark)
        pre.load_data()
        self.benchmark_data = pre.preprocess()
        self.training_data = pre.get_training_data()
        self.test_data = pre.get_test_data()
        self.training_benchmarks = list(self.training_data.keys())
        print(f"Loaded {len(self.training_benchmarks)} training benchmarks")
        print(f"Test benchmark: {list(self.test_data.keys())}")
        
    def setup_loocv_environment(self):
        """Create LOOCV environment for training benchmarks"""
        self.loocv_env = LOOCVCompilerEnvironment(self.training_data, self.test_data)
        
        # Get state and action sizes
        state_size = self.loocv_env.observation_space.shape[0]
        action_size = 7  # 7 compiler flags
        
        print(f"State size: {state_size} (7 flags + {self.loocv_env.num_static_features} static features + {len(self.training_benchmarks)} benchmarks + 3 context)")
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
        
    def train_loocv_model(self):
        """Train the LOOCV model on all benchmarks except the excluded one"""
        best_improvements = {name: -np.inf for name in self.training_benchmarks}
        episode_rewards = []
        episode_benchmarks = []
        
        print(f"Starting LOOCV expanded features training for {EPISODES} episodes...")
        print(f"Training on {len(self.training_benchmarks)} benchmarks, excluding '{self.exclude_benchmark}'")
        print("This model uses 7 flags + 279 static features for better performance prediction")
        
        for episode in range(EPISODES):
            # Select benchmark (round-robin)
            benchmark_name = self.training_benchmarks[episode % len(self.training_benchmarks)]
            
            # Reset environment with specific benchmark
            state = self.loocv_env.reset(benchmark_name)
            total_reward = 0
            
            while True:
                action = self.agent.act(state, training=True)
                next_state, reward, done, info = self.loocv_env.step(action)
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
                      f"Successful: {successful_benchmarks}/{len(self.training_benchmarks)}")
        
        # Save LOOCV model
        save_path = os.path.join(os.path.dirname(__file__), SAVE_PATH)
        self.agent.save(save_path)
        print(f"LOOCV expanded features model saved to: {save_path}")
        
        # Print final results
        print("\n" + "="*80)
        print("LOOCV EXPANDED FEATURES TRAINING RESULTS")
        print("="*80)
        print(f"Excluded benchmark: {self.exclude_benchmark}")
        print(f"Training benchmarks: {len(self.training_benchmarks)}")
        print(f"Total episodes: {EPISODES}")
        print(f"Average reward: {np.mean(episode_rewards):.4f}")
        print(f"Average best improvement: {np.mean(list(best_improvements.values())):.4f}")
        
        successful_benchmarks = len([imp for imp in best_improvements.values() if imp > 0])
        print(f"Successful optimizations: {successful_benchmarks}/{len(self.training_benchmarks)} ({successful_benchmarks/len(self.training_benchmarks)*100:.1f}%)")
        
        print(f"\nBest improvements by benchmark:")
        for name, improvement in sorted(best_improvements.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {improvement:.4f}")
        
        return best_improvements, episode_rewards
    
    def evaluate_on_test_benchmark(self, num_episodes: int = 10):
        """Evaluate the trained model on the excluded benchmark"""
        print(f"\nEvaluating trained model on excluded benchmark: {self.exclude_benchmark}")
        
        test_results = self.loocv_env.evaluate_on_test_benchmark(
            self.agent, 
            self.exclude_benchmark, 
            num_episodes=num_episodes
        )
        
        print(f"\nLOOCV Test Results for '{self.exclude_benchmark}':")
        print(f"Average reward: {test_results['avg_reward']:.4f}")
        print(f"Average improvement: {test_results['avg_improvement']:.4f}")
        print(f"Best improvement: {test_results['best_improvement']:.4f}")
        
        # Save test results
        test_results_df = pd.DataFrame({
            'metric': ['avg_reward', 'avg_improvement', 'best_improvement'],
            'value': [test_results['avg_reward'], test_results['avg_improvement'], test_results['best_improvement']]
        })
        test_results_csv = os.path.join(os.path.dirname(__file__), "loocv_test_results.csv")
        test_results_df.to_csv(test_results_csv, index=False)
        print(f"Test results saved to: {test_results_csv}")
        
        return test_results
    
    def save_training_results(self, best_improvements, episode_rewards):
        """Save training results to CSV and create plots"""
        # Save training results
        results_df = pd.DataFrame({
            'benchmark': list(best_improvements.keys()),
            'best_improvement': list(best_improvements.values())
        })
        results_csv = os.path.join(os.path.dirname(__file__), RESULTS_CSV)
        results_df.to_csv(results_csv, index=False)
        print(f"Training results saved to: {results_csv}")
        
        # Create training progress plot
        plt.figure(figsize=(12, 6))
        
        # Plot episode rewards
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards)
        plt.title('Training Progress - Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True, alpha=0.3)
        
        # Plot benchmark improvements
        plt.subplot(1, 2, 2)
        sorted_improvements = sorted(best_improvements.items(), key=lambda x: x[1], reverse=True)
        benchmarks, improvements = zip(*sorted_improvements)
        bars = plt.bar(range(len(benchmarks)), improvements, color='skyblue', alpha=0.8)
        plt.xlabel('Benchmark')
        plt.ylabel('Best Improvement')
        plt.title('Best Improvements by Benchmark (Training)')
        plt.xticks(range(len(benchmarks)), benchmarks, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__), PLOT_PNG)
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"Training progress plot saved to: {plot_path}")
        plt.close()

def main():
    trainer = LOOCVExpandedFeaturesTrainer(exclude_benchmark="security_sha")
    
    print("Loading and preprocessing data with LOOCV...")
    trainer.load_and_preprocess_data()
    
    print("Setting up LOOCV environment...")
    trainer.setup_loocv_environment()
    
    print("Starting LOOCV expanded features training...")
    best_improvements, episode_rewards = trainer.train_loocv_model()
    
    print("Saving training results...")
    trainer.save_training_results(best_improvements, episode_rewards)
    
    print("Evaluating on excluded benchmark...")
    test_results = trainer.evaluate_on_test_benchmark(num_episodes=10)
    
    print("\nLOOCV expanded features training and evaluation completed successfully!")
    print(f"Model saved as: {SAVE_PATH}")
    print(f"Training results: {RESULTS_CSV}")
    print(f"Test results: loocv_test_results.csv")

if __name__ == "__main__":
    main() 