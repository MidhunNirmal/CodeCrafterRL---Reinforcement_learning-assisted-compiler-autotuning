import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import random

# Import the original data preprocessor (assumes it's in parent directory)
import sys
sys.path.append('..')
from data_preprocess import DataPreprocessor

from .unified_dqn_agent import UnifiedDQNAgent
from .unified_environment import UnifiedCompilerEnvironment
from .benchmark_encoder import BenchmarkEncoder

logger = logging.getLogger(__name__)

class UnifiedCompilerOptimizationTrainer:
    """
    Unified trainer that trains a single DQN model on all benchmarks.
    Instead of separate models per benchmark, this trains one model that can handle all.
    """
    
    def __init__(self, data_path: str = "exec_times.csv"):
        self.data_path = data_path
        self.preprocessor = DataPreprocessor(data_path)
        self.benchmark_data = None
        self.agent = None
        self.environment = None
        self.benchmark_encoder = None
        self.training_stats = defaultdict(list)
        self.unified_stats = {
            'total_episodes': 0,
            'total_steps': 0,
            'losses': [],
            'rewards': [],
            'improvements': [],
            'benchmark_distribution': defaultdict(int)
        }
        
    def setup(self, 
              benchmark_names: Optional[List[str]] = None,
              use_benchmark_cycling: bool = True,
              use_one_hot_encoding: bool = True):
        """Setup unified trainer with data and create unified agent/environment"""
        logger.info("Setting up unified trainer...")
        
        # Load and preprocess data
        self.benchmark_data = self.preprocessor.preprocess_data()
        
        if not self.benchmark_data:
            raise ValueError("No benchmark data available after preprocessing!")
        
        # Filter benchmarks if specified
        if benchmark_names:
            filtered_data = {name: data for name, data in self.benchmark_data.items() 
                           if name in benchmark_names}
            if not filtered_data:
                logger.warning("No valid benchmarks found, using all available")
            else:
                self.benchmark_data = filtered_data
        
        benchmark_names_list = list(self.benchmark_data.keys())
        logger.info(f"Using {len(benchmark_names_list)} benchmarks: {benchmark_names_list}")
        
        # Initialize benchmark encoder
        self.benchmark_encoder = BenchmarkEncoder(benchmark_names_list)
        
        # Create unified environment
        self.environment = UnifiedCompilerEnvironment(
            benchmark_data=self.benchmark_data,
            benchmark_names=benchmark_names_list,
            use_benchmark_cycling=use_benchmark_cycling,
            use_one_hot_encoding=use_one_hot_encoding
        )
        
        # Create unified agent
        state_size = self.benchmark_encoder.get_state_size(use_one_hot=use_one_hot_encoding)
        
        self.agent = UnifiedDQNAgent(
            state_size=state_size,
            action_size=7,
            lr=0.0005,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.9995,  # Slower decay for unified model
            memory_size=100000,    # Larger memory for multiple benchmarks
            batch_size=128,        # Larger batch size
            target_update=2000,    # Less frequent updates
            benchmark_names=benchmark_names_list
        )
        
        logger.info(f"Unified setup complete:")
        logger.info(f"  Benchmarks: {len(benchmark_names_list)}")
        logger.info(f"  State size: {state_size}")
        logger.info(f"  Agent memory: {self.agent.memory_size}")
        logger.info(f"  Environment cycling: {use_benchmark_cycling}")
    
    def train_unified_model(self, 
                           total_episodes: int = 10000,
                           save_frequency: int = 1000,
                           evaluation_frequency: int = 2000,
                           verbose: bool = True):
        """
        Train the unified model on all benchmarks.
        
        Args:
            total_episodes: Total number of episodes to train
            save_frequency: How often to save the model
            evaluation_frequency: How often to run evaluation
            verbose: Whether to print progress
        """
        logger.info(f"Starting unified training for {total_episodes} episodes...")
        
        start_time = time.time()
        episode_rewards = []
        episode_improvements = []
        episode_losses = []
        benchmark_episodes = defaultdict(int)
        
                 for episode in range(total_episodes):
             # Reset environment (it will select a benchmark automatically)
             if self.environment is None or self.agent is None:
                 raise ValueError("Environment and agent must be initialized before training")
             
             state = self.environment.reset()
             episode_reward = 0
             episode_loss = []
             step_count = 0
             
             # Get current benchmark for tracking
             current_benchmark = self.environment.get_current_benchmark() or "unknown"
             benchmark_episodes[current_benchmark] += 1
             self.unified_stats['benchmark_distribution'][current_benchmark] += 1
            
            while True:
                # Choose action
                action = self.agent.act(state, training=True, benchmark_name=current_benchmark)
                
                # Take step
                next_state, reward, done, info = self.environment.step(action)
                
                # Store experience with benchmark information
                self.agent.remember(state, action, reward, next_state, done, current_benchmark)
                
                # Train agent
                loss_info = self.agent.replay()
                if loss_info is not None:
                    episode_loss.append(loss_info['total_loss'])
                
                                 # Update performance tracking
                 if current_benchmark != "unknown":
                     self.agent.update_benchmark_performance(current_benchmark, info['improvement'])
                
                state = next_state
                episode_reward += reward
                step_count += 1
                
                if done:
                    break
            
            # Track episode statistics
            episode_rewards.append(episode_reward)
            episode_improvements.append(info['improvement'])
            
            if episode_loss:
                avg_episode_loss = np.mean(episode_loss)
                episode_losses.append(avg_episode_loss)
                self.unified_stats['losses'].append(avg_episode_loss)
            
            self.unified_stats['rewards'].append(episode_reward)
            self.unified_stats['improvements'].append(info['improvement'])
            self.unified_stats['total_episodes'] += 1
            self.unified_stats['total_steps'] += step_count
            
            # Logging
            if verbose and (episode + 1) % 100 == 0:
                recent_rewards = np.mean(episode_rewards[-100:])
                recent_improvements = np.mean(episode_improvements[-100:])
                recent_losses = np.mean(episode_losses[-50:]) if episode_losses else 0
                
                logger.info(f"Episode {episode + 1}/{total_episodes}")
                logger.info(f"  Avg Reward (last 100): {recent_rewards:.2f}")
                logger.info(f"  Avg Improvement (last 100): {recent_improvements:.4f}")
                logger.info(f"  Avg Loss (last 50): {recent_losses:.4f}")
                logger.info(f"  Epsilon: {self.agent.epsilon:.3f}")
                logger.info(f"  Current benchmark: {current_benchmark}")
                
                # Show benchmark distribution
                if (episode + 1) % 500 == 0:
                    logger.info("Benchmark episode distribution:")
                    total_eps = sum(benchmark_episodes.values())
                    for bench, count in sorted(benchmark_episodes.items()):
                        percentage = count / total_eps * 100 if total_eps > 0 else 0
                        logger.info(f"    {bench}: {count} episodes ({percentage:.1f}%)")
            
            # Save model periodically
            if (episode + 1) % save_frequency == 0:
                model_path = f"unified_models/unified_model_episode_{episode + 1}.pth"
                os.makedirs("unified_models", exist_ok=True)
                self.agent.save(model_path)
                
                # Save training statistics
                self.save_unified_results(f"unified_models/training_stats_episode_{episode + 1}.json")
            
            # Run evaluation periodically
            if (episode + 1) % evaluation_frequency == 0:
                logger.info(f"Running evaluation at episode {episode + 1}...")
                eval_results = self.evaluate_unified_model(episodes_per_benchmark=50)
                logger.info("Evaluation results:")
                for benchmark, stats in eval_results.items():
                    logger.info(f"  {benchmark}: {stats['mean_improvement']:.4f} improvement")
        
        training_time = time.time() - start_time
        
        # Store final training statistics
        self.training_stats = {
            'episode_rewards': episode_rewards,
            'episode_improvements': episode_improvements,
            'episode_losses': episode_losses,
            'benchmark_episodes': dict(benchmark_episodes),
            'training_time': training_time,
            'total_episodes': total_episodes,
            'final_epsilon': self.agent.epsilon
        }
        
        logger.info(f"Unified training completed in {training_time:.2f} seconds")
        logger.info(f"Total episodes: {total_episodes}")
        logger.info(f"Final epsilon: {self.agent.epsilon:.3f}")
        
        # Save final model
        final_model_path = "unified_models/unified_model_final.pth"
        os.makedirs("unified_models", exist_ok=True)
        self.agent.save(final_model_path)
        
        return self.training_stats
    
    def evaluate_unified_model(self, 
                             episodes_per_benchmark: int = 100,
                             deterministic: bool = True) -> Dict:
        """
        Evaluate the unified model on all benchmarks.
        
        Args:
            episodes_per_benchmark: Number of episodes to run per benchmark
            deterministic: Whether to use deterministic (greedy) policy
            
        Returns:
            Dictionary with evaluation results per benchmark
        """
        logger.info(f"Evaluating unified model on {len(self.benchmark_data)} benchmarks...")
        
        evaluation_results = {}
        
        for benchmark_name in self.benchmark_data.keys():
            logger.info(f"Evaluating on {benchmark_name}...")
            
            benchmark_results = []
            
            for episode in range(episodes_per_benchmark):
                # Force environment to use specific benchmark
                self.environment.force_benchmark(benchmark_name)
                state = self.environment.reset(benchmark_name)
                
                episode_reward = 0
                step_count = 0
                
                while True:
                    # Use deterministic policy for evaluation
                    action = self.agent.act(state, training=not deterministic, benchmark_name=benchmark_name)
                    next_state, reward, done, info = self.environment.step(action)
                    
                    state = next_state
                    episode_reward += reward
                    step_count += 1
                    
                    if done:
                        benchmark_results.append({
                            'episode': episode,
                            'reward': episode_reward,
                            'improvement': info['improvement'],
                            'execution_time': info['execution_time'],
                            'config': info['config'].tolist(),
                            'steps': step_count
                        })
                        break
            
            # Calculate benchmark statistics
            improvements = [r['improvement'] for r in benchmark_results]
            rewards = [r['reward'] for r in benchmark_results]
            execution_times = [r['execution_time'] for r in benchmark_results]
            
            evaluation_results[benchmark_name] = {
                'episodes': episodes_per_benchmark,
                'mean_improvement': np.mean(improvements),
                'std_improvement': np.std(improvements),
                'best_improvement': max(improvements),
                'mean_reward': np.mean(rewards),
                'mean_execution_time': np.mean(execution_times),
                'best_execution_time': min(execution_times),
                'baseline_time': self.benchmark_data[benchmark_name]['baseline_time'],
                'results': benchmark_results
            }
            
            logger.info(f"  {benchmark_name} evaluation complete:")
            logger.info(f"    Mean improvement: {evaluation_results[benchmark_name]['mean_improvement']:.4f}")
            logger.info(f"    Best improvement: {evaluation_results[benchmark_name]['best_improvement']:.4f}")
        
        return evaluation_results
    
    def visualize_unified_training(self, save_plots: bool = True):
        """Visualize unified training progress"""
        if not self.training_stats:
            logger.warning("No training statistics available for visualization")
            return
        
        # Create comprehensive training visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Unified Model Training Progress', fontsize=16)
        
        # Plot 1: Episode Rewards
        axes[0, 0].plot(self.training_stats['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # Plot 2: Episode Improvements
        axes[0, 1].plot(self.training_stats['episode_improvements'])
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Performance Improvements')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Improvement Ratio')
        axes[0, 1].grid(True)
        
        # Plot 3: Training Losses
        if self.training_stats['episode_losses']:
            axes[0, 2].plot(self.training_stats['episode_losses'])
            axes[0, 2].set_title('Training Loss')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].grid(True)
        else:
            axes[0, 2].text(0.5, 0.5, 'No Loss Data', ha='center', va='center')
            axes[0, 2].set_title('Training Loss')
        
        # Plot 4: Benchmark Distribution
        benchmark_episodes = self.training_stats['benchmark_episodes']
        if benchmark_episodes:
            benchmarks = list(benchmark_episodes.keys())
            episodes = list(benchmark_episodes.values())
            
            axes[1, 0].bar(range(len(benchmarks)), episodes)
            axes[1, 0].set_title('Episodes per Benchmark')
            axes[1, 0].set_xlabel('Benchmark')
            axes[1, 0].set_ylabel('Episodes')
            axes[1, 0].set_xticks(range(len(benchmarks)))
            axes[1, 0].set_xticklabels(benchmarks, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Cumulative Performance
        cumulative_improvements = np.cumsum(self.training_stats['episode_improvements'])
        axes[1, 1].plot(cumulative_improvements)
        axes[1, 1].set_title('Cumulative Improvements')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Cumulative Improvement')
        axes[1, 1].grid(True)
        
        # Plot 6: Learning Rate and Epsilon
        if hasattr(self.agent, 'losses') and self.agent.losses:
            # Plot epsilon decay
            episodes = range(len(self.training_stats['episode_rewards']))
            epsilons = [max(self.agent.epsilon_min, 
                           self.agent.epsilon * (self.agent.epsilon_decay ** ep)) 
                       for ep in episodes]
            
            ax1 = axes[1, 2]
            ax1.plot(episodes, epsilons, 'b-', label='Epsilon')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Epsilon', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.grid(True)
            
            # Add learning rate on secondary y-axis if available
            if hasattr(self.agent, 'optimizer'):
                ax2 = ax1.twinx()
                lr = self.agent.optimizer.param_groups[0]['lr']
                ax2.axhline(y=lr, color='r', linestyle='--', label='Learning Rate')
                ax2.set_ylabel('Learning Rate', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
        
        plt.tight_layout()
        
        if save_plots:
            os.makedirs("unified_plots", exist_ok=True)
            plt.savefig("unified_plots/unified_training_progress.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_unified_results(self, filepath: str = "unified_models/unified_training_results.json"):
        """Save unified training results"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare results for JSON serialization
        results_to_save = {
            'training_stats': self.training_stats,
            'unified_stats': dict(self.unified_stats),
            'benchmark_stats': self.environment.get_benchmark_stats() if self.environment else {},
            'agent_stats': self.agent.get_network_stats() if self.agent else {},
            'benchmark_performance': self.agent.get_benchmark_performance_stats() if self.agent else {}
        }
        
        # Convert numpy arrays to lists
        for key, value in results_to_save['training_stats'].items():
            if isinstance(value, np.ndarray):
                results_to_save['training_stats'][key] = value.tolist()
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                results_to_save['training_stats'][key] = [v.tolist() for v in value]
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"Unified results saved to {filepath}")
    
    def load_unified_results(self, filepath: str):
        """Load unified training results"""
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            self.training_stats = results.get('training_stats', {})
            self.unified_stats = results.get('unified_stats', {})
            
            logger.info(f"Unified results loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading unified results: {e}")
            return False
    
    def get_unified_summary_report(self) -> Dict:
        """Generate comprehensive summary report of unified training"""
        if not self.training_stats:
            logger.warning("No training statistics available")
            return {}
        
        # Get benchmark performance from agent
        benchmark_performance = self.agent.get_benchmark_performance_stats() if self.agent else {}
        
        # Overall statistics
        report = {
            'total_benchmarks': len(self.benchmark_data) if self.benchmark_data else 0,
            'total_episodes': self.training_stats.get('total_episodes', 0),
            'training_time_hours': self.training_stats.get('training_time', 0) / 3600,
            'final_epsilon': self.training_stats.get('final_epsilon', 0),
            'model_parameters': self.agent.get_network_stats()['total_parameters'] if self.agent else 0
        }
        
        # Performance summary
        if self.training_stats.get('episode_improvements'):
            improvements = self.training_stats['episode_improvements']
            report['performance_summary'] = {
                'mean_improvement': np.mean(improvements),
                'std_improvement': np.std(improvements),
                'best_improvement': max(improvements),
                'final_100_mean': np.mean(improvements[-100:]) if len(improvements) >= 100 else np.mean(improvements),
                'positive_improvement_rate': sum(1 for imp in improvements if imp > 0) / len(improvements)
            }
        
        # Benchmark-specific performance
        report['benchmark_performance'] = benchmark_performance
        
        # Learning statistics
        if self.training_stats.get('episode_losses'):
            losses = self.training_stats['episode_losses']
            report['learning_stats'] = {
                'final_loss': losses[-1] if losses else 0,
                'mean_loss': np.mean(losses),
                'loss_trend': np.mean(losses[-100:]) - np.mean(losses[:100]) if len(losses) >= 200 else 0
            }
        
        return report