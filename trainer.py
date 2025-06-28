import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
import json
import numpy as np
from data_preprocess import DataPreprocessor
from comp_env import CompilerEnvironment
from dqn_agent import DQNAgent

logger = logging.getLogger(__name__)

class CompilerOptimizationTrainer:
    """Main trainer class for RL compiler optimization"""
    
    def __init__(self, data_path: str = "exec_times.csv"):
        self.data_path = data_path
        self.preprocessor = DataPreprocessor(data_path)
        self.benchmark_data = None
        self.agents = {}
        self.environments = {}
        self.training_stats = defaultdict(list)
        
    def setup(self):
        """Setup data and create agents/environments"""
        logger.info("Setting up trainer...")
        
        # Load and preprocess data
        self.benchmark_data = self.preprocessor.preprocess_data()
        
        if not self.benchmark_data:
            raise ValueError("No benchmark data available after preprocessing!")
        
        # Create agents and environments for each benchmark
        for benchmark_name in self.benchmark_data.keys():
            logger.info(f"Setting up agent for benchmark: {benchmark_name}")
            
            # Create environment
            env = CompilerEnvironment(self.benchmark_data, benchmark_name)
            self.environments[benchmark_name] = env
            
            # Create agent
            state_size = env.observation_space.shape[0]
            
            agent = DQNAgent(
                state_size=state_size,
                action_size=7,  # 7 compiler flags
                lr=0.001,
                gamma=0.95,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.995,
                memory_size=10000,
                batch_size=32,
                target_update=100
            )
            
            self.agents[benchmark_name] = agent
            
        logger.info(f"Setup complete for {len(self.benchmark_data)} benchmarks")
    
    def train_single_benchmark(self, benchmark_name: str, episodes: int = 1000, 
                             save_frequency: int = 100, verbose: bool = True):
        """Train agent on a single benchmark"""
        
        if benchmark_name not in self.agents:
            logger.error(f"No agent found for benchmark: {benchmark_name}")
            return
        
        agent = self.agents[benchmark_name]
        env = self.environments[benchmark_name]
        
        logger.info(f"Training {benchmark_name} for {episodes} episodes...")
        
        episode_rewards = []
        episode_improvements = []
        episode_losses = []
        best_configs = []
        
        start_time = time.time()
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            episode_loss = []
            
            while True:
                # Choose action
                action = agent.act(state, training=True)
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                loss = agent.replay()
                if loss is not None:
                    episode_loss.append(loss)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Track statistics
            episode_rewards.append(total_reward)
            episode_improvements.append(info['improvement'])
            if episode_loss:
                episode_losses.append(np.mean(episode_loss))
            
            # Track best configuration found
            if info['improvement'] > 0:  # Only if we found an improvement
                best_configs.append({
                    'episode': episode,
                    'config': info['config'].tolist(),
                    'improvement': info['improvement'],
                    'execution_time': info['execution_time']
                })
            
            # Logging
            if verbose and (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_improvement = np.mean(episode_improvements[-50:])
                current_eps = agent.epsilon
                
                logger.info(f"  Episode {episode + 1}/{episodes} - "
                           f"Avg Reward: {avg_reward:.2f}, "
                           f"Avg Improvement: {avg_improvement:.4f}, "
                           f"Epsilon: {current_eps:.3f}")
            
            # Save model periodically
            if (episode + 1) % save_frequency == 0:
                model_path = f"models/{benchmark_name}_episode_{episode + 1}.pth"
                os.makedirs("models", exist_ok=True)
                agent.save(model_path)
        
        training_time = time.time() - start_time
        
        # Store training statistics
        self.training_stats[benchmark_name] = {
            'episode_rewards': episode_rewards,
            'episode_improvements': episode_improvements,
            'episode_losses': episode_losses,
            'best_configs': best_configs,
            'training_time': training_time,
            'final_epsilon': agent.epsilon
        }
        
        logger.info(f"Training completed for {benchmark_name} in {training_time:.2f} seconds")
        
        # Save final model
        final_model_path = f"models/{benchmark_name}_final.pth"
        os.makedirs("models", exist_ok=True)
        agent.save(final_model_path)
        
        return self.training_stats[benchmark_name]
    
    def train_all_benchmarks(self, episodes: int = 1000, save_frequency: int = 100):
        """Train agents on all benchmarks"""
        logger.info(f"Training all {len(self.benchmark_data)} benchmarks...")
        
        for benchmark_name in self.benchmark_data.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training benchmark: {benchmark_name}")
            logger.info(f"{'='*60}")
            
            try:
                self.train_single_benchmark(
                    benchmark_name, episodes, save_frequency, verbose=True
                )
            except Exception as e:
                logger.error(f"Error training {benchmark_name}: {e}")
                continue
        
        logger.info("Training completed for all benchmarks")
    
    def evaluate_benchmark(self, benchmark_name: str, episodes: int = 100):
        """Evaluate trained agent on benchmark"""
        if benchmark_name not in self.agents:
            logger.error(f"No trained agent found for benchmark: {benchmark_name}")
            return None
        
        agent = self.agents[benchmark_name]
        env = self.environments[benchmark_name]
        
        logger.info(f"Evaluating {benchmark_name} for {episodes} episodes...")
        
        results = []
        
        for episode in range(episodes):
            state = env.reset()
            
            while True:
                # Use greedy policy (no exploration)
                action = agent.act(state, training=False)
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                
                if done:
                    results.append({
                        'episode': episode,
                        'config': info['config'].tolist(),
                        'execution_time': info['execution_time'],
                        'improvement': info['improvement'],
                        'reward': reward
                    })
                    break
        
        # Calculate statistics
        improvements = [r['improvement'] for r in results]
        execution_times = [r['execution_time'] for r in results]
        
        evaluation_stats = {
            'mean_improvement': np.mean(improvements),
            'std_improvement': np.std(improvements),
            'best_improvement': np.max(improvements),
            'mean_execution_time': np.mean(execution_times),
            'best_execution_time': np.min(execution_times),
            'baseline_time': env.baseline_time,
            'best_known_time': env.best_time,
            'results': results
        }
        
        logger.info(f"Evaluation results for {benchmark_name}:")
        logger.info(f"  Mean improvement: {evaluation_stats['mean_improvement']:.4f}")
        logger.info(f"  Best improvement: {evaluation_stats['best_improvement']:.4f}")
        logger.info(f"  Best execution time: {evaluation_stats['best_execution_time']:.6f}")
        logger.info(f"  Baseline time: {evaluation_stats['baseline_time']:.6f}")
        
        return evaluation_stats
    
    def visualize_training_progress(self, benchmark_name: str = None, save_plots: bool = True):
        """Visualize training progress"""
        if benchmark_name:
            benchmarks_to_plot = [benchmark_name]
        else:
            benchmarks_to_plot = list(self.training_stats.keys())
        
        for benchmark in benchmarks_to_plot:
            if benchmark not in self.training_stats:
                logger.warning(f"No training stats found for {benchmark}")
                continue
            
            stats = self.training_stats[benchmark]
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress: {benchmark}', fontsize=16)
            
            # Plot 1: Episode Rewards
            axes[0, 0].plot(stats['episode_rewards'])
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].grid(True)
            
            # Plot 2: Episode Improvements
            axes[0, 1].plot(stats['episode_improvements'])
            axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Performance Improvements')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Improvement Ratio')
            axes[0, 1].grid(True)
            
            # Plot 3: Training Losses
            if stats['episode_losses']:
                axes[1, 0].plot(stats['episode_losses'])
                axes[1, 0].set_title('Training Loss')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].grid(True)
            else:
                axes[1, 0].text(0.5, 0.5, 'No Loss Data', ha='center', va='center')
                axes[1, 0].set_title('Training Loss')
            
            # Plot 4: Best Configurations Over Time
            if stats['best_configs']:
                episodes = [c['episode'] for c in stats['best_configs']]
                improvements = [c['improvement'] for c in stats['best_configs']]
                axes[1, 1].scatter(episodes, improvements, alpha=0.6)
                axes[1, 1].set_title('Best Configurations Found')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Improvement')
                axes[1, 1].grid(True)
            else:
                axes[1, 1].text(0.5, 0.5, 'No Improvements Found', ha='center', va='center')
                axes[1, 1].set_title('Best Configurations Found')
            
            plt.tight_layout()
            
            if save_plots:
                os.makedirs("plots", exist_ok=True)
                plt.savefig(f"plots/{benchmark}_training_progress.png", dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def save_results(self, filepath: str = "results/training_results.json"):
        """Save training results to file"""
        os.makedirs("results", exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_to_save = {}
        for benchmark, stats in self.training_stats.items():
            results_to_save[benchmark] = {
                'episode_rewards': [float(x) for x in stats['episode_rewards']],
                'episode_improvements': [float(x) for x in stats['episode_improvements']],
                'episode_losses': [float(x) for x in stats['episode_losses']],
                'best_configs': stats['best_configs'],
                'training_time': stats['training_time'],
                'final_epsilon': stats['final_epsilon']
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str = "results/training_results.json"):
        """Load training results from file"""
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            # Convert back to numpy arrays if needed
            for benchmark, stats in results.items():
                self.training_stats[benchmark] = stats
            
            logger.info(f"Results loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return False
    
    def get_summary_report(self):
        """Generate summary report of all training results"""
        if not self.training_stats:
            logger.warning("No training statistics available")
            return None
        
        report = {
            'total_benchmarks': len(self.training_stats),
            'benchmark_results': {}
        }
        
        for benchmark, stats in self.training_stats.items():
            benchmark_summary = {
                'episodes_trained': len(stats['episode_rewards']),
                'final_avg_reward': np.mean(stats['episode_rewards'][-50:]) if len(stats['episode_rewards']) >= 50 else np.mean(stats['episode_rewards']),
                'best_improvement_found': max([c['improvement'] for c in stats['best_configs']]) if stats['best_configs'] else 0,
                'num_improvements_found': len(stats['best_configs']),
                'training_time_minutes': stats['training_time'] / 60,
                'final_epsilon': stats['final_epsilon']
            }
            
            report['benchmark_results'][benchmark] = benchmark_summary
        
        # Overall statistics
        all_improvements = []
        for stats in self.training_stats.values():
            if stats['best_configs']:
                all_improvements.extend([c['improvement'] for c in stats['best_configs']])
        
        if all_improvements:
            report['overall_stats'] = {
                'benchmarks_with_improvements': sum(1 for stats in self.training_stats.values() if stats['best_configs']),
                'total_improvements_found': len(all_improvements),
                'mean_improvement': np.mean(all_improvements),
                'best_overall_improvement': max(all_improvements)
            }
        
        return report