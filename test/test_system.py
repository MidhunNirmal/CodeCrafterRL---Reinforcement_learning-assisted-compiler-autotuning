import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
import json
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Dict, List, Tuple, Optional
from data_preprocess import DataPreprocessor
from comp_env import CompilerEnvironment
from dqn_agent import DQNAgent
from trainer import CompilerOptimizationTrainer

logger = logging.getLogger(__name__)

class CompilerOptimizationTester:
    """Test system for compiler optimization with dataset validation"""
    
    def __init__(self, data_path: str = "../data/exec_times.csv"):
        self.data_path = data_path
        self.preprocessor = DataPreprocessor(data_path)
        self.benchmark_data = None
        self.trainer = None
        self.test_results = defaultdict(list)
        
        # Compiler flag names for display
        self.flag_names = [
            'funsafe-math-optimizations',
            'fno-guess-branch-probability',
            'fno-ivopts',
            'fno-tree-loop-optimize',
            'fno-inline-functions',
            'funroll-all-loops',
            'O2'
        ]
    
    def setup(self, load_trained_models: bool = True):
        """Setup the testing environment"""
        logger.info("Setting up test environment...")
        
        # Load and preprocess data
        self.benchmark_data = self.preprocessor.preprocess_data()
        
        if not self.benchmark_data:
            raise ValueError("No benchmark data available!")
        
        # Initialize trainer
        self.trainer = CompilerOptimizationTrainer(self.data_path)
        self.trainer.benchmark_data = self.benchmark_data
        
        # Create agents and environments
        self.trainer.agents = {}
        self.trainer.environments = {}
        
        for benchmark_name in self.benchmark_data.keys():
            # Create environment
            env = CompilerEnvironment(self.benchmark_data, benchmark_name)
            self.trainer.environments[benchmark_name] = env
            
            # Create agent
            state_size = env.observation_space.shape[0]
            agent = DQNAgent(state_size=state_size, action_size=7)
            
            # Load trained model if available
            if load_trained_models:
                model_path = f"models/{benchmark_name}_final.pth"
                if os.path.exists(model_path):
                    try:
                        agent.load(model_path)
                        logger.info(f"Loaded trained model for {benchmark_name}")
                    except Exception as e:
                        logger.warning(f"Could not load model for {benchmark_name}: {e}")
                else:
                    logger.warning(f"No trained model found for {benchmark_name} at {model_path}")
            
            self.trainer.agents[benchmark_name] = agent
        
        logger.info(f"Setup complete for {len(self.benchmark_data)} benchmarks")
    
    def get_random_test_samples(self, benchmark_name: str, n_samples: int = 10) -> List[Dict]:
        """Get random test samples from the dataset"""
        if benchmark_name not in self.benchmark_data:
            logger.error(f"Benchmark {benchmark_name} not found")
            return []
        
        data = self.benchmark_data[benchmark_name]
        
        # Get random indices
        n_available = len(data['y_raw'])
        if n_samples > n_available:
            n_samples = n_available
            logger.warning(f"Requested {n_samples} samples but only {n_available} available")
        
        random_indices = np.random.choice(n_available, n_samples, replace=False)
        
        samples = []
        for idx in random_indices:
            # Get the configuration (first 7 features are compiler flags)
            config = data['X_raw'][idx][:7].astype(int)
            execution_time = data['y_raw'][idx]
            
            samples.append({
                'index': int(idx),
                'config': config.tolist(),
                'execution_time': float(execution_time),
                'config_str': self._config_to_string(config)
            })
        
        return samples
    
    def get_best_worst_samples(self, benchmark_name: str, n_each: int = 5) -> Dict:
        """Get best and worst performing samples from dataset"""
        if benchmark_name not in self.benchmark_data:
            logger.error(f"Benchmark {benchmark_name} not found")
            return {}
        
        data = self.benchmark_data[benchmark_name]
        execution_times = data['y_raw']
        
        # Get indices of best and worst times
        best_indices = np.argsort(execution_times)[:n_each]
        worst_indices = np.argsort(execution_times)[-n_each:]
        
        best_samples = []
        worst_samples = []
        
        for idx in best_indices:
            config = data['X_raw'][idx][:7].astype(int)
            best_samples.append({
                'index': int(idx),
                'config': config.tolist(),
                'execution_time': float(execution_times[idx]),
                'config_str': self._config_to_string(config)
            })
        
        for idx in worst_indices:
            config = data['X_raw'][idx][:7].astype(int)
            worst_samples.append({
                'index': int(idx),
                'config': config.tolist(),
                'execution_time': float(execution_times[idx]),
                'config_str': self._config_to_string(config)
            })
        
        return {
            'best_samples': best_samples,
            'worst_samples': worst_samples
        }
    
    def test_agent_vs_dataset(self, benchmark_name: str, n_tests: int = 100) -> Dict:
        """Test trained agent against dataset samples"""
        if benchmark_name not in self.trainer.agents:
            logger.error(f"No agent found for benchmark {benchmark_name}")
            return {}
        
        agent = self.trainer.agents[benchmark_name]
        env = self.trainer.environments[benchmark_name]
        data = self.benchmark_data[benchmark_name]
        
        logger.info(f"Testing agent vs dataset for {benchmark_name} with {n_tests} tests...")
        
        results = {
            'dataset_samples': [],
            'agent_recommendations': [],
            'comparisons': []
        }
        
        # Get random dataset samples
        test_samples = self.get_random_test_samples(benchmark_name, n_tests)
        
        for i, sample in enumerate(test_samples):
            # Get dataset configuration and performance
            dataset_config = np.array(sample['config'])
            dataset_time = sample['execution_time']
            
            # Get agent's recommendation
            state = env.reset()
            agent_config = agent.act(state, training=False)
            
            # Get predicted performance for agent's configuration
            agent_time = env._get_execution_time(agent_config)
            
            # Calculate improvements
            baseline_time = data['baseline_time']
            dataset_improvement = (baseline_time - dataset_time) / baseline_time
            agent_improvement = (baseline_time - agent_time) / baseline_time
            
            comparison = {
                'test_id': i,
                'dataset_config': dataset_config.tolist(),
                'dataset_config_str': self._config_to_string(dataset_config),
                'dataset_time': dataset_time,
                'dataset_improvement': dataset_improvement,
                'agent_config': agent_config.tolist(),
                'agent_config_str': self._config_to_string(agent_config),
                'agent_time': agent_time,
                'agent_improvement': agent_improvement,
                'agent_better': agent_time < dataset_time,
                'improvement_diff': agent_improvement - dataset_improvement
            }
            
            results['dataset_samples'].append(sample)
            results['agent_recommendations'].append({
                'config': agent_config.tolist(),
                'config_str': self._config_to_string(agent_config),
                'predicted_time': agent_time,
                'predicted_improvement': agent_improvement
            })
            results['comparisons'].append(comparison)
        
        # Calculate summary statistics
        comparisons = results['comparisons']
        agent_wins = sum(1 for c in comparisons if c['agent_better'])
        win_rate = agent_wins / len(comparisons)
        
        avg_dataset_improvement = np.mean([c['dataset_improvement'] for c in comparisons])
        avg_agent_improvement = np.mean([c['agent_improvement'] for c in comparisons])
        avg_improvement_diff = np.mean([c['improvement_diff'] for c in comparisons])
        
        results['summary'] = {
            'total_tests': len(comparisons),
            'agent_wins': agent_wins,
            'win_rate': win_rate,
            'avg_dataset_improvement': avg_dataset_improvement,
            'avg_agent_improvement': avg_agent_improvement,
            'avg_improvement_difference': avg_improvement_diff,
            'baseline_time': baseline_time,
            'best_known_time': data['best_time']
        }
        
        logger.info(f"Test results for {benchmark_name}:")
        logger.info(f"  Agent win rate: {win_rate:.2%}")
        logger.info(f"  Avg dataset improvement: {avg_dataset_improvement:.4f}")
        logger.info(f"  Avg agent improvement: {avg_agent_improvement:.4f}")
        logger.info(f"  Avg improvement difference: {avg_improvement_diff:.4f}")
        
        return results
    
    def test_specific_configurations(self, benchmark_name: str, configs: List[List[int]]) -> Dict:
        """Test specific compiler configurations"""
        if benchmark_name not in self.trainer.environments:
            logger.error(f"No environment found for benchmark {benchmark_name}")
            return {}
        
        env = self.trainer.environments[benchmark_name]
        data = self.benchmark_data[benchmark_name]
        
        logger.info(f"Testing {len(configs)} specific configurations for {benchmark_name}")
        
        results = []
        
        for i, config in enumerate(configs):
            config_array = np.array(config)
            
            # Get execution time
            exec_time = env._get_execution_time(config_array)
            
            # Calculate improvement
            baseline_time = data['baseline_time']
            improvement = (baseline_time - exec_time) / baseline_time
            
            result = {
                'config_id': i,
                'config': config,
                'config_str': self._config_to_string(config_array),
                'execution_time': exec_time,
                'improvement': improvement,
                'better_than_baseline': exec_time < baseline_time
            }
            
            results.append(result)
        
        # Sort by performance
        results.sort(key=lambda x: x['execution_time'])
        
        summary = {
            'total_configs': len(results),
            'configs_better_than_baseline': sum(1 for r in results if r['better_than_baseline']),
            'best_config': results[0] if results else None,
            'worst_config': results[-1] if results else None,
            'baseline_time': baseline_time,
            'best_known_time': data['best_time']
        }
        
        return {
            'results': results,
            'summary': summary
        }
    
    def compare_with_optimal_configs(self, benchmark_name: str) -> Dict:
        """Compare agent's recommendations with known optimal configurations"""
        if benchmark_name not in self.trainer.agents:
            logger.error(f"No agent found for benchmark {benchmark_name}")
            return {}
        
        # Get best and worst samples from dataset
        best_worst = self.get_best_worst_samples(benchmark_name, n_each=5)
        
        if not best_worst:
            return {}
        
        agent = self.trainer.agents[benchmark_name]
        env = self.trainer.environments[benchmark_name]
        
        # Test agent against best configurations
        comparisons = []
        
        for best_sample in best_worst['best_samples']:
            # Get agent recommendation
            state = env.reset()
            agent_config = agent.act(state, training=False)
            agent_time = env._get_execution_time(agent_config)
            
            comparison = {
                'optimal_config': best_sample['config'],
                'optimal_config_str': best_sample['config_str'],
                'optimal_time': best_sample['execution_time'],
                'agent_config': agent_config.tolist(),
                'agent_config_str': self._config_to_string(agent_config),
                'agent_time': agent_time,
                'agent_matches_optimal': np.array_equal(agent_config, best_sample['config']),
                'agent_better_than_optimal': agent_time < best_sample['execution_time']
            }
            
            comparisons.append(comparison)
        
        # Calculate statistics
        matches = sum(1 for c in comparisons if c['agent_matches_optimal'])
        agent_better = sum(1 for c in comparisons if c['agent_better_than_optimal'])
        
        return {
            'comparisons': comparisons,
            'summary': {
                'total_optimal_configs_tested': len(comparisons),
                'exact_matches': matches,
                'agent_better_than_optimal': agent_better,
                'match_rate': matches / len(comparisons) if comparisons else 0,
                'improvement_rate': agent_better / len(comparisons) if comparisons else 0
            }
        }
    
    def _config_to_string(self, config: np.ndarray) -> str:
        """Convert configuration array to readable string"""
        config_parts = []
        for i, flag_value in enumerate(config):
            if flag_value == 1:
                config_parts.append(f"-{self.flag_names[i]}")
        
        return " ".join(config_parts) if config_parts else "No flags"
    
    def visualize_test_results(self, benchmark_name: str, test_results: Dict, save_plots: bool = True):
        """Visualize test results"""
        if 'comparisons' not in test_results:
            logger.error("No comparison data found in test results")
            return
        
        comparisons = test_results['comparisons']
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Agent vs Dataset Comparison: {benchmark_name}', fontsize=16)
        
        # Plot 1: Execution Time Comparison
        dataset_times = [c['dataset_time'] for c in comparisons]
        agent_times = [c['agent_time'] for c in comparisons]
        
        axes[0, 0].scatter(dataset_times, agent_times, alpha=0.6)
        axes[0, 0].plot([min(dataset_times), max(dataset_times)], 
                       [min(dataset_times), max(dataset_times)], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('Dataset Execution Time')
        axes[0, 0].set_ylabel('Agent Predicted Time')
        axes[0, 0].set_title('Execution Time Comparison')
        axes[0, 0].grid(True)
        
        # Plot 2: Improvement Comparison
        dataset_improvements = [c['dataset_improvement'] for c in comparisons]
        agent_improvements = [c['agent_improvement'] for c in comparisons]
        
        axes[0, 1].scatter(dataset_improvements, agent_improvements, alpha=0.6)
        axes[0, 1].plot([min(dataset_improvements), max(dataset_improvements)], 
                       [min(dataset_improvements), max(dataset_improvements)], 'r--', alpha=0.5)
        axes[0, 1].set_xlabel('Dataset Improvement')
        axes[0, 1].set_ylabel('Agent Improvement')
        axes[0, 1].set_title('Improvement Comparison')
        axes[0, 1].grid(True)
        
        # Plot 3: Win/Loss Distribution
        agent_wins = sum(1 for c in comparisons if c['agent_better'])
        dataset_wins = len(comparisons) - agent_wins
        
        axes[1, 0].bar(['Agent Better', 'Dataset Better'], [agent_wins, dataset_wins])
        axes[1, 0].set_title('Performance Comparison')
        axes[1, 0].set_ylabel('Count')
        
        # Plot 4: Improvement Difference Distribution
        improvement_diffs = [c['improvement_diff'] for c in comparisons]
        
        axes[1, 1].hist(improvement_diffs, bins=20, alpha=0.6, edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Improvement Difference (Agent - Dataset)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Improvement Difference Distribution')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            os.makedirs("test_plots", exist_ok=True)
            plt.savefig(f"test_plots/{benchmark_name}_test_results.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def run_comprehensive_test(self, benchmark_name: str, n_random_tests: int = 100) -> Dict:
        """Run comprehensive test suite for a benchmark"""
        logger.info(f"Running comprehensive test for {benchmark_name}")
        
        results = {}
        
        # Test 1: Agent vs Random Dataset Samples
        logger.info("Testing agent vs random dataset samples...")
        results['agent_vs_dataset'] = self.test_agent_vs_dataset(benchmark_name, n_random_tests)
        
        # Test 2: Compare with Optimal Configurations
        logger.info("Comparing with optimal configurations...")
        results['agent_vs_optimal'] = self.compare_with_optimal_configs(benchmark_name)
        
        # Test 3: Test Specific Interesting Configurations
        logger.info("Testing specific configurations...")
        
        # Test some predefined configurations
        test_configs = [
            [0, 0, 0, 0, 0, 0, 0],  # No optimizations
            [1, 1, 1, 1, 1, 1, 1],  # All optimizations
            [0, 0, 0, 0, 0, 0, 1],  # Only O2
            [1, 0, 0, 0, 0, 0, 1],  # O2 + unsafe-math
            [0, 1, 1, 1, 1, 0, 0],  # Disable optimizations only
        ]
        
        results['specific_configs'] = self.test_specific_configurations(benchmark_name, test_configs)
        
        # Generate visualization
        if 'agent_vs_dataset' in results:
            self.visualize_test_results(benchmark_name, results['agent_vs_dataset'])
        
        return results
    
    def save_test_results(self, results: Dict, filepath: str):
        """Save test results to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {filepath}")
    
    def print_test_summary(self, benchmark_name: str, results: Dict):
        """Print a summary of test results"""
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY: {benchmark_name}")
        print(f"{'='*60}")
        
        if 'agent_vs_dataset' in results:
            summary = results['agent_vs_dataset']['summary']
            print(f"Agent vs Dataset ({summary['total_tests']} tests):")
            print(f"  Win Rate: {summary['win_rate']:.2%}")
            print(f"  Avg Dataset Improvement: {summary['avg_dataset_improvement']:.4f}")
            print(f"  Avg Agent Improvement: {summary['avg_agent_improvement']:.4f}")
            print(f"  Improvement Difference: {summary['avg_improvement_difference']:.4f}")
        
        if 'agent_vs_optimal' in results:
            summary = results['agent_vs_optimal']['summary']
            print(f"\nAgent vs Optimal Configurations:")
            print(f"  Exact Matches: {summary['exact_matches']}/{summary['total_optimal_configs_tested']}")
            print(f"  Match Rate: {summary['match_rate']:.2%}")
            print(f"  Agent Better Than Optimal: {summary['agent_better_than_optimal']}")
        
        if 'specific_configs' in results:
            summary = results['specific_configs']['summary']
            print(f"\nSpecific Configuration Tests:")
            print(f"  Total Configs: {summary['total_configs']}")
            print(f"  Better Than Baseline: {summary['configs_better_than_baseline']}")
            if summary['best_config']:
                print(f"  Best Config: {summary['best_config']['config_str']}")
                print(f"  Best Time: {summary['best_config']['execution_time']:.6f}")