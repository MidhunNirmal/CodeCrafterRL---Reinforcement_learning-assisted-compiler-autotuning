import os
import numpy as np
import torch
import pandas as pd
from alt_data_preprocessor_correct import AltDataPreprocessorCorrect
from alt_comp_env_unified import AltUnifiedCompilerEnvironment
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import seaborn as sns

class AltRLModelTester:
    def __init__(self, model_path="unified_alt_dqn_agent.pth"):
        self.model_path = model_path
        self.results_dir = "results"
        self.plots_dir = "plots"
        self.models_dir = "models"
        
        # Load data and environment
        self.pre = AltDataPreprocessorCorrect()
        self.pre.load_data()
        self.benchmark_data = self.pre.preprocess()
        self.unified_env = AltUnifiedCompilerEnvironment(self.benchmark_data)
        
        # Load trained agent
        self.agent = self.load_trained_agent()
        
    def load_trained_agent(self):
        """Load the trained DQN agent"""
        # Check multiple possible locations for the model
        possible_paths = [
            self.model_path,
            os.path.join(os.path.dirname(__file__), self.model_path),
            "unified_alt_dqn_agent.pth",
            os.path.join(os.path.dirname(__file__), "unified_alt_dqn_agent.pth")
        ]
        
        model_found = False
        for path in possible_paths:
            if os.path.exists(path):
                self.model_path = path
                model_found = True
                break
        
        if not model_found:
            print(f"Model not found. Checked paths: {possible_paths}")
            return None
            
        state_size = self.unified_env.observation_space.shape[0]
        action_size = 7
        
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            lr=0.0005,
            epsilon_decay=0.9995,
            memory_size=50000,
            batch_size=64
        )
        
        agent.load(self.model_path)
        print(f"Loaded trained model from {self.model_path}")
        return agent
    
    def test_single_benchmark(self, benchmark_name, num_episodes=10):
        """Test the model on a single benchmark"""
        if self.agent is None:
            print("No trained agent available")
            return None
            
        results = {
            'episodes': [],
            'rewards': [],
            'execution_times': [],
            'improvements': [],
            'configurations': [],
            'baseline_time': None,
            'best_time': None
        }
        
        # Get baseline metrics
        perf_model = self.unified_env.performance_models[benchmark_name]
        results['baseline_time'] = perf_model['baseline_time']
        results['best_time'] = perf_model['best_time']
        
        print(f"\nTesting on benchmark: {benchmark_name}")
        print(f"Baseline time: {results['baseline_time']:.4f} ms")
        print(f"Best possible time: {results['best_time']:.4f} ms")
        
        for episode in range(num_episodes):
            state = self.unified_env.reset(benchmark_name)
            total_reward = 0
            episode_times = []
            episode_improvements = []
            
            while True:
                # Use trained agent (no exploration)
                action = self.agent.act(state, training=False)
                next_state, reward, done, info = self.unified_env.step(action)
                
                total_reward += reward
                episode_times.append(info['execution_time'])
                episode_improvements.append(info['improvement'])
                
                state = next_state
                
                if done:
                    break
            
            # Record best performance from this episode
            best_time_in_episode = min(episode_times)
            best_improvement_in_episode = max(episode_improvements)
            
            results['episodes'].append(episode)
            results['rewards'].append(total_reward)
            results['execution_times'].append(best_time_in_episode)
            results['improvements'].append(best_improvement_in_episode)
            results['configurations'].append(info['config'])
            
            print(f"Episode {episode+1}: Reward={total_reward:.4f}, "
                  f"Best Time={best_time_in_episode:.4f} ms, "
                  f"Improvement={best_improvement_in_episode:.4f}")
        
        return results
    
    def test_all_benchmarks(self, num_episodes=5):
        """Test the model on all benchmarks"""
        all_results = {}
        
        for benchmark_name in self.benchmark_data.keys():
            results = self.test_single_benchmark(benchmark_name, num_episodes)
            if results:
                all_results[benchmark_name] = results
        
        return all_results
    
    def analyze_results(self, all_results):
        """Analyze and visualize test results"""
        if not all_results:
            print("No results to analyze")
            return
        
        # Create summary dataframe
        summary_data = []
        for bench, results in all_results.items():
            avg_reward = np.mean(results['rewards'])
            avg_time = np.mean(results['execution_times'])
            avg_improvement = np.mean(results['improvements'])
            best_improvement = max(results['improvements'])
            
            summary_data.append({
                'benchmark': bench,
                'category': bench.split('_')[0],
                'avg_reward': avg_reward,
                'avg_time': avg_time,
                'avg_improvement': avg_improvement,
                'best_improvement': best_improvement,
                'baseline_time': results['baseline_time'],
                'best_possible_time': results['best_time']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save results
        summary_df.to_csv(os.path.join(self.results_dir, 'test_results_summary.csv'), index=False)
        
        # Create visualizations
        self.create_test_visualizations(summary_df, all_results)
        
        return summary_df
    
    def create_test_visualizations(self, summary_df, all_results):
        """Create visualizations for test results"""
        plt.style.use('seaborn-v0_8')
        
        # 1. Performance comparison
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        sorted_df = summary_df.sort_values('best_improvement', ascending=False)
        colors = ['green' if imp > 0 else 'red' for imp in sorted_df['best_improvement']]
        bars = plt.bar(range(len(sorted_df)), sorted_df['best_improvement'], color=colors, alpha=0.7)
        plt.xlabel('Benchmarks')
        plt.ylabel('Best Improvement (%)')
        plt.title('Model Performance by Benchmark')
        plt.xticks(range(len(sorted_df)), [b.split('_')[-1] for b in sorted_df['benchmark']], 
                   rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars, sorted_df['best_improvement'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{imp:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Category performance
        plt.subplot(2, 3, 2)
        category_performance = summary_df.groupby('category')['best_improvement'].agg(['mean', 'std', 'count'])
        bars = plt.bar(category_performance.index, category_performance['mean'], 
                      yerr=category_performance['std'], capsize=5, alpha=0.7)
        plt.xlabel('Benchmark Categories')
        plt.ylabel('Average Best Improvement (%)')
        plt.title('Performance by Category')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, category_performance['count'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # 3. Reward vs Improvement correlation
        plt.subplot(2, 3, 3)
        plt.scatter(summary_df['avg_reward'], summary_df['best_improvement'], alpha=0.7)
        plt.xlabel('Average Reward')
        plt.ylabel('Best Improvement (%)')
        plt.title('Reward vs Improvement Correlation')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(summary_df['avg_reward'], summary_df['best_improvement'], 1)
        p = np.poly1d(z)
        plt.plot(summary_df['avg_reward'], p(summary_df['avg_reward']), "r--", alpha=0.8)
        
        # 4. Configuration analysis
        plt.subplot(2, 3, 4)
        flag_names = ['funsafe_math', 'fno_guess_branch', 'fno_ivopts', 'fno_tree_loop', 
                     'fno_inline', 'funroll_all', 'o2']
        
        # Analyze most common configurations
        all_configs = []
        for bench, results in all_results.items():
            all_configs.extend(results['configurations'])
        
        if all_configs:
            config_array = np.array(all_configs)
            flag_usage = np.mean(config_array, axis=0)
            
            bars = plt.bar(flag_names, flag_usage, alpha=0.7, color='skyblue')
            plt.xlabel('Compiler Flags')
            plt.ylabel('Usage Rate')
            plt.title('Flag Usage in Best Configurations')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (bar, usage) in enumerate(zip(bars, flag_usage)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{usage:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 5. Improvement distribution
        plt.subplot(2, 3, 5)
        positive_improvements = summary_df[summary_df['best_improvement'] > 0]['best_improvement']
        zero_improvements = summary_df[summary_df['best_improvement'] <= 0]['best_improvement']
        
        if len(positive_improvements) > 0:
            plt.hist(positive_improvements, bins=10, alpha=0.7, label='Positive Improvements', color='green')
        if len(zero_improvements) > 0:
            plt.hist(zero_improvements, bins=10, alpha=0.7, label='No Improvement', color='red')
        
        plt.xlabel('Best Improvement (%)')
        plt.ylabel('Number of Benchmarks')
        plt.title('Distribution of Best Improvements')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Performance summary
        plt.subplot(2, 3, 6)
        total_benchmarks = len(summary_df)
        successful_benchmarks = len(summary_df[summary_df['best_improvement'] > 0])
        avg_improvement = summary_df['best_improvement'].mean()
        
        plt.text(0.1, 0.8, f'Total Benchmarks: {total_benchmarks}', fontsize=12)
        plt.text(0.1, 0.6, f'Successful: {successful_benchmarks}', fontsize=12)
        plt.text(0.1, 0.4, f'Success Rate: {successful_benchmarks/total_benchmarks:.1%}', fontsize=12)
        plt.text(0.1, 0.2, f'Avg Improvement: {avg_improvement:.3f}%', fontsize=12)
        plt.title('Test Summary')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'test_results.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed results
        detailed_results = []
        for bench, results in all_results.items():
            for i, (reward, time, improvement, config) in enumerate(zip(
                results['rewards'], results['execution_times'], 
                results['improvements'], results['configurations'])):
                detailed_results.append({
                    'benchmark': bench,
                    'episode': i,
                    'reward': reward,
                    'execution_time': time,
                    'improvement': improvement,
                    'config': str(config)
                })
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(os.path.join(self.results_dir, 'test_results_detailed.csv'), index=False)
        
        print(f"Test visualizations saved to {self.plots_dir}/")
        print(f"Test results saved to {self.results_dir}/")
    
    def compare_with_baseline(self, all_results):
        """Compare model performance with baseline configurations"""
        comparison_data = []
        
        for bench, results in all_results.items():
            baseline_time = results['baseline_time']
            best_time = results['best_time']
            model_best_time = min(results['execution_times'])
            model_best_improvement = max(results['improvements'])
            
            # Calculate theoretical best improvement
            theoretical_improvement = (baseline_time - best_time) / baseline_time * 100
            
            comparison_data.append({
                'benchmark': bench,
                'category': bench.split('_')[0],
                'baseline_time': baseline_time,
                'theoretical_best_time': best_time,
                'model_best_time': model_best_time,
                'theoretical_improvement': theoretical_improvement,
                'model_improvement': model_best_improvement * 100,
                'efficiency': (model_best_improvement * 100) / theoretical_improvement if theoretical_improvement > 0 else 0
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(os.path.join(self.results_dir, 'baseline_comparison.csv'), index=False)
        
        # Create comparison visualization
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        x = comparison_df['theoretical_improvement']
        y = comparison_df['model_improvement']
        plt.scatter(x, y, alpha=0.7)
        plt.plot([0, max(x)], [0, max(x)], 'r--', alpha=0.8, label='Perfect Performance')
        plt.xlabel('Theoretical Best Improvement (%)')
        plt.ylabel('Model Achieved Improvement (%)')
        plt.title('Model vs Theoretical Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        efficiency = comparison_df['efficiency'].clip(0, 1)  # Cap at 100%
        plt.hist(efficiency, bins=10, alpha=0.7, color='green')
        plt.xlabel('Efficiency (Model/Theoretical)')
        plt.ylabel('Number of Benchmarks')
        plt.title('Model Efficiency Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        category_efficiency = comparison_df.groupby('category')['efficiency'].mean()
        bars = plt.bar(category_efficiency.index, category_efficiency, alpha=0.7)
        plt.xlabel('Benchmark Categories')
        plt.ylabel('Average Efficiency')
        plt.title('Efficiency by Category')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, f'Average Efficiency: {comparison_df["efficiency"].mean():.1%}', fontsize=12)
        plt.text(0.1, 0.6, f'Benchmarks with >50% efficiency: {len(comparison_df[comparison_df["efficiency"] > 0.5])}', fontsize=12)
        plt.text(0.1, 0.4, f'Benchmarks with >80% efficiency: {len(comparison_df[comparison_df["efficiency"] > 0.8])}', fontsize=12)
        plt.title('Efficiency Summary')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'baseline_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df

def main():
    tester = AltRLModelTester()
    
    if tester.agent is None:
        print("No trained model found. Please train a model first.")
        return
    
    print("Testing trained model on all benchmarks...")
    
    # Test all benchmarks
    all_results = tester.test_all_benchmarks(num_episodes=5)
    
    # Analyze results
    summary_df = tester.analyze_results(all_results)
    
    # Compare with baseline
    comparison_df = tester.compare_with_baseline(all_results)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    print(f"Total benchmarks tested: {len(summary_df)}")
    print(f"Successful optimizations: {len(summary_df[summary_df['best_improvement'] > 0])}")
    print(f"Success rate: {len(summary_df[summary_df['best_improvement'] > 0])/len(summary_df):.1%}")
    print(f"Average improvement: {summary_df['best_improvement'].mean():.3f}%")
    print(f"Best improvement: {summary_df['best_improvement'].max():.3f}%")
    print(f"Average efficiency: {comparison_df['efficiency'].mean():.1%}")
    
    print("\nTop 5 performing benchmarks:")
    top_5 = summary_df.nlargest(5, 'best_improvement')
    for _, row in top_5.iterrows():
        print(f"  {row['benchmark']}: {row['best_improvement']:.3f}% improvement")

if __name__ == "__main__":
    main() 