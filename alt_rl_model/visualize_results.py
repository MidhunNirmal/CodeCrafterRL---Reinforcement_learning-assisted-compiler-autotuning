import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from alt_data_preprocessor_correct import AltDataPreprocessorCorrect
from alt_comp_env_unified import AltUnifiedCompilerEnvironment
from dqn_agent import DQNAgent
import torch

class AltRLVisualizer:
    def __init__(self):
        self.results_dir = "results"
        self.plots_dir = "plots"
        self.models_dir = "models"
        
    def create_training_visualizations(self, episode_rewards, best_improvements, benchmark_names):
        """Create training progress visualizations"""
        plt.style.use('seaborn-v0_8')
        
        # 1. Training Reward Curve
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        window_size = 100
        if len(episode_rewards) >= window_size:
            moving_avg = pd.Series(episode_rewards).rolling(window=window_size).mean()
            plt.plot(episode_rewards, alpha=0.3, label='Episode Rewards', color='lightblue')
            plt.plot(moving_avg, label=f'{window_size}-Episode Moving Average', linewidth=2, color='blue')
        else:
            plt.plot(episode_rewards, label='Episode Rewards', color='blue')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress - Reward Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Best Improvements by Benchmark
        plt.subplot(2, 2, 2)
        sorted_improvements = sorted(best_improvements.items(), key=lambda x: x[1], reverse=True)
        benchmarks, improvements = zip(*sorted_improvements)
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = plt.bar(range(len(benchmarks)), improvements, color=colors, alpha=0.7)
        plt.xlabel('Benchmarks')
        plt.ylabel('Best Improvement (%)')
        plt.title('Best Performance Improvements by Benchmark')
        plt.xticks(range(len(benchmarks)), [b.split('_')[-1] for b in benchmarks], rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{imp:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Improvement Distribution
        plt.subplot(2, 2, 3)
        positive_improvements = [imp for imp in improvements if imp > 0]
        negative_improvements = [imp for imp in improvements if imp <= 0]
        
        if positive_improvements:
            plt.hist(positive_improvements, bins=10, alpha=0.7, label='Positive Improvements', color='green')
        if negative_improvements:
            plt.hist(negative_improvements, bins=10, alpha=0.7, label='No Improvement', color='red')
        
        plt.xlabel('Improvement (%)')
        plt.ylabel('Number of Benchmarks')
        plt.title('Distribution of Performance Improvements')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Benchmark Categories Performance
        plt.subplot(2, 2, 4)
        categories = {}
        for bench, imp in best_improvements.items():
            category = bench.split('_')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(imp)
        
        category_means = {cat: np.mean(imps) for cat, imps in categories.items()}
        category_counts = {cat: len(imps) for cat, imps in categories.items()}
        
        cats = list(category_means.keys())
        means = list(category_means.values())
        counts = list(category_counts.values())
        
        bars = plt.bar(cats, means, alpha=0.7, color='skyblue')
        plt.xlabel('Benchmark Categories')
        plt.ylabel('Average Improvement (%)')
        plt.title('Performance by Benchmark Category')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'n={count}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_summary.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save results to CSV
        results_df = pd.DataFrame({
            'benchmark': list(best_improvements.keys()),
            'best_improvement': list(best_improvements.values()),
            'category': [b.split('_')[0] for b in best_improvements.keys()]
        })
        results_df.to_csv(os.path.join(self.results_dir, 'training_results.csv'), index=False)
        
        print(f"Visualizations saved to {self.plots_dir}/")
        print(f"Results saved to {self.results_dir}/training_results.csv")
        
    def create_benchmark_analysis(self, benchmark_data):
        """Create detailed benchmark analysis plots"""
        plt.figure(figsize=(15, 10))
        
        # 1. Execution Time Distribution by Benchmark
        plt.subplot(2, 3, 1)
        times_data = []
        bench_labels = []
        for bench, df in benchmark_data.items():
            times_data.extend(df['mean_exec_time'].values)
            bench_labels.extend([bench.split('_')[-1]] * len(df))
        
        time_df = pd.DataFrame({'time': times_data, 'benchmark': bench_labels})
        sns.boxplot(data=time_df, x='benchmark', y='time')
        plt.xticks(rotation=45, ha='right')
        plt.title('Execution Time Distribution by Benchmark')
        plt.ylabel('Execution Time (ms)')
        
        # 2. Flag Usage Analysis
        plt.subplot(2, 3, 2)
        flag_names = ['funsafe_math_optimizations', 'fno_guess_branch_probability', 'fno_ivopts', 'fno_tree_loop_optimize', 
                     'fno_inline_functions', 'funroll_all_loops', 'o2']
        flag_usage = []
        
        for bench, df in benchmark_data.items():
            for flag in flag_names:
                if flag in df.columns:
                    usage_rate = df[flag].mean()
                    flag_usage.append({'benchmark': bench, 
                                     'flag': flag, 'usage_rate': usage_rate})
        
        flag_df = pd.DataFrame(flag_usage)
        if not flag_df.empty:
            # Handle potential duplicate entries by using full benchmark names
            pivot_df = flag_df.pivot(index='benchmark', columns='flag', values='usage_rate')
            sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd')
            plt.title('Flag Usage Rate by Benchmark')
        
        # 3. Performance vs Flag Count
        plt.subplot(2, 3, 3)
        flag_counts = []
        best_times = []
        for bench, df in benchmark_data.items():
            flag_cols = [col for col in df.columns if col != 'mean_exec_time']
            for _, row in df.iterrows():
                flag_count = sum(row[flag_cols])
                flag_counts.append(flag_count)
                best_times.append(row['mean_exec_time'])
        
        plt.scatter(flag_counts, best_times, alpha=0.6)
        plt.xlabel('Number of Flags Enabled')
        plt.ylabel('Execution Time (ms)')
        plt.title('Performance vs Flag Count')
        
        # 4. Benchmark Categories Performance
        plt.subplot(2, 3, 4)
        categories = {}
        for bench, df in benchmark_data.items():
            category = bench.split('_')[0]
            if category not in categories:
                categories[category] = []
            categories[category].extend(df['mean_exec_time'].values)
        
        category_stats = {cat: {'mean': np.mean(times), 'std': np.std(times)} 
                         for cat, times in categories.items()}
        
        cats = list(category_stats.keys())
        means = [stats['mean'] for stats in category_stats.values()]
        stds = [stats['std'] for stats in category_stats.values()]
        
        plt.bar(cats, means, yerr=stds, capsize=5, alpha=0.7)
        plt.xlabel('Benchmark Categories')
        plt.ylabel('Average Execution Time (ms)')
        plt.title('Performance by Category')
        plt.xticks(rotation=45, ha='right')
        
        # 5. Improvement Potential Analysis
        plt.subplot(2, 3, 5)
        improvement_potentials = []
        for bench, df in benchmark_data.items():
            baseline = df['mean_exec_time'].mean()
            best_time = df['mean_exec_time'].min()
            worst_time = df['mean_exec_time'].max()
            potential = (worst_time - best_time) / baseline * 100
            improvement_potentials.append({
                'benchmark': bench.split('_')[-1],
                'potential': potential,
                'category': bench.split('_')[0]
            })
        
        imp_df = pd.DataFrame(improvement_potentials)
        if not imp_df.empty:
            sns.boxplot(data=imp_df, x='category', y='potential')
            plt.xlabel('Benchmark Categories')
            plt.ylabel('Improvement Potential (%)')
            plt.title('Improvement Potential by Category')
            plt.xticks(rotation=45, ha='right')
        
        # 6. Training Progress Summary
        plt.subplot(2, 3, 6)
        # This will be filled with training metrics if available
        plt.text(0.5, 0.5, 'Training Progress\nSummary', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=14)
        plt.title('Training Summary')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'benchmark_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_flag_importance_analysis(self, benchmark_data):
        """Analyze the importance of each compiler flag"""
        flag_names = ['funsafe_math_optimizations', 'fno_guess_branch_probability', 
                     'fno_ivopts', 'fno_tree_loop_optimize', 'fno_inline_functions', 
                     'funroll_all_loops', 'o2']
        
        flag_effects = {flag: [] for flag in flag_names}
        
        for bench, df in benchmark_data.items():
            for flag in flag_names:
                if flag in df.columns:
                    # Calculate effect when flag is on vs off
                    flag_on = df[df[flag] == 1]['mean_exec_time'].mean()
                    flag_off = df[df[flag] == 0]['mean_exec_time'].mean()
                    
                    if flag_off > 0:
                        effect = (flag_off - flag_on) / flag_off * 100
                        flag_effects[flag].append(effect)
        
        # Plot flag importance
        plt.figure(figsize=(12, 6))
        
        flag_short_names = ['funsafe_math', 'fno_guess_branch', 'fno_ivopts', 
                           'fno_tree_loop', 'fno_inline', 'funroll_all', 'o2']
        
        effects_data = []
        for flag, effects in flag_effects.items():
            if effects:
                effects_data.append({
                    'flag': flag_short_names[flag_names.index(flag)],
                    'mean_effect': np.mean(effects),
                    'std_effect': np.std(effects),
                    'count': len(effects)
                })
        
        if effects_data:
            effects_df = pd.DataFrame(effects_data)
            
            plt.subplot(1, 2, 1)
            # Handle potential NaN values in std_effect
            effects_df['std_effect'] = effects_df['std_effect'].fillna(0)
            
            bars = plt.bar(effects_df['flag'], effects_df['mean_effect'], 
                          yerr=effects_df['std_effect'], capsize=5, alpha=0.7)
            plt.xlabel('Compiler Flags')
            plt.ylabel('Average Performance Effect (%)')
            plt.title('Compiler Flag Importance')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Color bars based on effect
            for bar, effect in zip(bars, effects_df['mean_effect']):
                bar.set_color('green' if effect > 0 else 'red')
            
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, effects_df['count'])):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'n={count}', ha='center', va='bottom', fontsize=8)
            
            plt.subplot(1, 2, 2)
            # Flag usage heatmap
            usage_data = []
            for bench, df in benchmark_data.items():
                for flag in flag_names:
                    if flag in df.columns:
                        usage_rate = df[flag].mean()
                        usage_data.append({
                            'benchmark': bench.split('_')[-1],
                            'flag': flag_short_names[flag_names.index(flag)],
                            'usage': usage_rate
                        })
            
            usage_df = pd.DataFrame(usage_data)
            if not usage_df.empty:
                pivot_usage = usage_df.pivot(index='benchmark', columns='flag', values='usage')
                sns.heatmap(pivot_usage, annot=True, fmt='.2f', cmap='YlOrRd')
                plt.title('Flag Usage Rate by Benchmark')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'flag_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()

def main():
    visualizer = AltRLVisualizer()
    
    # Load benchmark data
    pre = AltDataPreprocessorCorrect()
    pre.load_data()
    benchmark_data = pre.preprocess()
    
    print("Creating visualizations...")
    
    # Create benchmark analysis
    visualizer.create_benchmark_analysis(benchmark_data)
    
    # Create flag importance analysis
    visualizer.create_flag_importance_analysis(benchmark_data)
    
    print("Visualization completed!")

if __name__ == "__main__":
    main() 