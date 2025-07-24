import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from alt_data_preprocessor import AltDataPreprocessor
import os

def analyze_original_data():
    """Analyze the original data.csv file"""
    print("="*80)
    print("ORIGINAL DATASET ANALYSIS")
    print("="*80)
    
    # Load original data
    pre = AltDataPreprocessor()
    raw_data = pre.load_data()
    
    print(f"Dataset shape: {raw_data.shape}")
    print(f"Total samples: {len(raw_data)}")
    print(f"Total columns: {len(raw_data.columns)}")
    
    # Column analysis
    print(f"\nColumn types:")
    print(raw_data.dtypes.value_counts())
    
    # Check for missing values
    missing_data = raw_data.isnull().sum()
    if missing_data.sum() > 0:
        print(f"\nMissing values:")
        print(missing_data[missing_data > 0])
    else:
        print(f"\nNo missing values found!")
    
    # Analyze APP_NAME (benchmarks)
    if 'APP_NAME' in raw_data.columns:
        benchmarks = raw_data['APP_NAME'].value_counts()
        print(f"\nBenchmark distribution:")
        print(f"Total unique benchmarks: {len(benchmarks)}")
        print(f"Average samples per benchmark: {benchmarks.mean():.1f}")
        print(f"Min samples per benchmark: {benchmarks.min()}")
        print(f"Max samples per benchmark: {benchmarks.max()}")
        
        print(f"\nTop 10 benchmarks by sample count:")
        print(benchmarks.head(10))
    
    # Analyze compiler flags
    exec_time_cols = getattr(pre, 'exec_time_columns', [])
    if exec_time_cols is None:
        exec_time_cols = []
    flag_cols = [col for col in raw_data.columns if col not in ['APP_NAME', 'code_size'] + exec_time_cols]
    print(f"\nCompiler flag analysis:")
    print(f"Total flag columns: {len(flag_cols)}")
    
    # Show flag value distribution
    print(f"\nFlag value distribution (first 10 flags):")
    for i, flag in enumerate(flag_cols[:10]):
        values = raw_data[flag].value_counts()
        print(f"  {flag}: {dict(values)}")
    
    # Analyze execution times
    print(f"\nExecution time analysis:")
    if exec_time_cols:
        for i, col in enumerate(exec_time_cols):
            times = pd.to_numeric(raw_data[col], errors='coerce')
            print(f"  {col}:")
            print(f"    Mean: {times.mean():.2f}")
            print(f"    Std: {times.std():.2f}")
            print(f"    Min: {times.min():.2f}")
            print(f"    Max: {times.max():.2f}")
    else:
        print("  No execution time columns found")
    
    # Analyze code_size if present
    if 'code_size' in raw_data.columns:
        code_sizes = pd.to_numeric(raw_data['code_size'], errors='coerce')
        print(f"\nCode size analysis:")
        print(f"  Mean: {code_sizes.mean():.0f}")
        print(f"  Std: {code_sizes.std():.0f}")
        print(f"  Min: {code_sizes.min():.0f}")
        print(f"  Max: {code_sizes.max():.0f}")
    
    return raw_data

def analyze_preprocessed_data():
    """Analyze the preprocessed benchmark data"""
    print("\n" + "="*80)
    print("PREPROCESSED DATASET ANALYSIS")
    print("="*80)
    
    # Load preprocessed data
    processed_dir = os.path.join(os.path.dirname(__file__), 'processed')
    benchmark_data = {}
    
    for filename in os.listdir(processed_dir):
        if filename.endswith('.csv'):
            benchmark_name = filename.replace('.csv', '')
            filepath = os.path.join(processed_dir, filename)
            df = pd.read_csv(filepath)
            benchmark_data[benchmark_name] = df
    
    print(f"Total preprocessed benchmarks: {len(benchmark_data)}")
    
    # Analyze each benchmark
    all_stats = []
    for name, df in benchmark_data.items():
        flag_cols = [col for col in df.columns if col != 'mean_exec_time']
        exec_times = df['mean_exec_time']
        
        stats = {
            'benchmark': name,
            'samples': len(df),
            'flags': len(flag_cols),
            'mean_time': exec_times.mean(),
            'std_time': exec_times.std(),
            'min_time': exec_times.min(),
            'max_time': exec_times.max(),
            'time_range': exec_times.max() - exec_times.min(),
            'improvement_potential': (exec_times.max() - exec_times.min()) / exec_times.max() * 100
        }
        all_stats.append(stats)
    
    stats_df = pd.DataFrame(all_stats)
    
    print(f"\nBenchmark statistics summary:")
    print(stats_df.describe())
    
    print(f"\nTop 5 benchmarks by improvement potential:")
    top_improvements = stats_df.nlargest(5, 'improvement_potential')
    for _, row in top_improvements.iterrows():
        print(f"  {row['benchmark']}: {row['improvement_potential']:.1f}% improvement potential")
    
    print(f"\nFlag configuration analysis:")
    print(f"  Total unique flag combinations across all benchmarks: {sum(len(df) for df in benchmark_data.values())}")
    print(f"  Average configurations per benchmark: {np.mean([len(df) for df in benchmark_data.values()]):.1f}")
    
    # Analyze flag patterns
    print(f"\nFlag pattern analysis (first benchmark as example):")
    first_bench = list(benchmark_data.values())[0]
    flag_cols = [col for col in first_bench.columns if col != 'mean_exec_time']
    
    print(f"  Flag columns: {len(flag_cols)}")
    print(f"  Sample flag configurations:")
    for i in range(min(3, len(first_bench))):
        config = first_bench.iloc[i][flag_cols].values
        time = first_bench.iloc[i]['mean_exec_time']
        print(f"    Config {i+1}: {config[:10]}... (time: {time:.2f})")
    
    return benchmark_data, stats_df

def create_visualizations(benchmark_data, stats_df):
    """Create visualizations of the dataset"""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Create output directory
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Execution time distribution across benchmarks
    plt.figure(figsize=(15, 8))
    all_times = []
    all_names = []
    for name, df in benchmark_data.items():
        all_times.extend(df['mean_exec_time'].values)
        all_names.extend([name] * len(df))
    
    time_df = pd.DataFrame({'benchmark': all_names, 'execution_time': all_times})
    
    plt.subplot(2, 2, 1)
    sns.boxplot(data=time_df, x='benchmark', y='execution_time')
    plt.xticks(rotation=45, ha='right')
    plt.title('Execution Time Distribution by Benchmark')
    plt.ylabel('Execution Time')
    
    # 2. Improvement potential by benchmark
    plt.subplot(2, 2, 2)
    sns.barplot(data=stats_df, x='benchmark', y='improvement_potential')
    plt.xticks(rotation=45, ha='right')
    plt.title('Improvement Potential by Benchmark')
    plt.ylabel('Improvement Potential (%)')
    
    # 3. Sample count by benchmark
    plt.subplot(2, 2, 3)
    sns.barplot(data=stats_df, x='benchmark', y='samples')
    plt.xticks(rotation=45, ha='right')
    plt.title('Sample Count by Benchmark')
    plt.ylabel('Number of Samples')
    
    # 4. Flag count by benchmark
    plt.subplot(2, 2, 4)
    sns.barplot(data=stats_df, x='benchmark', y='flags')
    plt.xticks(rotation=45, ha='right')
    plt.title('Flag Count by Benchmark')
    plt.ylabel('Number of Flags')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'dataset_overview.png'), dpi=300, bbox_inches='tight')
    print(f"Saved dataset overview plot to: {plots_dir}/dataset_overview.png")
    
    # 5. Execution time histogram
    plt.figure(figsize=(12, 6))
    plt.hist(all_times, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Execution Time')
    plt.ylabel('Frequency')
    plt.title('Distribution of Execution Times Across All Benchmarks')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'execution_time_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"Saved execution time distribution plot to: {plots_dir}/execution_time_distribution.png")
    
    plt.close('all')

def main():
    """Main analysis function"""
    print("ALTERNATE DATASET COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # Analyze original data
    raw_data = analyze_original_data()
    
    # Analyze preprocessed data
    benchmark_data, stats_df = analyze_preprocessed_data()
    
    # Create visualizations
    create_visualizations(benchmark_data, stats_df)
    
    # Summary
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    print(f"✅ Original dataset: {raw_data.shape[0]} samples, {raw_data.shape[1]} columns")
    print(f"✅ Preprocessed: {len(benchmark_data)} benchmarks")
    print(f"✅ Total configurations: {sum(len(df) for df in benchmark_data.values())}")
    print(f"✅ Average improvement potential: {stats_df['improvement_potential'].mean():.1f}%")
    print(f"✅ Best improvement potential: {stats_df['improvement_potential'].max():.1f}%")
    print(f"✅ Dataset ready for RL training!")

if __name__ == "__main__":
    main() 