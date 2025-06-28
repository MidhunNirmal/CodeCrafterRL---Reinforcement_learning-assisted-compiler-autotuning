from test_system import CompilerOptimizationTester
import logging
import numpy as np

def quick_test_examples():
    """Run some quick test examples"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize tester
    tester = CompilerOptimizationTester("../data/exec_times.csv")
    tester.setup(load_trained_models=True)
    
    # Get first benchmark
    benchmark_name = list(tester.benchmark_data.keys())[0]
    print(f"Testing benchmark: {benchmark_name}")
    
    # Example 1: Test random samples from dataset
    print("\n1. Testing agent vs random dataset samples...")
    results = tester.test_agent_vs_dataset(benchmark_name, n_tests=10)
    if results:
        summary = results['summary']
        print(f"   Agent win rate: {summary['win_rate']:.2%}")
        print(f"   Average improvement difference: {summary['avg_improvement_difference']:.4f}")
    
    # Example 2: Test specific configurations
    print("\n2. Testing specific configurations...")
    test_configs = [
        [0, 0, 0, 0, 0, 0, 0],  # No optimizations
        [1, 1, 1, 1, 1, 1, 1],  # All optimizations
        [0, 0, 0, 0, 0, 0, 1],  # Only -O2
        [1, 0, 0, 0, 0, 0, 1],  # -O2 + unsafe math
    ]
    
    results = tester.test_specific_configurations(benchmark_name, test_configs)
    if results and results['results']:
        print("   Configuration test results:")
        for result in results['results']:
            print(f"     {result['config_str']}: {result['execution_time']:.6f} "
                  f"(improvement: {result['improvement']:.4f})")
    
    # Example 3: Get best/worst samples from dataset
    print("\n3. Best and worst configurations from dataset...")
    best_worst = tester.get_best_worst_samples(benchmark_name, n_each=3)
    if best_worst:
        print("   Best configurations:")
        for sample in best_worst['best_samples']:
            print(f"     {sample['config_str']}: {sample['execution_time']:.6f}")
        
        print("   Worst configurations:")
        for sample in best_worst['worst_samples']:
            print(f"     {sample['config_str']}: {sample['execution_time']:.6f}")
    
    # Example 4: Compare agent with optimal configurations
    print("\n4. Comparing agent with optimal configurations...")
    optimal_results = tester.compare_with_optimal_configs(benchmark_name)
    if optimal_results:
        summary = optimal_results['summary']
        print(f"   Exact matches with optimal: {summary['exact_matches']}")
        print(f"   Match rate: {summary['match_rate']:.2%}")
        print(f"   Times agent beat optimal: {summary['agent_better_than_optimal']}")
    
    # Example 5: Test agent's consistency
    print("\n5. Testing agent consistency...")
    agent = tester.trainer.agents[benchmark_name]
    env = tester.trainer.environments[benchmark_name]
    
    # Test multiple predictions from same state
    recommendations = []
    for i in range(10):
        state = env.reset()
        config = agent.act(state, training=False)
        recommendations.append(config.tolist())
    
    # Check if agent gives consistent recommendations
    unique_recommendations = [list(x) for x in set(tuple(x) for x in recommendations)]
    print(f"   Unique recommendations out of 10 tests: {len(unique_recommendations)}")
    if len(unique_recommendations) <= 3:
        print("   Agent shows good consistency")
        for i, rec in enumerate(unique_recommendations):
            config_str = tester._config_to_string(np.array(rec))
            print(f"     Recommendation {i+1}: {config_str}")
    else:
        print("   Agent shows high variability in recommendations")

def benchmark_comparison_example():
    """Compare performance across different benchmarks"""
    logging.basicConfig(level=logging.INFO)
    
    tester = CompilerOptimizationTester("data/exec_times.csv")
    tester.setup(load_trained_models=True)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPARISON EXAMPLE")
    print("="*60)
    
    benchmark_stats = {}
    
    # Test first 3 benchmarks
    benchmarks_to_test = list(tester.benchmark_data.keys())[:3]
    
    for benchmark in benchmarks_to_test:
        print(f"\nTesting {benchmark}...")
        
        # Quick test
        results = tester.test_agent_vs_dataset(benchmark, n_tests=20)
        
        if results:
            summary = results['summary']
            benchmark_stats[benchmark] = {
                'win_rate': summary['win_rate'],
                'avg_improvement': summary['avg_agent_improvement'],
                'baseline_time': summary['baseline_time'],
                'best_known_time': summary['best_known_time']
            }
            
            print(f"  Win rate: {summary['win_rate']:.2%}")
            print(f"  Avg improvement: {summary['avg_agent_improvement']:.4f}")
    
    # Summary comparison
    if benchmark_stats:
        print(f"\n{'='*60}")
        print("BENCHMARK COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        # Find best performing benchmark for the agent
        best_benchmark = max(benchmark_stats.keys(), 
                           key=lambda x: benchmark_stats[x]['win_rate'])
        worst_benchmark = min(benchmark_stats.keys(), 
                            key=lambda x: benchmark_stats[x]['win_rate'])
        
        print(f"Best agent performance: {best_benchmark} "
              f"(win rate: {benchmark_stats[best_benchmark]['win_rate']:.2%})")
        print(f"Worst agent performance: {worst_benchmark} "
              f"(win rate: {benchmark_stats[worst_benchmark]['win_rate']:.2%})")
        
        avg_win_rate = np.mean([stats['win_rate'] for stats in benchmark_stats.values()])
        print(f"Average win rate across benchmarks: {avg_win_rate:.2%}")

def configuration_analysis_example():
    """Analyze which compiler configurations work best"""
    logging.basicConfig(level=logging.INFO)
    
    tester = CompilerOptimizationTester("data/exec_times.csv")
    tester.setup(load_trained_models=True)
    
    print("\n" + "="*60)
    print("CONFIGURATION ANALYSIS EXAMPLE")
    print("="*60)
    
    benchmark_name = list(tester.benchmark_data.keys())[0]
    print(f"Analyzing configurations for: {benchmark_name}")
    
    # Test all possible single-flag configurations
    single_flag_configs = []
    for i in range(7):
        config = [0] * 7
        config[i] = 1
        single_flag_configs.append(config)
    
    # Add baseline (no flags) and all flags
    test_configs = [
        [0, 0, 0, 0, 0, 0, 0],  # Baseline
        [1, 1, 1, 1, 1, 1, 1],  # All flags
    ] + single_flag_configs
    
    results = tester.test_specific_configurations(benchmark_name, test_configs)
    
    if results and results['results']:
        print("\nConfiguration Performance Analysis:")
        print("-" * 80)
        print(f"{'Configuration':<40} {'Exec Time':<12} {'Improvement':<12} {'Better?'}")
        print("-" * 80)
        
        for result in results['results']:
            better_str = "✓" if result['better_than_baseline'] else "✗"
            print(f"{result['config_str']:<40} {result['execution_time']:<12.6f} "
                  f"{result['improvement']:<12.4f} {better_str}")
        
        # Find most effective single flags
        single_flag_results = results['results'][2:]  # Skip baseline and all-flags
        best_single_flag = min(single_flag_results, key=lambda x: x['execution_time'])
        
        print(f"\nMost effective single flag: {best_single_flag['config_str']}")
        print(f"Improvement: {best_single_flag['improvement']:.4f}")

def agent_decision_analysis():
    """Analyze agent decision patterns"""
    logging.basicConfig(level=logging.INFO)
    
    tester = CompilerOptimizationTester("data/exec_times.csv")
    tester.setup(load_trained_models=True)
    
    print("\n" + "="*60)
    print("AGENT DECISION ANALYSIS")
    print("="*60)
    
    benchmark_name = list(tester.benchmark_data.keys())[0]
    agent = tester.trainer.agents[benchmark_name]
    env = tester.trainer.environments[benchmark_name]
    
    print(f"Analyzing agent decisions for: {benchmark_name}")
    
    # Collect multiple agent recommendations
    recommendations = []
    for i in range(100):
        state = env.reset()
        config = agent.act(state, training=False)
        recommendations.append(config)
    
    # Analyze flag usage frequency
    flag_usage = np.mean(recommendations, axis=0)
    
    print("\nFlag Usage Frequency (out of 100 recommendations):")
    print("-" * 50)
    for i, (flag_name, usage) in enumerate(zip(tester.flag_names, flag_usage)):
        print(f"{flag_name:<30}: {usage:.1%} ({int(usage*100)}/100)")
    
    # Find most common configurations
    unique_configs, counts = np.unique(recommendations, axis=0, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    
    print(f"\nMost Common Configurations:")
    print("-" * 50)
    for i in range(min(5, len(unique_configs))):
        idx = sorted_indices[i]
        config = unique_configs[idx]
        count = counts[idx]
        config_str = tester._config_to_string(config)
        print(f"{config_str:<40}: {count} times ({count/100:.1%})")

def main():
    """Run all quick test examples"""
    print("COMPILER OPTIMIZATION TESTING EXAMPLES")
    print("=" * 60)
    
    try:
        # Run basic examples
        quick_test_examples()
        
        # Run benchmark comparison
        benchmark_comparison_example()
        
        # Run configuration analysis
        configuration_analysis_example()
        
        # Run agent decision analysis
        agent_decision_analysis()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()