import numpy as np
import logging
from test.test_system import CompilerOptimizationTester

def setup_logging():
    logging.basicConfig(level=logging.INFO)

def interactive_test():
    """Interactive testing interface"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print("COMPILER OPTIMIZATION INTERACTIVE TESTER")
    print("="*60)
    
    # Initialize tester
    tester = CompilerOptimizationTester("data/exec_times.csv")
    tester.setup(load_trained_models=True)
    
    available_benchmarks = list(tester.benchmark_data.keys())
    
    while True:
        print(f"\nAvailable benchmarks:")
        for i, benchmark in enumerate(available_benchmarks):
            print(f"  {i+1}. {benchmark}")
        
        print("\nOptions:")
        print("  0. Exit")
        print("  1-N. Select benchmark to test")
        print("  'all'. Test all benchmarks")
        
        choice = input("\nEnter your choice: ").strip()
        
        if choice == '0':
            break
        elif choice.lower() == 'all':
            for benchmark in available_benchmarks[:3]:  # Limit to first 3
                print(f"\n{'='*40}")
                print(f"Testing {benchmark}")
                print(f"{'='*40}")
                results = tester.run_comprehensive_test(benchmark, n_random_tests=20)
                tester.print_test_summary(benchmark, results)
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(available_benchmarks):
                    benchmark = available_benchmarks[idx]
                    
                    print(f"\nTesting {benchmark}...")
                    print("Select test type:")
                    print("  1. Quick test (20 random samples)")
                    print("  2. Comprehensive test (100 random samples)")
                    print("  3. Test specific configuration")
                    print("  4. Compare with best/worst from dataset")
                    
                    test_choice = input("Enter test type: ").strip()
                    
                    if test_choice == '1':
                        results = tester.test_agent_vs_dataset(benchmark, n_tests=20)
                        tester.print_test_summary(benchmark, {'agent_vs_dataset': results})
                        
                    elif test_choice == '2':
                        results = tester.run_comprehensive_test(benchmark, n_random_tests=100)
                        tester.print_test_summary(benchmark, results)
                        
                    elif test_choice == '3':
                        print("Enter compiler configuration (7 flags, 0 or 1):")
                        print("Flags: funsafe-math, fno-guess-branch, fno-ivopts, fno-tree-loop, fno-inline, funroll-all, O2")
                        config_str = input("Enter config (e.g., 1 0 0 0 0 0 1): ").strip()
                        
                        try:
                            config = [int(x) for x in config_str.split()]
                            if len(config) == 7:
                                results = tester.test_specific_configurations(benchmark, [config])
                                if results['results']:
                                    result = results['results'][0]
                                    print(f"\nTest Result:")
                                    print(f"  Configuration: {result['config_str']}")
                                    print(f"  Execution Time: {result['execution_time']:.6f}")
                                    print(f"  Improvement: {result['improvement']:.4f}")
                                    print(f"  Better than baseline: {result['better_than_baseline']}")
                            else:
                                print("Error: Please enter exactly 7 values (0 or 1)")
                        except ValueError:
                            print("Error: Please enter valid integers (0 or 1)")
                    
                    elif test_choice == '4':
                        results = tester.compare_with_optimal_configs(benchmark)
                        if results:
                            summary = results['summary']
                            print(f"\nOptimal Configuration Comparison:")
                            print(f"  Exact matches: {summary['exact_matches']}")
                            print(f"  Match rate: {summary['match_rate']:.2%}")
                            print(f"  Agent better than optimal: {summary['agent_better_than_optimal']}")
                    
                else:
                    print("Invalid choice!")
            except ValueError:
                print("Invalid input!")
    
    print("Goodbye!")

if __name__ == "__main__":
    interactive_test()