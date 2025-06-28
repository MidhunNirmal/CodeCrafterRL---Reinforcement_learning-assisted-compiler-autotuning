import logging
import os
from test_system import CompilerOptimizationTester

def setup_logging():
    logging.basicConfig(level=logging.INFO)

def main():
    """Run comprehensive tests on the compiler optimization system"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check if data file exists
    data_path = "data/exec_times.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    # Initialize tester
    logger.info("Initializing test system...")
    tester = CompilerOptimizationTester(data_path)
    
    # Setup (this will try to load trained models)
    tester.setup(load_trained_models=True)
    
    # Get available benchmarks
    available_benchmarks = list(tester.benchmark_data.keys())
    logger.info(f"Available benchmarks: {available_benchmarks}")
    
    # Test first few benchmarks (or all if you want)
    benchmarks_to_test = available_benchmarks[:3]  # Test first 3 benchmarks
    
    all_results = {}
    
    for benchmark in benchmarks_to_test:
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING BENCHMARK: {benchmark}")
        logger.info(f"{'='*80}")
        
        try:
            # Run comprehensive test
            results = tester.run_comprehensive_test(benchmark, n_random_tests=50)
            all_results[benchmark] = results
            
            # Print summary
            tester.print_test_summary(benchmark, results)
            
            # Save individual results
            os.makedirs("test_results", exist_ok=True)
            tester.save_test_results(results, f"test_results/{benchmark}_test_results.json")
            
        except Exception as e:
            logger.error(f"Error testing {benchmark}: {e}")
            continue
    
    # Save all results
    if all_results:
        tester.save_test_results(all_results, "test_results/all_test_results.json")
        logger.info("All test results saved to test_results/all_test_results.json")
    
    logger.info("Testing completed!")

if __name__ == "__main__":
    main()