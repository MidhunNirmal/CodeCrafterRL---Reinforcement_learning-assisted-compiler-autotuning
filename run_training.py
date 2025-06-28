"""
Simple script to run the compiler optimization training
"""

import os
import sys
from trainer import CompilerOptimizationTrainer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training():
    """Run the complete training pipeline"""
    
    # Check if data file exists
    data_path = "data/exec_times.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.error("Please ensure the PolyBench execution data is available")
        return
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = CompilerOptimizationTrainer(data_path)
    
    # Setup (load data and create agents)
    logger.info("Setting up data and agents...")
    trainer.setup()
    
    # Train on all benchmarks
    logger.info("Starting training...")
    trainer.train_all_benchmarks(episodes=500)  # Reduced episodes for faster testing
    
    # Save results
    logger.info("Saving results...")
    trainer.save_results()
    
    # Run evaluation
    logger.info("Running evaluation...")
    for benchmark in list(trainer.benchmark_data.keys())[:3]:  # Evaluate first 3 benchmarks
        trainer.evaluate_benchmark(benchmark, episodes=50)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    trainer.visualize_training_progress()
    
    # Print summary
    report = trainer.get_summary_report()
    if report:
        print("\n" + "="*60)
        print("TRAINING SUMMARY REPORT")
        print("="*60)
        print(f"Total benchmarks trained: {report['total_benchmarks']}")
        
        if 'overall_stats' in report:
            print(f"Benchmarks with improvements: {report['overall_stats']['benchmarks_with_improvements']}")
            print(f"Total improvements found: {report['overall_stats']['total_improvements_found']}")
            print(f"Best overall improvement: {report['overall_stats']['best_overall_improvement']:.4f}")
            print(f"Mean improvement: {report['overall_stats']['mean_improvement']:.4f}")
        
        print("\nPer-benchmark results:")
        for benchmark, results in report['benchmark_results'].items():
            print(f"  {benchmark}:")
            print(f"    Best improvement: {results['best_improvement_found']:.4f}")
            print(f"    Improvements found: {results['num_improvements_found']}")
            print(f"    Training time: {results['training_time_minutes']:.2f} minutes")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    run_training()