import logging
import argparse
import os
from trainer import CompilerOptimizationTrainer

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('compiler_optimization.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Compiler Flag Optimization with RL')
    parser.add_argument('--data_path', type=str, default='data/exec_times.csv',
                       help='Path to the execution times CSV file')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes per benchmark')
    parser.add_argument('--benchmark', type=str, default=None,
                       help='Specific benchmark to train (if None, trains all)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation after training')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate training visualizations')
    parser.add_argument('--load_results', type=str, default=None,
                       help='Load previous results from JSON file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Compiler Flag Optimization with Reinforcement Learning")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Episodes: {args.episodes}")
    
    try:
        # Initialize trainer
        trainer = CompilerOptimizationTrainer(args.data_path)
        
        # Load previous results if specified
        if args.load_results:
            trainer.load_results(args.load_results)
        
        # Setup trainer (load data and create agents)
        trainer.setup()
        
        # Training
        if args.benchmark:
            # Train specific benchmark
            logger.info(f"Training specific benchmark: {args.benchmark}")
            trainer.train_single_benchmark(args.benchmark, args.episodes)
        else:
            # Train all benchmarks
            logger.info("Training all benchmarks")
            trainer.train_all_benchmarks(args.episodes)
        
        # Save results
        trainer.save_results()
        
        # Evaluation
        if args.evaluate:
            logger.info("Running evaluation...")
            for benchmark in trainer.benchmark_data.keys():
                trainer.evaluate_benchmark(benchmark, episodes=100)
        
        # Visualization
        if args.visualize:
            logger.info("Generating visualizations...")
            trainer.visualize_training_progress()
        
        # Summary report
        report = trainer.get_summary_report()
        if report:
            logger.info("Training Summary Report:")
            logger.info(f"  Total benchmarks: {report['total_benchmarks']}")
            if 'overall_stats' in report:
                logger.info(f"  Benchmarks with improvements: {report['overall_stats']['benchmarks_with_improvements']}")
                logger.info(f"  Best overall improvement: {report['overall_stats']['best_overall_improvement']:.4f}")
        
        logger.info("Compiler optimization training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()