import logging
import argparse
import os
import sys

# Add parent directory to path to import original modules
sys.path.append('..')

from unified_trainer import UnifiedCompilerOptimizationTrainer

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('unified_compiler_optimization.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Unified Compiler Flag Optimization with RL')
    parser.add_argument('--data_path', type=str, default='../exec_times.csv',
                       help='Path to the execution times CSV file')
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of training episodes')
    parser.add_argument('--benchmarks', type=str, nargs='*', default=None,
                       help='Specific benchmarks to train on (if None, uses all)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation after training')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate training visualizations')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Load pre-trained model from path')
    parser.add_argument('--save_frequency', type=int, default=1000,
                       help='How often to save the model during training')
    parser.add_argument('--eval_frequency', type=int, default=2000,
                       help='How often to run evaluation during training')
    parser.add_argument('--use_cycling', action='store_true', default=True,
                       help='Use benchmark cycling for balanced training')
    parser.add_argument('--use_embedding', action='store_true',
                       help='Use benchmark embeddings instead of one-hot encoding')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Unified Compiler Flag Optimization with Reinforcement Learning")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Selected benchmarks: {args.benchmarks}")
    
    try:
        # Initialize unified trainer
        trainer = UnifiedCompilerOptimizationTrainer(args.data_path)
        
        # Setup trainer (load data and create unified agent/environment)
        trainer.setup(
            benchmark_names=args.benchmarks,
            use_benchmark_cycling=args.use_cycling,
            use_one_hot_encoding=not args.use_embedding
        )
        
                 # Load pre-trained model if specified
         if args.load_model and os.path.exists(args.load_model) and trainer.agent:
             logger.info(f"Loading pre-trained model from {args.load_model}")
             trainer.agent.load(args.load_model)
        
        # Training
        logger.info("Starting unified training...")
        training_stats = trainer.train_unified_model(
            total_episodes=args.episodes,
            save_frequency=args.save_frequency,
            evaluation_frequency=args.eval_frequency,
            verbose=True
        )
        
        # Save results
        trainer.save_unified_results()
        
        # Evaluation
        if args.evaluate:
            logger.info("Running comprehensive evaluation...")
            evaluation_results = trainer.evaluate_unified_model(episodes_per_benchmark=100)
            
            # Log evaluation summary
            logger.info("Final Evaluation Results:")
            for benchmark, stats in evaluation_results.items():
                logger.info(f"  {benchmark}:")
                logger.info(f"    Mean improvement: {stats['mean_improvement']:.4f}")
                logger.info(f"    Best improvement: {stats['best_improvement']:.4f}")
                logger.info(f"    Mean execution time: {stats['mean_execution_time']:.6f}")
                logger.info(f"    Best execution time: {stats['best_execution_time']:.6f}")
            
            # Save evaluation results
            import json
            os.makedirs("unified_models", exist_ok=True)
            with open("unified_models/evaluation_results.json", 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = {}
                for benchmark, stats in evaluation_results.items():
                    serializable_results[benchmark] = {
                        k: v if not isinstance(v, list) or not v or not hasattr(v[0], 'tolist') 
                        else [item.tolist() if hasattr(item, 'tolist') else item for item in v]
                        for k, v in stats.items()
                    }
                json.dump(serializable_results, f, indent=2)
        
        # Visualization
        if args.visualize:
            logger.info("Generating visualizations...")
            trainer.visualize_unified_training()
        
        # Summary report
        report = trainer.get_unified_summary_report()
        if report:
            logger.info("Training Summary Report:")
            logger.info(f"  Total benchmarks: {report['total_benchmarks']}")
            logger.info(f"  Total episodes: {report['total_episodes']}")
            logger.info(f"  Training time: {report['training_time_hours']:.2f} hours")
            logger.info(f"  Model parameters: {report['model_parameters']:,}")
            
            if 'performance_summary' in report:
                perf = report['performance_summary']
                logger.info(f"  Mean improvement: {perf['mean_improvement']:.4f}")
                logger.info(f"  Best improvement: {perf['best_improvement']:.4f}")
                logger.info(f"  Positive improvement rate: {perf['positive_improvement_rate']:.2%}")
            
            # Save summary report
            with open("unified_models/summary_report.json", 'w') as f:
                json.dump(report, f, indent=2)
        
        logger.info("Unified compiler optimization completed successfully!")
        
        # Compare with original approach
        logger.info("\n" + "="*60)
        logger.info("UNIFIED MODEL ADVANTAGES:")
        logger.info("="*60)
        logger.info("1. Single model handles all benchmarks (vs separate models)")
        logger.info("2. Shared learning across benchmarks")
        logger.info("3. More efficient memory usage")
        logger.info("4. Transfer learning between similar benchmarks")
        logger.info("5. Easier deployment and maintenance")
        logger.info(f"6. Model size: {report.get('model_parameters', 0):,} parameters")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error during unified training: {e}")
        raise

def demo_unified_vs_separate():
    """Demonstrate the difference between unified and separate model approaches"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("UNIFIED MODEL ARCHITECTURE COMPARISON")
    logger.info("="*80)
    
    logger.info("\nORIGINAL APPROACH (Separate Models):")
    logger.info("- Each benchmark has its own DQN agent")
    logger.info("- Each agent has its own neural network")
    logger.info("- Each agent is trained independently")
    logger.info("- No knowledge sharing between benchmarks")
    logger.info("- Memory usage: N_benchmarks × model_size")
    logger.info("- Training time: N_benchmarks × individual_training_time")
    
    logger.info("\nUNIFIED APPROACH (Single Model):")
    logger.info("- One DQN agent handles all benchmarks")
    logger.info("- Single neural network with benchmark context")
    logger.info("- Training data from all benchmarks mixed together")
    logger.info("- Knowledge transfer between similar benchmarks")
    logger.info("- Memory usage: 1 × model_size")
    logger.info("- Training time: efficient batch learning")
    
    logger.info("\nKEY INNOVATIONS:")
    logger.info("- Benchmark encoder: embeds benchmark identity in state")
    logger.info("- Unified environment: cycles through benchmarks")
    logger.info("- Shared experience replay: learns from all benchmarks")
    logger.info("- Benchmark-aware agent: adapts behavior per benchmark")
    
    logger.info("="*80)

if __name__ == "__main__":
    # Show the comparison first
    demo_unified_vs_separate()
    
    # Then run the main training
    main()