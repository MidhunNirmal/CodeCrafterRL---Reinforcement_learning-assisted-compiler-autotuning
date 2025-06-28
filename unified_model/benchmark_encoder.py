import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class BenchmarkEncoder:
    """
    Utility class to encode/decode benchmark information for the unified model.
    This allows the single model to differentiate between different benchmarks.
    """
    
    def __init__(self, benchmarks: List[str]):
        """
        Initialize the encoder with a list of benchmark names.
        
        Args:
            benchmarks: List of benchmark names (e.g., ['gemm', 'gesummv', 'symm'])
        """
        self.benchmarks = sorted(benchmarks)  # Sort for consistency
        self.benchmark_to_id = {name: idx for idx, name in enumerate(self.benchmarks)}
        self.id_to_benchmark = {idx: name for idx, name in enumerate(self.benchmarks)}
        self.num_benchmarks = len(self.benchmarks)
        
        logger.info(f"BenchmarkEncoder initialized with {self.num_benchmarks} benchmarks")
        logger.info(f"Benchmarks: {self.benchmarks}")
    
    def encode_benchmark(self, benchmark_name: str) -> np.ndarray:
        """
        Encode benchmark name as one-hot vector.
        
        Args:
            benchmark_name: Name of the benchmark
            
        Returns:
            One-hot encoded vector representing the benchmark
        """
        if benchmark_name not in self.benchmark_to_id:
            logger.warning(f"Unknown benchmark: {benchmark_name}. Using first benchmark.")
            benchmark_name = self.benchmarks[0]
        
        one_hot = np.zeros(self.num_benchmarks, dtype=np.float32)
        one_hot[self.benchmark_to_id[benchmark_name]] = 1.0
        return one_hot
    
    def encode_benchmark_embedding(self, benchmark_name: str, embedding_dim: int = 8) -> np.ndarray:
        """
        Encode benchmark as a learned embedding (placeholder for future enhancement).
        For now, returns a simple hash-based encoding.
        
        Args:
            benchmark_name: Name of the benchmark
            embedding_dim: Dimension of the embedding vector
            
        Returns:
            Embedding vector representing the benchmark
        """
        if benchmark_name not in self.benchmark_to_id:
            benchmark_name = self.benchmarks[0]
        
        # Simple hash-based embedding (can be replaced with learned embeddings)
        benchmark_id = self.benchmark_to_id[benchmark_name]
        
        # Create a deterministic but diverse embedding based on benchmark ID
        np.random.seed(benchmark_id)
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        np.random.seed(None)  # Reset seed
        
        # Normalize to unit vector
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def decode_benchmark(self, one_hot_vector: np.ndarray) -> str:
        """
        Decode one-hot vector back to benchmark name.
        
        Args:
            one_hot_vector: One-hot encoded benchmark vector
            
        Returns:
            Benchmark name
        """
        benchmark_id = np.argmax(one_hot_vector)
        return self.id_to_benchmark.get(benchmark_id, self.benchmarks[0])
    
    def get_benchmark_features(self, benchmark_name: str, benchmark_data: Dict) -> np.ndarray:
        """
        Extract statistical features about a benchmark for the unified model.
        
        Args:
            benchmark_name: Name of the benchmark
            benchmark_data: Dictionary containing benchmark data
            
        Returns:
            Feature vector with benchmark characteristics
        """
        if benchmark_name not in benchmark_data:
            logger.warning(f"No data found for benchmark: {benchmark_name}")
            return np.zeros(6, dtype=np.float32)
        
        data = benchmark_data[benchmark_name]
        
        # Extract statistical features
        baseline_time = data.get('baseline_time', 1.0)
        best_time = data.get('best_time', 1.0)
        worst_time = data.get('worst_time', 1.0)
        data_size = data.get('data_size', 100)
        
        # Calculate normalized features
        improvement_potential = (baseline_time - best_time) / baseline_time if baseline_time > 0 else 0
        performance_variance = (worst_time - best_time) / baseline_time if baseline_time > 0 else 0
        log_data_size = np.log10(max(1, data_size))
        
        # Relative performance metrics
        baseline_percentile = 0.5  # Baseline is typically median
        best_percentile = 0.0  # Best is minimum
        worst_percentile = 1.0  # Worst is maximum
        
        features = np.array([
            improvement_potential,     # How much improvement is possible
            performance_variance,      # How much performance varies
            log_data_size / 4.0,      # Log of dataset size (normalized)
            baseline_percentile,       # Where baseline sits in distribution
            best_percentile,          # Where best performance sits
            worst_percentile          # Where worst performance sits
        ], dtype=np.float32)
        
        return features
    
    def create_unified_state(self, 
                           compiler_flags: np.ndarray,
                           benchmark_name: str,
                           benchmark_data: Dict,
                           performance_context: Optional[np.ndarray] = None,
                           use_one_hot: bool = True) -> np.ndarray:
        """
        Create a unified state representation that includes benchmark information.
        
        Args:
            compiler_flags: Current compiler flag configuration (7 elements)
            benchmark_name: Name of the current benchmark
            benchmark_data: Dictionary containing all benchmark data
            performance_context: Optional performance context (current time, steps, etc.)
            use_one_hot: Whether to use one-hot encoding (vs embedding)
            
        Returns:
            Unified state vector for the model
        """
        # Ensure compiler flags are the right shape
        flags = np.array(compiler_flags, dtype=np.float32).flatten()[:7]
        if len(flags) < 7:
            flags = np.pad(flags, (0, 7 - len(flags)), 'constant')
        
        # Encode benchmark
        if use_one_hot:
            benchmark_encoding = self.encode_benchmark(benchmark_name)
        else:
            benchmark_encoding = self.encode_benchmark_embedding(benchmark_name)
        
        # Get benchmark features
        benchmark_features = self.get_benchmark_features(benchmark_name, benchmark_data)
        
        # Default performance context if not provided
        if performance_context is None:
            performance_context = np.array([0.5, 1.0, 0.0], dtype=np.float32)  # [normalized_perf, steps_remaining, improvement]
        
        # Combine all components
        unified_state = np.concatenate([
            flags,                    # 7 compiler flags
            benchmark_encoding,       # N benchmark encoding (one-hot or embedding)
            benchmark_features,       # 6 benchmark statistical features
            performance_context       # 3 performance context features
        ])
        
        return unified_state.astype(np.float32)
    
    def get_state_size(self, use_one_hot: bool = True, embedding_dim: int = 8) -> int:
        """
        Get the size of the unified state vector.
        
        Args:
            use_one_hot: Whether using one-hot encoding
            embedding_dim: Size of embedding if not using one-hot
            
        Returns:
            Total state size
        """
        base_size = 7 + 6 + 3  # flags + benchmark_features + performance_context
        
        if use_one_hot:
            benchmark_size = self.num_benchmarks
        else:
            benchmark_size = embedding_dim
        
        return base_size + benchmark_size
    
    def get_benchmark_stats(self, benchmark_data: Dict) -> Dict:
        """
        Get statistics about all benchmarks for analysis.
        
        Args:
            benchmark_data: Dictionary containing all benchmark data
            
        Returns:
            Dictionary with benchmark statistics
        """
        stats = {
            'num_benchmarks': len(self.benchmarks),
            'benchmark_details': {}
        }
        
        for benchmark in self.benchmarks:
            if benchmark in benchmark_data:
                data = benchmark_data[benchmark]
                stats['benchmark_details'][benchmark] = {
                    'data_size': data.get('data_size', 0),
                    'baseline_time': data.get('baseline_time', 0),
                    'best_time': data.get('best_time', 0),
                    'improvement_potential': (data.get('baseline_time', 1) - data.get('best_time', 1)) / data.get('baseline_time', 1)
                }
            else:
                stats['benchmark_details'][benchmark] = {
                    'data_size': 0,
                    'baseline_time': 0,
                    'best_time': 0,
                    'improvement_potential': 0
                }
        
        return stats