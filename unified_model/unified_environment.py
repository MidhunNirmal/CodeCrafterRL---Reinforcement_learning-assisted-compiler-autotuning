import gym
from gym import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import random
from .benchmark_encoder import BenchmarkEncoder

logger = logging.getLogger(__name__)

class UnifiedCompilerEnvironment(gym.Env):
    """
    Unified compiler optimization environment that can handle multiple benchmarks.
    Instead of separate environments per benchmark, this single environment can 
    switch between benchmarks and provide unified state representations.
    """
    
    def __init__(self, 
                 benchmark_data: Dict, 
                 benchmark_names: Optional[List[str]] = None,
                 use_benchmark_cycling: bool = True,
                 use_one_hot_encoding: bool = True,
                 max_steps_per_episode: int = 50):
        """
        Initialize unified environment.
        
        Args:
            benchmark_data: Dictionary with data for all benchmarks
            benchmark_names: List of benchmark names to include (None = all)
            use_benchmark_cycling: Whether to cycle through benchmarks during training
            use_one_hot_encoding: Whether to use one-hot encoding for benchmarks
            max_steps_per_episode: Maximum steps per episode
        """
        super(UnifiedCompilerEnvironment, self).__init__()
        
        self.benchmark_data = benchmark_data
        self.use_benchmark_cycling = use_benchmark_cycling
        self.use_one_hot_encoding = use_one_hot_encoding
        self.max_steps_per_episode = max_steps_per_episode
        
        # Initialize benchmark names
        if benchmark_names is None:
            self.benchmark_names = list(benchmark_data.keys())
        else:
            self.benchmark_names = [name for name in benchmark_names if name in benchmark_data]
        
        if not self.benchmark_names:
            raise ValueError("No valid benchmarks found in the data!")
        
        logger.info(f"Unified environment initialized with {len(self.benchmark_names)} benchmarks")
        logger.info(f"Benchmarks: {self.benchmark_names}")
        
        # Initialize benchmark encoder
        self.benchmark_encoder = BenchmarkEncoder(self.benchmark_names)
        
        # Action space: 7 compiler flags (binary)
        self.action_space = spaces.MultiBinary(7)
        
        # Observation space: unified state representation
        state_size = self.benchmark_encoder.get_state_size(use_one_hot=use_one_hot_encoding)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(state_size,), dtype=np.float32
        )
        
        # Environment state
        self.current_benchmark = None
        self.current_config = None
        self.step_count = 0
        self.episode_count = 0
        
        # Performance tracking per benchmark
        self.benchmark_episode_counts = {name: 0 for name in self.benchmark_names}
        self.benchmark_performance_history = {name: [] for name in self.benchmark_names}
        
        # Current episode state
        self.current_baseline_time = 1.0
        self.current_best_time = 1.0
        self.current_worst_time = 1.0
        self.current_execution_time = 1.0
        
        # Create performance models for each benchmark (similar to original environment)
        self._create_performance_models()
        
        logger.info(f"State size: {state_size}")
        logger.info(f"Action space: {self.action_space}")
        
    def _create_performance_models(self):
        """Create performance lookup models for all benchmarks"""
        self.exact_lookups = {}
        self.nn_models = {}
        
        for benchmark_name in self.benchmark_names:
            data = self.benchmark_data[benchmark_name]
            X_raw = data['X_raw']
            y_raw = data['y_raw']
            
            # Use only compiler flags (first 7 features) for configuration matching
            X_flags = X_raw[:, :7]
            
            # Create exact lookup table
            exact_lookup = {}
            for config, time in zip(X_flags, y_raw):
                config_tuple = tuple(config.astype(int))
                if config_tuple not in exact_lookup:
                    exact_lookup[config_tuple] = []
                exact_lookup[config_tuple].append(time)
            
            # Average multiple measurements for same configuration
            for config in exact_lookup:
                exact_lookup[config] = np.mean(exact_lookup[config])
            
            self.exact_lookups[benchmark_name] = exact_lookup
            
            # For simplicity, store raw data for approximate matching
            self.nn_models[benchmark_name] = {
                'X_flags': X_flags,
                'y_raw': y_raw
            }
            
            logger.info(f"Performance model for {benchmark_name}: {len(exact_lookup)} configurations")
    
    def reset(self, benchmark_name: Optional[str] = None):
        """
        Reset environment to initial state.
        
        Args:
            benchmark_name: Specific benchmark to use (None = random or cycling)
            
        Returns:
            Initial observation
        """
        # Select benchmark
        if benchmark_name is not None and benchmark_name in self.benchmark_names:
            self.current_benchmark = benchmark_name
        elif self.use_benchmark_cycling:
            # Cycle through benchmarks to ensure balanced training
            min_episodes = min(self.benchmark_episode_counts.values())
            least_trained = [name for name, count in self.benchmark_episode_counts.items() 
                           if count == min_episodes]
            self.current_benchmark = random.choice(least_trained)
        else:
            # Random benchmark selection
            self.current_benchmark = random.choice(self.benchmark_names)
        
        # Initialize episode state
        self.step_count = 0
        self.episode_count += 1
        self.benchmark_episode_counts[self.current_benchmark] += 1
        
        # Set benchmark-specific parameters
        benchmark_data = self.benchmark_data[self.current_benchmark]
        self.current_baseline_time = benchmark_data['baseline_time']
        self.current_best_time = benchmark_data['best_time']
        self.current_worst_time = benchmark_data['worst_time']
        
        # Start with random configuration
        self.current_config = np.random.randint(0, 2, size=7)
        self.current_execution_time = self._get_execution_time(self.current_config)
        
        logger.debug(f"Reset to benchmark: {self.current_benchmark}, "
                    f"episode: {self.benchmark_episode_counts[self.current_benchmark]}")
        
        return self._get_observation()
    
    def step(self, action):
        """
        Execute action and return new state, reward, done, info.
        
        Args:
            action: Compiler flag configuration
            
        Returns:
            observation, reward, done, info
        """
        self.step_count += 1
        
        # Ensure action is the right shape and type
        if isinstance(action, (list, tuple)):
            action = np.array(action)
        action = action.astype(int).flatten()[:7]
        
        # Update configuration
        self.current_config = action.copy()
        
        # Get execution time for this configuration
        execution_time = self._get_execution_time(self.current_config)
        self.current_execution_time = execution_time
        
        # Calculate reward
        reward = self._calculate_reward(execution_time)
        
        # Check if done
        done = self.step_count >= self.max_steps_per_episode
        
        # Calculate improvement percentage
        improvement = (self.current_baseline_time - execution_time) / self.current_baseline_time
        
        # Store performance for this benchmark
        self.benchmark_performance_history[self.current_benchmark].append(improvement)
        
        info = {
            'execution_time': execution_time,
            'baseline_time': self.current_baseline_time,
            'best_time': self.current_best_time,
            'improvement': improvement,
            'config': self.current_config.copy(),
            'benchmark': self.current_benchmark,
            'step': self.step_count,
            'episode': self.episode_count,
            'benchmark_episode': self.benchmark_episode_counts[self.current_benchmark]
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Get current unified observation including benchmark context"""
        # Performance context
        normalized_current = (self.current_execution_time - self.current_best_time) / \
                           (self.current_worst_time - self.current_best_time)
        normalized_current = np.clip(normalized_current, 0, 1)
        
        steps_remaining = (self.max_steps_per_episode - self.step_count) / self.max_steps_per_episode
        
        improvement_from_baseline = (self.current_baseline_time - self.current_execution_time) / \
                                  self.current_baseline_time
        
        performance_context = np.array([
            normalized_current,
            steps_remaining,
            improvement_from_baseline
        ], dtype=np.float32)
        
                 # Create unified state using benchmark encoder
        benchmark_name = self.current_benchmark or self.benchmark_names[0]
        unified_state = self.benchmark_encoder.create_unified_state(
            compiler_flags=self.current_config,
            benchmark_name=benchmark_name,
            benchmark_data=self.benchmark_data,
            performance_context=performance_context,
            use_one_hot=self.use_one_hot_encoding
        )
        
        return unified_state
    
    def _get_execution_time(self, config):
        """Get execution time for given configuration and current benchmark"""
        config_tuple = tuple(config.astype(int))
        
        # Try exact lookup first
        exact_lookup = self.exact_lookups[self.current_benchmark]
        if config_tuple in exact_lookup:
            return exact_lookup[config_tuple]
        
        # Approximate using nearest neighbor
        nn_data = self.nn_models[self.current_benchmark]
        X_flags = nn_data['X_flags']
        y_raw = nn_data['y_raw']
        
        # Find nearest configurations using Hamming distance
        distances = []
        for x in X_flags:
            dist = np.sum(x != config)
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Use the 3 nearest neighbors
        nearest_indices = np.argsort(distances)[:3]
        nearest_distances = distances[nearest_indices]
        
        # Weight by inverse distance
        weights = 1.0 / (nearest_distances + 1e-8)
        weights = weights / weights.sum()
        
        predicted_time = np.average(y_raw[nearest_indices], weights=weights)
        
        # Add some noise for exploration
        noise_factor = 0.02
        noise = np.random.normal(0, noise_factor * predicted_time)
        
        return max(0.000001, predicted_time + noise)
    
    def _calculate_reward(self, execution_time):
        """Calculate reward based on execution time"""
        # Primary reward: improvement over baseline
        improvement_reward = (self.current_baseline_time - execution_time) / \
                           self.current_baseline_time * 100
        
        # Bonus for beating the best known time
        if execution_time < self.current_best_time:
            best_improvement = (self.current_best_time - execution_time) / \
                             self.current_best_time * 50
            improvement_reward += best_improvement
        
        # Small penalty for each step to encourage efficiency
        step_penalty = -0.1
        
        # Penalty for extremely bad configurations
        if execution_time > self.current_worst_time * 1.1:
            penalty = -10
        else:
            penalty = 0
        
        total_reward = improvement_reward + step_penalty + penalty
        
        return total_reward
    
    def get_benchmark_stats(self) -> Dict:
        """Get statistics about benchmark usage and performance"""
        stats = {
            'total_episodes': self.episode_count,
            'benchmark_episodes': self.benchmark_episode_counts.copy(),
            'benchmark_performance': {}
        }
        
        for benchmark in self.benchmark_names:
            performances = self.benchmark_performance_history[benchmark]
            if performances:
                stats['benchmark_performance'][benchmark] = {
                    'episodes': len(performances),
                    'mean_improvement': np.mean(performances),
                    'best_improvement': max(performances),
                    'recent_improvement': np.mean(performances[-10:]) if len(performances) >= 10 else 0
                }
            else:
                stats['benchmark_performance'][benchmark] = {
                    'episodes': 0,
                    'mean_improvement': 0,
                    'best_improvement': 0,
                    'recent_improvement': 0
                }
        
        return stats
    
    def force_benchmark(self, benchmark_name: str):
        """Force the environment to use a specific benchmark for the next reset"""
        if benchmark_name in self.benchmark_names:
            self.current_benchmark = benchmark_name
            logger.info(f"Forced benchmark to: {benchmark_name}")
        else:
            logger.warning(f"Unknown benchmark: {benchmark_name}")
    
    def get_current_benchmark(self) -> Optional[str]:
        """Get the currently active benchmark"""
        return self.current_benchmark
    
    def get_available_benchmarks(self) -> List[str]:
        """Get list of available benchmarks"""
        return self.benchmark_names.copy()
    
    def get_benchmark_data_summary(self) -> Dict:
        """Get summary of available benchmark data"""
        summary = {}
        for benchmark in self.benchmark_names:
            data = self.benchmark_data[benchmark]
            summary[benchmark] = {
                'data_size': data.get('data_size', 0),
                'baseline_time': data.get('baseline_time', 0),
                'best_time': data.get('best_time', 0),
                'worst_time': data.get('worst_time', 0),
                'improvement_potential': (data.get('baseline_time', 1) - data.get('best_time', 1)) / data.get('baseline_time', 1),
                'exact_configurations': len(self.exact_lookups.get(benchmark, {}))
            }
        
        return summary