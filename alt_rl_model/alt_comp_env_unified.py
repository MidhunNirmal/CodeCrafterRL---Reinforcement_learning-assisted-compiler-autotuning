import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class AltUnifiedCompilerEnvironment(gym.Env):
    """Unified RL environment for alternate data with 7 compiler flags"""
    def __init__(self, benchmark_data: Dict[str, pd.DataFrame], benchmark_encoder: LabelEncoder = None):
        super(AltUnifiedCompilerEnvironment, self).__init__()
        
        self.benchmark_data = benchmark_data
        self.benchmark_names = list(benchmark_data.keys())
        
        # Create benchmark encoder if not provided
        if benchmark_encoder is None:
            self.benchmark_encoder = LabelEncoder()
            self.benchmark_encoder.fit(self.benchmark_names)
        else:
            self.benchmark_encoder = benchmark_encoder
        
        # Action space: 7 compiler flags (binary)
        self.action_space = spaces.MultiBinary(7)
        
        # Get feature dimensions from the first benchmark
        first_benchmark = list(benchmark_data.keys())[0]
        first_df = benchmark_data[first_benchmark]
        # Count features: flags + static features (excluding mean_exec_time)
        flag_cols = [col for col in first_df.columns if col in [
            'funsafe_math_optimizations', 'fno_guess_branch_probability', 'fno_ivopts',
            'fno_tree_loop_optimize', 'fno_inline_functions', 'funroll_all_loops', 'o2'
        ]]
        static_feature_cols = [col for col in first_df.columns if col not in flag_cols + ['mean_exec_time']]
        
        num_flags = len(flag_cols)
        num_static_features = len(static_feature_cols)
        benchmark_encoding_size = len(self.benchmark_names)
        
        # Observation space: 
        # - 7 compiler flags
        # - all static features (normalized)
        # - benchmark encoding (one-hot)
        # - 3 context features (performance, steps, improvement)
        total_obs_size = num_flags + num_static_features + benchmark_encoding_size + 3
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32
        )
        
        # Store feature information
        self.flag_columns = flag_cols
        self.static_feature_columns = static_feature_cols
        self.num_flags = num_flags
        self.num_static_features = num_static_features
        
        # Environment state
        self.current_config = None
        self.current_benchmark = None
        self.current_benchmark_data = None
        self.step_count = 0
        self.max_steps = 50
        
        # Performance tracking
        self.baseline_time = None
        self.best_time = None
        self.worst_time = None
        self.current_time = None
        
        # Create performance models for each benchmark
        self.performance_models = {}
        self._create_performance_models()
        
        logger.info(f"Unified environment initialized for {len(self.benchmark_names)} benchmarks")
        logger.info(f"State size: {total_obs_size} (7 flags + {num_static_features} static features + {benchmark_encoding_size} benchmarks + 3 context)")
        logger.info(f"Action size: 7 (7 binary flags)")
        
    def _create_performance_models(self):
        """Create performance lookup models for each benchmark"""
        for benchmark_name in self.benchmark_names:
            df = self.benchmark_data[benchmark_name]
            
            # Only use the 7 flag columns for performance prediction
            flag_cols = [col for col in df.columns if col in [
                'funsafe_math_optimizations', 'fno_guess_branch_probability', 'fno_ivopts',
                'fno_tree_loop_optimize', 'fno_inline_functions', 'funroll_all_loops', 'o2'
            ]]
            
            X_raw = df[flag_cols].values
            y_raw = df['mean_exec_time'].values
            
            # Create exact lookup
            exact_lookup = {}
            for config, time in zip(X_raw, y_raw):
                config_tuple = tuple(config.astype(int))
                if config_tuple not in exact_lookup:
                    exact_lookup[config_tuple] = []
                exact_lookup[config_tuple].append(time)
            
            # Average multiple measurements
            for config in exact_lookup:
                exact_lookup[config] = np.mean(exact_lookup[config])
            
            # Create nearest neighbor model (only for flags)
            if len(X_raw) > 5:
                nn_model = NearestNeighbors(n_neighbors=3, metric='hamming')
                nn_model.fit(X_raw)
                nn_data = {'X': X_raw, 'y': y_raw, 'model': nn_model}
            else:
                nn_data = None
            
            self.performance_models[benchmark_name] = {
                'exact_lookup': exact_lookup,
                'nn_model': nn_data,
                'baseline_time': np.mean(y_raw),
                'best_time': np.min(y_raw),
                'worst_time': np.max(y_raw)
            }
    
    def reset(self, benchmark_name: str = None):
        """Reset environment to initial state"""
        # Select benchmark if not specified
        if benchmark_name is None:
            benchmark_name = np.random.choice(self.benchmark_names)
        
        self.current_benchmark = benchmark_name
        self.current_benchmark_data = self.benchmark_data[benchmark_name]
        
        # Start with a random configuration
        self.current_config = np.random.randint(0, 2, size=7)
        self.step_count = 0
        
        # Set performance metrics for this benchmark
        perf_model = self.performance_models[benchmark_name]
        self.baseline_time = perf_model['baseline_time']
        self.best_time = perf_model['best_time']
        self.worst_time = perf_model['worst_time']
        self.current_time = self._get_execution_time(self.current_config)
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        self.step_count += 1
        
        # Ensure action is the right shape and type
        if isinstance(action, (list, tuple)):
            action = np.array(action)
        action = action.astype(int).flatten()[:7]  # Ensure 7 flags
        
        # Update configuration
        self.current_config = action.copy()
        
        # Get execution time for this configuration
        execution_time = self._get_execution_time(self.current_config)
        self.current_time = execution_time
        
        # Calculate reward
        reward = self._calculate_reward(execution_time)
        
        # Check if done
        done = self.step_count >= self.max_steps
        
        # Calculate improvement percentage
        improvement = (self.baseline_time - execution_time) / self.baseline_time
        
        info = {
            'execution_time': execution_time,
            'best_time': self.best_time,
            'baseline_time': self.baseline_time,
            'improvement': improvement,
            'config': self.current_config.copy(),
            'benchmark': self.current_benchmark,
            'step': self.step_count
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Get current observation (state) with benchmark context"""
        # Normalize performance metrics
        normalized_current = (self.current_time - self.best_time) / (self.worst_time - self.best_time + 1e-8)
        normalized_current = np.clip(normalized_current, 0, 1)
        
        steps_remaining = (self.max_steps - self.step_count) / self.max_steps
        
        # Benchmark encoding (one-hot)
        benchmark_encoding = np.zeros(len(self.benchmark_names))
        benchmark_idx = self.benchmark_encoder.transform([self.current_benchmark])[0]
        benchmark_encoding[benchmark_idx] = 1.0
        
        # Improvement from baseline (can be negative)
        improvement_from_baseline = (self.baseline_time - self.current_time) / self.baseline_time
        
        # Get static features from current benchmark data (use mean across all samples)
        static_features = self.current_benchmark_data[self.static_feature_columns].mean().values
        
        obs = np.concatenate([
            self.current_config.astype(np.float32),  # 7 flags
            static_features.astype(np.float32),      # all static features (already normalized by preprocessor)
            benchmark_encoding.astype(np.float32),   # benchmark encoding
            [normalized_current],                     # 1 performance metric
            [steps_remaining],                        # 1 time remaining
            [improvement_from_baseline]               # 1 improvement metric
        ])
        
        return obs
    
    def _get_execution_time(self, config):
        """Get execution time for given configuration"""
        perf_model = self.performance_models[self.current_benchmark]
        config_tuple = tuple(config.astype(int))
        
        # First, try exact lookup
        if config_tuple in perf_model['exact_lookup']:
            return perf_model['exact_lookup'][config_tuple]
        
        # If no exact match, try nearest neighbor
        if perf_model['nn_model'] is not None:
            distances, indices = perf_model['nn_model']['model'].kneighbors([config])
            
            # Weight predictions by inverse distance
            weights = 1.0 / (distances[0] + 1e-8)
            weights = weights / weights.sum()
            
            predicted_time = np.average(perf_model['nn_model']['y'][indices[0]], weights=weights)
            
            # Add some noise for exploration
            noise_factor = 0.02
            noise = np.random.normal(0, noise_factor * predicted_time)
            
            return max(0.000001, predicted_time + noise)
        
        # Fallback: use heuristic model
        return self._heuristic_execution_time(config)
    
    def _heuristic_execution_time(self, config):
        """Heuristic model when no data is available"""
        base_time = self.baseline_time
        
        # Simple heuristic based on typical flag behavior
        flag_effects = [
            -0.05,  # -funsafe-math-optimizations (usually good)
            0.02,   # -fno-guess-branch-probability (usually bad)
            0.01,   # -fno-ivopts (usually bad)
            0.08,   # -fno-tree-loop-optimize (usually bad)
            0.03,   # -fno-inline-functions (usually bad)
            -0.02,  # -funroll-all-loops (mixed)
            -0.15   # -O2 (usually very good)
        ]
        
        # Calculate effect
        total_effect = sum(config[i] * flag_effects[i] for i in range(7))
        
        # Apply effect with some randomness
        modified_time = base_time * (1 + total_effect)
        
        # Add noise
        noise = np.random.normal(0, 0.05 * base_time)
        
        return max(0.000001, modified_time + noise)
    
    def _calculate_reward(self, execution_time):
        """Calculate reward based on performance improvement"""
        # Base reward from improvement
        improvement = (self.baseline_time - execution_time) / self.baseline_time
        
        # Exploration bonus for early steps
        exploration_bonus = 0.1 if self.step_count < 10 else 0
        
        # Consistency bonus
        consistency_bonus = 0.05 if improvement > 0 else -0.05
        
        return improvement + exploration_bonus + consistency_bonus
    
    def get_benchmark_encoder(self):
        """Get the benchmark encoder for use in other components"""
        return self.benchmark_encoder 