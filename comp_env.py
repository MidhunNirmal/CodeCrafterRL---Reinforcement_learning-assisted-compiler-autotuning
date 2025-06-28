import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

class CompilerEnvironment(gym.Env):
    """Compiler optimization environment for RL training"""
    
    def __init__(self, benchmark_data: Dict, benchmark_name: str):
        super(CompilerEnvironment, self).__init__()
        
        self.benchmark_name = benchmark_name
        self.benchmark_data = benchmark_data[benchmark_name]
        
        # Action space: 7 compiler flags (binary)
        self.action_space = spaces.MultiBinary(7)
        
        # Observation space: current flag configuration + benchmark context
        # 7 flags + 3 context features (normalized code_size, current_performance, steps_remaining)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )
        
        # Environment state
        self.current_config = None
        self.step_count = 0
        self.max_steps = 50
        
        # Performance tracking
        self.best_time = self.benchmark_data['best_time']
        self.baseline_time = self.benchmark_data['baseline_time']
        self.worst_time = self.benchmark_data['worst_time']
        self.current_time = self.baseline_time
        
        # Create nearest neighbor model for faster lookup
        self._create_performance_model()
        
        logger.info(f"Environment initialized for {benchmark_name}")
        logger.info(f"  Baseline time: {self.baseline_time:.6f}")
        logger.info(f"  Best time: {self.best_time:.6f}")
        logger.info(f"  Improvement potential: {((self.baseline_time - self.best_time) / self.baseline_time * 100):.2f}%")
        
    def _create_performance_model(self):
        """Create a model to predict execution time from compiler configuration"""
        X_raw = self.benchmark_data['X_raw']
        y_raw = self.benchmark_data['y_raw']
        
        # Use only compiler flags (first 7 features) for configuration matching
        X_flags = X_raw[:, :7]  # First 7 columns are compiler flags
        
        # Create lookup table for exact matches
        self.exact_lookup = {}
        for i, (config, time) in enumerate(zip(X_flags, y_raw)):
            config_tuple = tuple(config.astype(int))
            if config_tuple not in self.exact_lookup:
                self.exact_lookup[config_tuple] = []
            self.exact_lookup[config_tuple].append(time)
        
        # Average multiple measurements for same configuration
        for config in self.exact_lookup:
            self.exact_lookup[config] = np.mean(self.exact_lookup[config])
        
        # Create nearest neighbor model for approximate matches
        if len(X_flags) > 5:  # Only if we have enough data
            self.nn_model = NearestNeighbors(n_neighbors=3, metric='hamming')
            self.nn_model.fit(X_flags)
            self.nn_X = X_flags
            self.nn_y = y_raw
        else:
            self.nn_model = None
        
        logger.info(f"Created performance model with {len(self.exact_lookup)} exact configurations")
    
    def reset(self):
        """Reset environment to initial state"""
        # Start with a random configuration
        self.current_config = np.random.randint(0, 2, size=7)
        self.step_count = 0
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
            'step': self.step_count
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Get current observation (state)"""
        # Normalize performance metrics
        normalized_current = (self.current_time - self.best_time) / (self.worst_time - self.best_time)
        normalized_current = np.clip(normalized_current, 0, 1)
        
        steps_remaining = (self.max_steps - self.step_count) / self.max_steps
        
        # Observation includes:
        # - 7 compiler flags
        # - normalized current performance 
        # - steps remaining ratio
        # - improvement from baseline (can be negative)
        improvement_from_baseline = (self.baseline_time - self.current_time) / self.baseline_time
        
        obs = np.concatenate([
            self.current_config.astype(np.float32),  # 7 flags
            [normalized_current],                     # 1 performance metric
            [steps_remaining],                        # 1 time remaining
            [improvement_from_baseline]               # 1 improvement metric
        ])
        
        return obs
    
    def _get_execution_time(self, config):
        """Get execution time for given configuration"""
        config_tuple = tuple(config.astype(int))
        
        # First, try exact lookup
        if config_tuple in self.exact_lookup:
            return self.exact_lookup[config_tuple]
        
        # If no exact match, try nearest neighbor
        if self.nn_model is not None:
            distances, indices = self.nn_model.kneighbors([config])
            
            # Weight predictions by inverse distance
            weights = 1.0 / (distances[0] + 1e-8)  # Add small epsilon to avoid division by zero
            weights = weights / weights.sum()
            
            predicted_time = np.average(self.nn_y[indices[0]], weights=weights)
            
            # Add some noise for exploration
            noise_factor = 0.02  # 2% noise
            noise = np.random.normal(0, noise_factor * predicted_time)
            
            return max(0.000001, predicted_time + noise)  # Ensure positive time
        
        # Fallback: use heuristic model
        return self._heuristic_execution_time(config)
    
    def _heuristic_execution_time(self, config):
        """Heuristic model when no data is available"""
        base_time = self.baseline_time
        
        # Simple heuristic based on typical flag behavior
        # These are rough estimates - actual effects vary by benchmark
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
        """Calculate reward based on execution time"""
        # Primary reward: improvement over baseline (can be negative)
        improvement_reward = (self.baseline_time - execution_time) / self.baseline_time * 100
        
        # Bonus for beating the best known time
        if execution_time < self.best_time:
            best_improvement = (self.best_time - execution_time) / self.best_time * 50
            improvement_reward += best_improvement
        
        # Small penalty for each step to encourage finding good solutions quickly
        step_penalty = -0.1
        
        # Penalty for extremely bad configurations
        if execution_time > self.worst_time:
            penalty = -10
        else:
            penalty = 0
        
        total_reward = improvement_reward + step_penalty + penalty
        
        return total_reward