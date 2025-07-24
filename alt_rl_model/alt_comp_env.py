import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict
import logging
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

class AltCompilerEnvironment(gym.Env):
    """RL environment for alternate data model (from AltDataPreprocessor)"""
    def __init__(self, benchmark_data: Dict[str, pd.DataFrame], benchmark_name: str):
        super(AltCompilerEnvironment, self).__init__()
        self.benchmark_name = benchmark_name
        self.benchmark_data = benchmark_data[benchmark_name]
        self.flag_columns = [col for col in self.benchmark_data.columns if col != 'mean_exec_time']
        self.n_flags = len(self.flag_columns)
        # Action space: n_flags binary
        self.action_space = spaces.MultiBinary(self.n_flags)
        # Observation space: n_flags + 3 context features
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_flags + 3,), dtype=np.float32)
        # State
        self.current_config = None
        self.step_count = 0
        self.max_steps = 50
        # Performance metrics
        self.exec_times = self.benchmark_data['mean_exec_time'].values
        self.baseline_time = float(np.mean(self.exec_times))
        self.best_time = float(np.min(self.exec_times))
        self.worst_time = float(np.max(self.exec_times))
        self.current_time = self.baseline_time
        # Prepare lookup for config -> exec_time
        self._create_performance_model()
        logger.info(f"AltCompilerEnvironment initialized for {benchmark_name}")
        logger.info(f"  Baseline time: {self.baseline_time:.6f}")
        logger.info(f"  Best time: {self.best_time:.6f}")
        logger.info(f"  Worst time: {self.worst_time:.6f}")
    def _create_performance_model(self):
        X_raw = self.benchmark_data[self.flag_columns].values
        y_raw = self.benchmark_data['mean_exec_time'].values
        self.exact_lookup = {}
        for config, time in zip(X_raw, y_raw):
            config_tuple = tuple(config.astype(int))
            if config_tuple not in self.exact_lookup:
                self.exact_lookup[config_tuple] = []
            self.exact_lookup[config_tuple].append(time)
        for config in self.exact_lookup:
            self.exact_lookup[config] = np.mean(self.exact_lookup[config])
        if len(X_raw) > 5:
            self.nn_model = NearestNeighbors(n_neighbors=3, metric='hamming')
            self.nn_model.fit(X_raw)
            self.nn_X = X_raw
            self.nn_y = y_raw
        else:
            self.nn_model = None
    def reset(self):
        self.current_config = np.random.randint(0, 2, size=self.n_flags)
        self.step_count = 0
        self.current_time = self._get_execution_time(self.current_config)
        return self._get_observation()
    def step(self, action):
        self.step_count += 1
        if isinstance(action, (list, tuple)):
            action = np.array(action)
        action = action.astype(int).flatten()[:self.n_flags]
        self.current_config = action.copy()
        execution_time = self._get_execution_time(self.current_config)
        self.current_time = execution_time
        reward = self._calculate_reward(execution_time)
        done = self.step_count >= self.max_steps
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
        normalized_current = (self.current_time - self.best_time) / (self.worst_time - self.best_time + 1e-8)
        normalized_current = np.clip(normalized_current, 0, 1)
        steps_remaining = (self.max_steps - self.step_count) / self.max_steps
        improvement_from_baseline = (self.baseline_time - self.current_time) / self.baseline_time
        obs = np.concatenate([
            self.current_config.astype(np.float32),
            [normalized_current],
            [steps_remaining],
            [improvement_from_baseline]
        ])
        return obs
    def _get_execution_time(self, config):
        config_tuple = tuple(config.astype(int))
        if config_tuple in self.exact_lookup:
            return self.exact_lookup[config_tuple]
        if self.nn_model is not None:
            distances, indices = self.nn_model.kneighbors([config])
            weights = 1.0 / (distances[0] + 1e-8)
            weights = weights / weights.sum()
            predicted_time = np.average(self.nn_y[indices[0]], weights=weights)
            noise_factor = 0.02
            noise = np.random.normal(0, noise_factor * predicted_time)
            return max(0.000001, predicted_time + noise)
        return self._heuristic_execution_time(config)
    def _heuristic_execution_time(self, config):
        base_time = self.baseline_time
        effect = -0.05 * np.sum(config)
        modified_time = base_time * (1 + effect)
        noise = np.random.normal(0, 0.05 * base_time)
        return max(0.000001, modified_time + noise)
    def _calculate_reward(self, execution_time):
        improvement = (self.baseline_time - execution_time) / self.baseline_time
        exploration_bonus = 0.1 if self.step_count < 10 else 0
        consistency_bonus = 0.05 if improvement > 0 else -0.05
        return improvement + exploration_bonus + consistency_bonus 