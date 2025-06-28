"""
Unified Model Package for Compiler Flag Optimization

This package provides a unified approach to compiler flag optimization using Deep Q-Learning.
Instead of training separate models for each benchmark, this package trains a single model
that can handle all benchmarks through benchmark-aware state representations.

Key Components:
- BenchmarkEncoder: Encodes benchmark information into state representations
- UnifiedDQNAgent: Single DQN agent that handles multiple benchmarks
- UnifiedCompilerEnvironment: Environment that cycles through benchmarks
- UnifiedCompilerOptimizationTrainer: Main trainer for the unified approach

Advantages over separate models:
1. Shared learning across benchmarks
2. More efficient memory usage
3. Transfer learning between similar benchmarks
4. Easier deployment and maintenance
5. Single model to manage instead of many
"""

from .benchmark_encoder import BenchmarkEncoder
from .unified_dqn_agent import UnifiedDQNAgent, UnifiedDQNNetwork
from .unified_environment import UnifiedCompilerEnvironment
from .unified_trainer import UnifiedCompilerOptimizationTrainer

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Unified Model for Compiler Flag Optimization"

__all__ = [
    'BenchmarkEncoder',
    'UnifiedDQNAgent', 
    'UnifiedDQNNetwork',
    'UnifiedCompilerEnvironment',
    'UnifiedCompilerOptimizationTrainer'
]