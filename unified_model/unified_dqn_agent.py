import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class UnifiedDQNNetwork(nn.Module):
    """
    Unified Deep Q-Network that can handle multiple benchmarks.
    Uses benchmark context to adapt its behavior for different optimization problems.
    """
    
    def __init__(self, state_size: int, action_size: int = 7, hidden_size: int = 256):
        super(UnifiedDQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Benchmark-aware adaptation layer
        self.benchmark_adaptation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Separate heads for each compiler flag
        self.flag_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size // 2, 32),
                nn.ReLU(),
                nn.Linear(32, 2)  # Binary choice for each flag
            ) for _ in range(action_size)
        ])
        
        # Value head for advantage calculation (Dueling DQN)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Shared feature extraction
        shared_features = self.shared_layers(x)
        
        # Benchmark-aware adaptation
        adapted_features = self.benchmark_adaptation(shared_features)
        
        # Get Q-values for each flag
        flag_q_values = []
        for head in self.flag_heads:
            flag_q_values.append(head(adapted_features))
        
        # Get state value for Dueling DQN
        state_value = self.value_head(adapted_features)
        
        # Combine advantage and value (Dueling DQN architecture)
        dueling_q_values = []
        for flag_q in flag_q_values:
            advantage = flag_q - flag_q.mean(dim=1, keepdim=True)
            dueling_q = state_value + advantage
            dueling_q_values.append(dueling_q)
        
        return dueling_q_values


class UnifiedDQNAgent:
    """
    Unified DQN Agent that can optimize compiler flags across multiple benchmarks.
    This single agent learns to handle all benchmarks instead of having separate agents.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 7,
        lr: float = 0.0005,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 50000,  # Larger memory for multiple benchmarks
        batch_size: int = 64,      # Larger batch size
        target_update: int = 1000,  # Less frequent updates for stability
        benchmark_names: Optional[List[str]] = None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update = target_update
        self.benchmark_names = benchmark_names or []
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Unified agent using device: {self.device}")
        
        # Networks
        self.q_network = UnifiedDQNNetwork(state_size, action_size).to(self.device)
        self.target_network = UnifiedDQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.9)
        
        # Experience replay buffer with benchmark information
        self.memory = deque(maxlen=memory_size)
        
        # Training statistics
        self.step_count = 0
        self.episode_count = 0
        self.losses = []
        self.benchmark_performance = {benchmark: [] for benchmark in self.benchmark_names}
        
        # Priority weights for different loss components
        self.flag_loss_weight = 1.0
        self.value_loss_weight = 0.5
        
        # Copy parameters to target network
        self.update_target_network()
        
        logger.info(f"Unified DQN Agent initialized")
        logger.info(f"  State size: {state_size}")
        logger.info(f"  Action size: {action_size}")
        logger.info(f"  Memory size: {memory_size}")
        logger.info(f"  Benchmarks: {len(self.benchmark_names)}")
        
    def update_target_network(self):
        """Copy parameters from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done, benchmark_name: Optional[str] = None):
        """Store experience in replay buffer with benchmark information"""
        self.memory.append((state, action, reward, next_state, done, benchmark_name))
        
    def act(self, state, training=True, benchmark_name: Optional[str] = None):
        """Choose action using epsilon-greedy policy with benchmark awareness"""
        if training and np.random.random() <= self.epsilon:
            # Random action for exploration
            return np.random.randint(0, 2, size=self.action_size)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values for each flag
        with torch.no_grad():
            flag_q_values = self.q_network(state_tensor)
        
        # Choose best action for each flag
        actions = []
        for flag_q in flag_q_values:
            action = torch.argmax(flag_q, dim=1).cpu().numpy()[0]
            actions.append(action)
        
        return np.array(actions)
    
    def replay(self):
        """Train the model on a batch of experiences from multiple benchmarks"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, benchmark_names = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values for each flag
        current_flag_q_values = self.q_network(states)
        
        # Next Q values from target network
        with torch.no_grad():
            next_flag_q_values = self.target_network(next_states)
            
            # Double DQN: use main network to select actions, target network to evaluate
            next_main_q_values = self.q_network(next_states)
            
            # Get max Q-value for each flag using Double DQN
            next_q_values = []
            for flag_idx in range(self.action_size):
                # Select action using main network
                next_actions = torch.argmax(next_main_q_values[flag_idx], dim=1)
                # Evaluate action using target network
                next_q = next_flag_q_values[flag_idx].gather(1, next_actions.unsqueeze(1)).squeeze(1)
                next_q_values.append(next_q)
            
            # Average Q-values across flags for target calculation
            next_q_value = torch.stack(next_q_values).mean(dim=0)
            target_q_values = rewards + (self.gamma * next_q_value * ~dones)
        
        # Calculate loss for each flag
        total_flag_loss = 0
        for flag_idx in range(self.action_size):
            # Get current Q-values for this flag
            current_q = current_flag_q_values[flag_idx].gather(1, actions[:, flag_idx].unsqueeze(1)).squeeze(1)
            
            # Smooth L1 loss (Huber loss) for better stability
            flag_loss = F.smooth_l1_loss(current_q, target_q_values)
            total_flag_loss += flag_loss
        
        # Average loss across flags
        flag_loss = total_flag_loss / self.action_size
        
        # Total loss
        total_loss = self.flag_loss_weight * flag_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.step_count += 1
        
        # Update target network
        if self.step_count % self.target_update == 0:
            self.update_target_network()
            logger.info(f"Target network updated at step {self.step_count}")
        
        loss_value = total_loss.item()
        self.losses.append(loss_value)
        
        return {
            'total_loss': loss_value,
            'flag_loss': flag_loss.item(),
            'epsilon': self.epsilon,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def update_benchmark_performance(self, benchmark_name: str, performance_metric: float):
        """Update performance tracking for a specific benchmark"""
        if benchmark_name in self.benchmark_performance:
            self.benchmark_performance[benchmark_name].append(performance_metric)
        
    def get_benchmark_performance_stats(self) -> Dict:
        """Get performance statistics across all benchmarks"""
        stats = {}
        for benchmark, performances in self.benchmark_performance.items():
            if performances:
                stats[benchmark] = {
                    'mean': np.mean(performances[-100:]),  # Last 100 episodes
                    'std': np.std(performances[-100:]),
                    'best': max(performances),
                    'recent_trend': np.mean(performances[-10:]) - np.mean(performances[-50:-10]) if len(performances) >= 50 else 0
                }
            else:
                stats[benchmark] = {'mean': 0, 'std': 0, 'best': 0, 'recent_trend': 0}
        
        return stats
    
    def save(self, filepath: str):
        """Save model weights and training state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'losses': self.losses[-1000:],  # Save last 1000 losses
            'benchmark_performance': {k: v[-100:] for k, v in self.benchmark_performance.items()},  # Last 100 per benchmark
            'benchmark_names': self.benchmark_names
        }, filepath)
        
        logger.info(f"Unified model saved to {filepath}")
        
    def load(self, filepath: str):
        """Load model weights and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint.get('episode_count', 0)
        
        if 'losses' in checkpoint:
            self.losses = checkpoint['losses']
        
        if 'benchmark_performance' in checkpoint:
            self.benchmark_performance.update(checkpoint['benchmark_performance'])
        
        if 'benchmark_names' in checkpoint:
            self.benchmark_names = checkpoint['benchmark_names']
        
        logger.info(f"Unified model loaded from {filepath}")
        logger.info(f"  Step count: {self.step_count}")
        logger.info(f"  Epsilon: {self.epsilon}")
    
    def get_network_stats(self) -> Dict:
        """Get statistics about the neural network"""
        total_params = sum(p.numel() for p in self.q_network.parameters())
        trainable_params = sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'memory_usage': len(self.memory),
            'epsilon': self.epsilon,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }