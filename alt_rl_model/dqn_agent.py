import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import logging

logger = logging.getLogger(__name__)

class DQNNetwork(nn.Module):
    """Deep Q-Network for compiler flag optimization with multi-binary output"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Neural network layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        
        # Separate heads for each compiler flag (dynamic number)
        self.flag_heads = nn.ModuleList([
            nn.Linear(hidden_size // 2, 2) for _ in range(self.action_size)  # 2 outputs per flag (on/off)
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # Get Q-values for each flag
        flag_q_values = []
        for head in self.flag_heads:
            flag_q_values.append(head(x))
        
        return flag_q_values


class DQNAgent:
    """DQN Agent for compiler optimization with multi-binary actions"""
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 7,  # 7 compiler flags
        lr: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update: int = 100
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
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Training counters
        self.step_count = 0
        self.episode_count = 0
        self.losses = []
        
        # Copy parameters to target network
        self.update_target_network()
        
    def update_target_network(self):
        """Copy parameters from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy for multi-binary actions"""
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
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)  # [batch_size, 7]
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values for each flag
        current_flag_q_values = self.q_network(states)
        
        # Next Q values from target network
        with torch.no_grad():
            next_flag_q_values = self.target_network(next_states)
            
            # Get max Q-value for each flag
            next_q_values = []
            for flag_q in next_flag_q_values:
                max_q = torch.max(flag_q, dim=1)[0]
                next_q_values.append(max_q)
            
            # Average the Q-values across flags for the target
            next_q_value = torch.stack(next_q_values).mean(dim=0)
            
            target_q_values = rewards + (self.gamma * next_q_value * ~dones)
        
        # Calculate loss for each flag
        total_loss = 0
        for flag_idx in range(self.action_size):
            # Get current Q-values for this flag
            current_q = current_flag_q_values[flag_idx].gather(1, actions[:, flag_idx].unsqueeze(1)).squeeze(1)
            
            # Loss for this flag
            flag_loss = F.mse_loss(current_q, target_q_values)
            total_loss += flag_loss
        
        # Average loss across flags
        loss = total_loss / self.action_size
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.step_count += 1
        
        # Update target network
        if self.step_count % self.target_update == 0:
            self.update_target_network()
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def save(self, filepath: str):
        """Save model weights"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'losses': self.losses[-1000:]  # Save last 1000 losses
        }, filepath)
        
    def load(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        if 'losses' in checkpoint:
            self.losses = checkpoint['losses']
