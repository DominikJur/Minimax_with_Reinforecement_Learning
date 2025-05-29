import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

from collections import deque

from src.dqn import DQNNetwork


class DQNAgent:
    def __init__(self, state_size=50, lr=0.001, gamma=0.95, epsilon=1.0, 
                 epsilon_min=0.05, epsilon_decay=0.995, memory_size=50000):
        self.state_size = state_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = DQNNetwork(state_size).to(self.device)
        self.target_network = DQNNetwork(state_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.update_target_network()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, reward, next_state, done):
        self.memory.append((state, reward, next_state, done))
    
    def get_value(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        elif isinstance(state, list):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.q_network(state)
        return value.item()
    
    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return 0.0
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[1] for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([e[3] for e in batch])).to(self.device)
        
        current_q_values = self.q_network(states).squeeze()
        next_q_values = self.target_network(next_states).squeeze().detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save_model(self, filepath="dqn_model.pth"):
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="dqn_model.pth"):
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found!")
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
