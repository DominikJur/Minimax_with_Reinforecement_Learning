import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple

class DQNNetwork(nn.Module):
    """Deep Q-Network for board position evaluation"""
    
    def __init__(self, input_size=25, hidden_size=512):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 25)  # Output Q-values for each position
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """DQN Agent for learning board evaluation"""
    
    def __init__(self, state_size=25, action_size=25, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQNNetwork().to(self.device)
        self.target_network = DQNNetwork().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_moves):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.choice(valid_moves)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        
        # Mask invalid moves
        masked_q_values = q_values.clone()
        for i in range(self.action_size):
            move = (i // 5 + 1) * 10 + (i % 5 + 1)  # Convert to your move format
            if move not in valid_moves:
                masked_q_values[0][i] = float('-inf')
        
        action_idx = masked_q_values.argmax().item()
        return (action_idx // 5 + 1) * 10 + (action_idx % 5 + 1)
    
    def replay(self, batch_size=32):
        """Train the network on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class RLMinimaxBot:
    """Minimax bot using DQN for position evaluation"""
    
    def __init__(self, player: int, max_depth: int):
        self.player = player
        self.opponent = 3 - player
        self.max_depth = max_depth
        self.board = GameBoard()
        self.dqn_agent = DQNAgent()
    
    def board_to_state(self):
        """Convert board to state vector for neural network"""
        state = []
        for row in self.board.board:
            for cell in row:
                state.append(cell)
        return np.array(state, dtype=np.float32)
    
    def evaluate_position_dqn(self, player: int) -> float:
        """Use DQN to evaluate position instead of heuristics"""
        # Check terminal states first
        result = self.board.get_game_result(player)
        if result is not None:
            if result == 1:
                return 10000  # Win
            elif result == -1:
                return -10000  # Loss
            else:
                return 0  # Draw
        
        # Use DQN for evaluation
        state = self.board_to_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.dqn_agent.device)
        with torch.no_grad():
            q_values = self.dqn_agent.q_network(state_tensor)
        
        # Return average Q-value as position evaluation
        valid_moves = self.board.get_valid_moves()
        valid_q_values = []
        for move in valid_moves:
            action_idx = ((move // 10) - 1) * 5 + ((move % 10) - 1)
            valid_q_values.append(q_values[0][action_idx].item())
        
        return np.mean(valid_q_values) if valid_q_values else 0
    
    def train_from_self_play(self, episodes=1000):
        """Train the DQN through self-play"""
        for episode in range(episodes):
            self.board.reset()
            state = self.board_to_state()
            total_reward = 0
            
            while not self.board.is_terminal():
                valid_moves = self.board.get_valid_moves()
                action = self.dqn_agent.act(state, valid_moves)
                
                # Make move
                current_player = 1 if len([cell for row in self.board.board for cell in row if cell != 0]) % 2 == 0 else 2
                self.board.set_move(action, current_player)
                
                # Get reward and next state
                next_state = self.board_to_state()
                reward = self.calculate_reward(current_player)
                done = self.board.is_terminal()
                
                # Convert action to index for storage
                action_idx = ((action // 10) - 1) * 5 + ((action % 10) - 1)
                
                # Store experience
                self.dqn_agent.remember(state, action_idx, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                if len(self.dqn_agent.memory) > 32:
                    self.dqn_agent.replay()
            
            # Update target network every 100 episodes
            if episode % 100 == 0:
                self.dqn_agent.update_target_network()
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.dqn_agent.epsilon:.3f}")
    
    def calculate_reward(self, player: int) -> float:
        """Calculate reward for current position"""
        result = self.board.get_game_result(player)
        if result == 1:  # Win
            return 100
        elif result == -1:  # Loss
            return -100
        else:  # Ongoing game
            return 0  # Small reward for good positions could be added here