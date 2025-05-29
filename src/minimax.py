from typing import List, Optional, Tuple
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os

from src.board import GameBoard
from tqdm import tqdm

# Advanced DQN Network with modern techniques
class DQNNetwork(nn.Module):
    def __init__(self, input_size=50, hidden_size=1024, output_size=1):
        super(DQNNetwork, self).__init__()
        
        # Feature extraction layers with residual connections
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # Deep feature processing with skip connections
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        # Attention mechanism for pattern focus
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_size)
        
        # Value estimation pathway
        self.value_path = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ELU(),
            nn.Linear(hidden_size//4, output_size),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(0.15)
        
    def forward(self, x):
        # Input processing
        x = torch.relu(self.input_norm(self.input_layer(x)))
        identity = x
        
        # Deep feature extraction with residuals
        x = torch.relu(self.norm1(self.hidden1(x)))
        x = self.dropout(x)
        x = torch.relu(self.norm2(self.hidden2(x)))
        x = x + identity  # Skip connection
        
        x = torch.relu(self.norm3(self.hidden3(x)))
        x = self.dropout(x)
        
        # Self-attention for pattern recognition
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension for attention
        
        attended, _ = self.attention(x, x, x)
        x = self.attention_norm(attended + x)  # Residual connection
        x = x.squeeze(1)  # Remove sequence dimension
        
        # Value estimation
        return self.value_path(x)

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

class MinimaxBot:
    """Clean DQN-enhanced Minimax bot"""

    def __init__(self, player: int, max_depth: int, use_dqn=True, model_path="dqn_model.pth"):
        self.player = player
        self.opponent = 3 - player
        self.max_depth = max_depth
        self.board = GameBoard()
        self.use_dqn = use_dqn
        self.model_path = model_path
        self.is_training = False
        
        # Initialize DQN
        if self.use_dqn:
            self.dqn_agent = DQNAgent()
            if os.path.exists(model_path):
                success = self.dqn_agent.load_model(model_path)
                if not success:
                    print("Starting with fresh model")
        else:
            self.dqn_agent = None

    def board_to_features(self):
        """Convert board to feature vector - simplified but effective"""
        features = []
        board = self.board.board
        
        # Basic board state (25 features)
        for row in board:
            for cell in row:
                if cell == self.player:
                    features.append(1.0)
                elif cell == self.opponent:
                    features.append(-1.0)
                else:
                    features.append(0.0)
        
        # Pattern analysis (20 features)
        my_win_threats = 0
        opp_win_threats = 0
        my_win_opps = 0
        opp_win_opps = 0
        
        for pattern in self.board.win_patterns:
            my_count = sum(1 for r, c in pattern if board[r][c] == self.player)
            opp_count = sum(1 for r, c in pattern if board[r][c] == self.opponent)
            empty = sum(1 for r, c in pattern if board[r][c] == 0)
            
            if opp_count == 0:
                if my_count == 3:
                    my_win_threats += 1
                elif my_count == 2 and empty >= 1:
                    my_win_opps += 1
            
            if my_count == 0:
                if opp_count == 3:
                    opp_win_threats += 1
                elif opp_count == 2 and empty >= 1:
                    opp_win_opps += 1
        
        my_lose_dangers = 0
        opp_lose_dangers = 0
        
        for pattern in self.board.lose_patterns:
            my_count = sum(1 for r, c in pattern if board[r][c] == self.player)
            opp_count = sum(1 for r, c in pattern if board[r][c] == self.opponent)
            
            if opp_count == 0 and my_count == 2:
                my_lose_dangers += 1
            if my_count == 0 and opp_count == 2:
                opp_lose_dangers += 1
        
        features.extend([
            min(my_win_threats / 5.0, 1.0),
            min(opp_win_threats / 5.0, 1.0),
            min(my_win_opps / 10.0, 1.0),
            min(opp_win_opps / 10.0, 1.0),
            min(my_lose_dangers / 10.0, 1.0),
            min(opp_lose_dangers / 10.0, 1.0)
        ])
        
        # Simple positional features (5 features)
        center_moves = [33, 32, 34, 23, 43]
        my_center = sum(1 for move in center_moves 
                       if board[(move//10)-1][(move%10)-1] == self.player)
        opp_center = sum(1 for move in center_moves 
                        if board[(move//10)-1][(move%10)-1] == self.opponent)
        
        total_pieces = sum(1 for row in board for cell in row if cell != 0)
        
        features.extend([
            my_center / 5.0,
            opp_center / 5.0,
            total_pieces / 25.0,
            len(self.board.get_valid_moves()) / 25.0
        ])
        
        # Pad to 50 features
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50], dtype=np.float32)

    def evaluate_position(self, player: int) -> float:
        """Combined DQN + heuristic evaluation"""
        # Terminal states
        result = self.board.get_game_result(player)
        if result is not None:
            return 10000 if result == 1 else (-10000 if result == -1 else 0)

        heuristic_score = self._evaluate_heuristic(player)
        
        # Get DQN score
        dqn_score = 0.0
        if self.use_dqn and self.dqn_agent is not None:
            try:
                state = self.board_to_features()
                if player != self.player:
                    # Flip perspective for opponent
                    for i in range(25):
                        if state[i] != 0:
                            state[i] = -state[i]
                
                dqn_value = self.dqn_agent.get_value(state)
                dqn_score = dqn_value * 2000
                
            except Exception as e:
                print(f"DQN eval failed: {e}")
                dqn_score = 0.0
        
        # Combine scores
        if self.is_training:
            return 0.3 * dqn_score + 0.7 * heuristic_score
        else:
            return 0.6 * dqn_score + 0.4 * heuristic_score

    def _evaluate_heuristic(self, player: int) -> float:
        """Clean heuristic evaluation"""
        score = 0
        board = self.board.board
        opponent = 3 - player

        # Win patterns
        for pattern in self.board.win_patterns:
            my_count = sum(1 for r, c in pattern if board[r][c] == player)
            opp_count = sum(1 for r, c in pattern if board[r][c] == opponent)
            
            if opp_count == 0:
                if my_count == 3:
                    score += 200
                elif my_count == 2:
                    score += 20
                elif my_count == 1:
                    score += 2
            elif my_count == 0:
                if opp_count == 3:
                    score -= 200
                elif opp_count == 2:
                    score -= 20
                elif opp_count == 1:
                    score -= 2

        # Lose patterns
        for pattern in self.board.lose_patterns:
            my_count = sum(1 for r, c in pattern if board[r][c] == player)
            opp_count = sum(1 for r, c in pattern if board[r][c] == opponent)

            if opp_count == 0 and my_count == 2:
                score -= 100
            elif my_count == 0 and opp_count == 2:
                score += 100

        return score

    def minimax(self, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[float, Optional[int]]:
        """Standard minimax with alpha-beta pruning"""
        if depth == 0 or self.board.is_terminal():
            return self.evaluate_position(self.player), None

        current_player = self.player if maximizing else self.opponent
        moves = self.board.get_valid_moves()
        best_move = None

        # Opening book
        if self.board.is_empty():
            return None, 33
        
        if maximizing:
            max_eval = float("-inf")
            for move in moves:
                self.board.set_move(move, current_player)
                
                if self.board.check_win(current_player):
                    self.board.undo_move(move)
                    return None, move
                
                if self.board.check_lose(current_player):
                    self.board.undo_move(move)
                    continue

                eval_score, _ = self.minimax(depth - 1, alpha, beta, False)
                self.board.undo_move(move)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break

            return max_eval, best_move
        else:
            min_eval = float("inf")
            for move in moves:
                self.board.set_move(move, current_player)
                
                if self.board.check_win(current_player):
                    self.board.undo_move(move)
                    return None, move
                
                if self.board.check_lose(current_player):
                    self.board.undo_move(move)
                    continue

                eval_score, _ = self.minimax(depth - 1, alpha, beta, True)
                self.board.undo_move(move)

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break

            return min_eval, best_move

    def get_best_move(self, training_mode=False) -> int:
        """Get best move"""
        depth = max(2, self.max_depth - 1) if training_mode else self.max_depth
        _, best_move = self.minimax(depth, float("-inf"), float("inf"), True)

        if best_move is None:
            best_move = self.board.get_valid_moves()[0]
        
        if not training_mode and not self.is_training:
            time.sleep(0.1)
        
        return best_move

    def get_heuristic_move(self) -> int:
        """Get move using only heuristics"""
        old_dqn = self.use_dqn
        self.use_dqn = False
        move = self.get_best_move(training_mode=True)
        self.use_dqn = old_dqn
        return move

    def train_dqn(self, episodes=3000, save_interval=300):
        """Clean training with curriculum learning"""
        if not self.use_dqn:
            print("DQN not enabled!")
            return
        
        self.is_training = True
        print(f"Training DQN for {episodes} episodes...")
        
        # Training phases
        random_phase = int(episodes * 0.15)     # 15% random
        heuristic_phase = int(episodes * 0.75)  # 75% vs heuristic
        # 10% self-play at the end
        
        for episode in tqdm(range(episodes), desc="Training"):
            self.board.reset()
            current_player = 1
            episode_data = []
            
            while not self.board.is_terminal():
                state = self.board_to_features()
                
                # Choose opponent strategy
                if episode < random_phase:
                    if random.random() < 0.5:
                        move = random.choice(self.board.get_valid_moves())
                    else:
                        old_player = self.player
                        self.player = current_player
                        move = self.get_heuristic_move()
                        self.player = old_player
                
                elif episode < heuristic_phase:
                    if random.random() < self.dqn_agent.epsilon:
                        move = random.choice(self.board.get_valid_moves())
                    else:
                        old_player = self.player
                        self.player = current_player
                        move = self.get_heuristic_move()
                        self.player = old_player
                else:
                    # Self-play
                    old_player = self.player
                    self.player = current_player
                    move = self.get_best_move(training_mode=True)
                    self.player = old_player
                
                episode_data.append((current_player, state.copy()))
                self.board.set_move(move, current_player)
                current_player = 3 - current_player
            
            # Store experiences
            final_result_p1 = self.board.get_game_result(1)
            final_result_p2 = self.board.get_game_result(2)
            
            for i, (player, state) in enumerate(episode_data):
                reward = 1.0 if (player == 1 and final_result_p1 == 1) or (player == 2 and final_result_p2 == 1) else \
                        -1.0 if (player == 1 and final_result_p1 == -1) or (player == 2 and final_result_p2 == -1) else 0.0
                
                next_state = episode_data[i + 1][1] if i < len(episode_data) - 1 else state
                done = i == len(episode_data) - 1
                
                self.dqn_agent.remember(state, reward, next_state, done)
            
            # Train
            if len(self.dqn_agent.memory) > 64:
                self.dqn_agent.replay(batch_size=32)
            
            if episode % 100 == 0:
                self.dqn_agent.update_target_network()
            
            if episode % save_interval == 0 and episode > 0:
                self.dqn_agent.save_model(self.model_path)
                print(f"Episode {episode}: ε={self.dqn_agent.epsilon:.3f}")
        
        self.is_training = False
        self.dqn_agent.save_model(self.model_path)
        print("Training completed!")

    def save_model(self, filepath=None):
        if self.dqn_agent:
            self.dqn_agent.save_model(filepath or self.model_path)

    def load_model(self, filepath=None):
        if self.dqn_agent:
            return self.dqn_agent.load_model(filepath or self.model_path)
        return False