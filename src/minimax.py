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

# Simplified DQN Network for faster training
class DQNNetwork(nn.Module):
    def __init__(self, input_size=25, hidden_size=256, output_size=1):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),  # Reduced dropout
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size=25, lr=0.001, gamma=0.95, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995, memory_size=50000):  # Reduced memory
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
    
    def replay(self, batch_size=32):  # Reduced batch size for faster training
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
            'memory_size': len(self.memory)
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="dqn_model.pth"):
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found!")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        
        print(f"Model loaded from {filepath}")
        return True

class MinimaxBot:
    """Enhanced Minimax bot with DQN evaluation"""

    def __init__(self, player: int, max_depth: int, use_dqn=True, model_path="dqn_model.pth"):
        self.player = player
        self.opponent = 3 - player
        self.max_depth = max_depth
        self.training_depth = max(1, max_depth - 2)  # Use shallower depth during training
        self.board = GameBoard()
        self.use_dqn = use_dqn
        self.model_path = model_path
        self.is_training = False  # Flag to track training mode
        
        # Move ordering for better pruning
        self.center_moves = [33, 32, 34, 23, 43, 22, 24, 42, 44]
        
        # Initialize DQN if requested
        if self.use_dqn:
            self.dqn_agent = DQNAgent()
            if os.path.exists(model_path):
                self.dqn_agent.load_model(model_path)
                print(f"Loaded DQN model: {model_path}")
            else:
                print(f"No DQN model found at {model_path}")
        else:
            self.dqn_agent = None

    def board_to_state(self):
        """Convert board to state vector for DQN"""
        state = []
        for row in self.board.board:
            for cell in row:
                if cell == self.player:
                    state.append(1.0)
                elif cell == self.opponent:
                    state.append(-1.0)
                else:
                    state.append(0.0)
        return np.array(state, dtype=np.float32)

    def order_moves(self, moves: List[int]) -> List[int]:
        """Order moves for better alpha-beta pruning"""
        # Prioritize center moves
        center_first = []
        others = []
        
        for move in moves:
            if move in self.center_moves:
                center_first.append(move)
            else:
                others.append(move)
        
        return center_first + others

    def evaluate_position(self, player: int) -> float:
        """Enhanced evaluation function - uses DQN if available, falls back to heuristics"""
        # Check terminal states first
        result = self.board.get_game_result(player)
        if result is not None:
            if result == 1:
                return 10000  # Win
            elif result == -1:
                return -10000  # Loss
            else:
                return 0  # Draw

        # Use DQN evaluation if available and trained
        if self.use_dqn and self.dqn_agent is not None:
            try:
                state = self.board_to_state()
                
                # Adjust perspective for the evaluating player
                if player != self.player:
                    state = -state
                
                value = self.dqn_agent.get_value(state)
                scaled_value = value * 5000  # Scale from [-1,1] to [-5000,5000]
                
                return scaled_value if player == self.player else -scaled_value
            except Exception as e:
                print(f"DQN evaluation failed: {e}, falling back to heuristics")
        
        # Fallback to original heuristic evaluation
        return self._evaluate_patterns(player)

    def _evaluate_patterns(self, player: int) -> float:
        """Original heuristic evaluation - optimized version"""
        score = 0
        
        # Cache board lookups
        board = self.board.board
        opponent = 3 - player

        # Optimized pattern evaluation
        for pattern in self.board.win_patterns:
            player_count = 0
            opponent_count = 0
            
            for r, c in pattern:
                cell = board[r][c]
                if cell == player:
                    player_count += 1
                elif cell == opponent:
                    opponent_count += 1
            
            if opponent_count == 0:
                if player_count == 3:
                    score += 100
                elif player_count == 2:
                    score += 10
                elif player_count == 1:
                    score += 1
            elif player_count == 0:
                if opponent_count == 3:
                    score -= 100
                elif opponent_count == 2:
                    score -= 10
                elif opponent_count == 1:
                    score -= 1

        for pattern in self.board.lose_patterns:
            player_count = 0
            opponent_count = 0
            
            for r, c in pattern:
                cell = board[r][c]
                if cell == player:
                    player_count += 1
                elif cell == opponent:
                    opponent_count += 1

            if opponent_count == 0:
                if player_count == 2:
                    score -= 50
                elif player_count == 1:
                    score -= 5
            elif player_count == 0:
                if opponent_count == 2:
                    score += 50
                elif opponent_count == 1:
                    score += 5

        return score

    def minimax(self, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[float, Optional[int]]:
        """Optimized minimax algorithm"""
        if depth == 0 or self.board.is_terminal():
            return self.evaluate_position(self.player), None

        current_player = self.player if maximizing else self.opponent
        opponent_player = self.opponent if maximizing else self.player
        moves = self.order_moves(self.board.get_valid_moves())  # Order moves for better pruning
        best_move = None

        # Early game optimization
        if self.board.is_empty():
            return None, 33 
            
        if self.board.board == [[0,0,0,0,0],
                                [0,0,0,0,0],
                                [0,0,1,0,0],
                                [0,0,0,0,0],
                                [0,0,0,0,0]]:
            return None, 32
        
        if maximizing:
            max_eval = float("-inf")
            for move in moves:
                self.board.set_move(move, current_player)

                # Check for immediate win
                if depth == self.max_depth and self.board.check_win(current_player):
                    self.board.undo_move(move)
                    return None, move

                # Skip losing moves
                if self.board.check_lose(current_player):
                    self.board.undo_move(move)
                    continue

                # Check for defensive moves
                if depth == self.max_depth:
                    self.board.undo_move(move)
                    self.board.set_move(move, opponent_player)
                    if self.board.check_win(opponent_player):
                        self.board.undo_move(move)
                        return None, move
                    self.board.undo_move(move)
                    self.board.set_move(move, current_player)

                eval_score, _ = self.minimax(depth - 1, alpha, beta, False)
                self.board.undo_move(move)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning

            return max_eval, best_move

        else:
            min_eval = float("inf")
            
            for move in moves:
                self.board.set_move(move, current_player)
                
                # Check for immediate win
                if depth == self.max_depth and self.board.check_win(current_player):
                    self.board.undo_move(move)
                    return None, move

                # Skip losing moves
                if self.board.check_lose(current_player):
                    self.board.undo_move(move)
                    continue

                # Check for defensive moves
                if depth == self.max_depth:
                    self.board.undo_move(move)
                    self.board.set_move(move, opponent_player)
                    if self.board.check_win(opponent_player):
                        self.board.undo_move(move)
                        return None, move
                    self.board.undo_move(move)
                    self.board.set_move(move, current_player)

                eval_score, _ = self.minimax(depth - 1, alpha, beta, True)
                self.board.undo_move(move)

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning

            return min_eval, best_move

    def get_best_move(self, training_mode=False) -> int:
        """Get the best move using minimax algorithm"""
        # Use different depth for training vs playing
        depth = self.training_depth if training_mode else self.max_depth
        
        _, best_move = self.minimax(depth, float("-inf"), float("inf"), True)

        if best_move is None:
            valid_moves = self.board.get_valid_moves()
            best_move = valid_moves[0]
        
        # Only add delay when actually playing, not during training
        if not training_mode and not self.is_training:
            time.sleep(0.1)  # Much shorter delay for actual gameplay
        
        return best_move

    def get_fast_move(self) -> int:
        """Get a quick move for training - uses shallow search or heuristics"""
        valid_moves = self.board.get_valid_moves()
        
        # Check for immediate wins/losses first
        for move in valid_moves:
            self.board.set_move(move, self.player)
            if self.board.check_win(self.player):
                self.board.undo_move(move)
                return move
            self.board.undo_move(move)
        
        # Check for blocks
        for move in valid_moves:
            self.board.set_move(move, self.opponent)
            if self.board.check_win(self.opponent):
                self.board.undo_move(move)
                return move
            self.board.undo_move(move)
        
        # Use very shallow minimax (depth 1-2)
        _, best_move = self.minimax(1, float("-inf"), float("inf"), True)
        return best_move if best_move else random.choice(valid_moves)

    # Optimized training methods for DQN
    def train_dqn(self, episodes=2000, save_interval=200):
        """Optimized DQN training"""
        if not self.use_dqn or self.dqn_agent is None:
            print("DQN not enabled!")
            return
        
        self.is_training = True
        print(f"Training DQN for {episodes} episodes...")
        print(f"Training on device: {self.dqn_agent.device}")
        
        # Training phases with different strategies
        exploration_phase = int(episodes * 0.4)  # 40% exploration
        mixed_phase = int(episodes * 0.8)        # 80% mixed strategy
        
        for episode in tqdm(range(episodes), desc="Training DQN"):
            self.board.reset()
            current_player = 1
            episode_states = []
            game_length = 0
            max_game_length = 30  # Prevent very long games
            
            while not self.board.is_terminal() and game_length < max_game_length:
                state = self.board_to_state()
                valid_moves = self.board.get_valid_moves()
                
                # Dynamic strategy based on training phase
                if episode < exploration_phase:
                    # Heavy exploration phase - mostly random
                    if random.random() < 0.8:
                        move = random.choice(valid_moves)
                    else:
                        old_player = self.player
                        self.player = current_player
                        move = self.get_fast_move()  # Use fast move instead of full minimax
                        self.player = old_player
                elif episode < mixed_phase:
                    # Mixed phase - balanced exploration/exploitation
                    if random.random() < self.dqn_agent.epsilon:
                        move = random.choice(valid_moves)
                    else:
                        old_player = self.player
                        self.player = current_player
                        move = self.get_fast_move()
                        self.player = old_player
                else:
                    # Exploitation phase - use trained model more
                    if random.random() < max(0.1, self.dqn_agent.epsilon):
                        move = random.choice(valid_moves)
                    else:
                        old_player = self.player
                        self.player = current_player
                        move = self.get_best_move(training_mode=True)
                        self.player = old_player
                
                episode_states.append((current_player, state.copy()))
                self.board.set_move(move, current_player)
                current_player = 3 - current_player
                game_length += 1
            
            # Store experiences with reward shaping
            final_result_p1 = self.board.get_game_result(1)
            final_result_p2 = self.board.get_game_result(2)
            
            for i, (player, state) in enumerate(episode_states):
                if player == 1:
                    reward = 1.0 if final_result_p1 == 1 else (-1.0 if final_result_p1 == -1 else 0.0)
                else:
                    reward = 1.0 if final_result_p2 == 1 else (-1.0 if final_result_p2 == -1 else 0.0)
                
                # Add small step penalty to encourage shorter games
                reward -= 0.01 * (game_length / max_game_length)
                
                next_state = episode_states[i + 1][1] if i < len(episode_states) - 1 else state
                done = i == len(episode_states) - 1
                
                self.dqn_agent.remember(state, reward, next_state, done)
            
            # Train more frequently
            if len(self.dqn_agent.memory) > 32:
                self.dqn_agent.replay(batch_size=32)
            
            # Update target network more frequently
            if episode % 50 == 0:
                self.dqn_agent.update_target_network()
            
            # Save progress
            if episode % save_interval == 0 and episode > 0:
                self.dqn_agent.save_model(self.model_path)
                print(f"Episode {episode}: Epsilon={self.dqn_agent.epsilon:.4f}, "
                      f"Memory={len(self.dqn_agent.memory)}, Avg game length: {game_length}")
        
        self.is_training = False
        self.dqn_agent.save_model(self.model_path)
        print("Training completed!")

    def save_dqn_model(self, filepath=None):
        """Save DQN model"""
        if self.use_dqn and self.dqn_agent is not None:
            if filepath is None:
                filepath = self.model_path
            self.dqn_agent.save_model(filepath)

    def load_dqn_model(self, filepath=None):
        """Load DQN model"""
        if self.use_dqn and self.dqn_agent is not None:
            if filepath is None:
                filepath = self.model_path
            return self.dqn_agent.load_model(filepath)
        return False

    def toggle_dqn(self, use_dqn=True):
        """Toggle between DQN and heuristic evaluation"""
        self.use_dqn = use_dqn
        if use_dqn and self.dqn_agent is None:
            self.dqn_agent = DQNAgent()
            if os.path.exists(self.model_path):
                self.dqn_agent.load_model(self.model_path)
        print(f"DQN evaluation: {'Enabled' if use_dqn else 'Disabled'}")