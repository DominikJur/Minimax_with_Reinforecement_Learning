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

# Enhanced DQN Network with better architecture
class DQNNetwork(nn.Module):
    def __init__(self, input_size=75, hidden_size=512, output_size=1):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.LeakyReLU(0.1),
            
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.LeakyReLU(0.1),
            
            nn.Linear(hidden_size//4, output_size),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size=75, lr=0.0005, gamma=0.95, epsilon=1.0, 
                 epsilon_min=0.05, epsilon_decay=0.995, memory_size=100000):
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
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
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
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Failed to load model from {filepath}: {e}")
            print("This usually happens when the model architecture has changed.")
            print("Starting with a fresh model instead...")
            return False

class MinimaxBot:
    """Enhanced Minimax bot with improved DQN evaluation"""

    def __init__(self, player: int, max_depth: int, use_dqn=True, model_path="dqn_model.pth"):
        self.player = player
        self.opponent = 3 - player
        self.max_depth = max_depth
        self.training_depth = max(2, max_depth - 1)  # Less shallow than before
        self.board = GameBoard()
        self.use_dqn = use_dqn
        self.model_path = model_path
        self.is_training = False
        
        # Move ordering for better pruning
        self.center_moves = [33, 32, 34, 23, 43, 22, 24, 42, 44]
        
        # Initialize DQN if requested
        if self.use_dqn:
            self.dqn_agent = DQNAgent()
            if os.path.exists(model_path):
                success = self.dqn_agent.load_model(model_path)
                if success:
                    print(f"Loaded DQN model: {model_path}")
                else:
                    print(f"Could not load existing model, starting fresh training")
            else:
                print(f"No DQN model found at {model_path}, will create new one")
        else:
            self.dqn_agent = None

    def board_to_enhanced_state(self):
        """Convert board to enhanced feature vector for DQN"""
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
        
        # Pattern-based features
        # Win patterns analysis (10 features)
        win_threats_me = 0
        win_threats_opp = 0
        win_opportunities_me = 0
        win_opportunities_opp = 0
        
        for pattern in self.board.win_patterns:
            my_count = sum(1 for r, c in pattern if board[r][c] == self.player)
            opp_count = sum(1 for r, c in pattern if board[r][c] == self.opponent)
            empty_count = sum(1 for r, c in pattern if board[r][c] == 0)
            
            if opp_count == 0:
                if my_count == 3:
                    win_threats_me += 1
                elif my_count == 2 and empty_count == 2:
                    win_opportunities_me += 1
            
            if my_count == 0:
                if opp_count == 3:
                    win_threats_opp += 1
                elif opp_count == 2 and empty_count == 2:
                    win_opportunities_opp += 1
        
        features.extend([
            min(win_threats_me / 10.0, 1.0),
            min(win_threats_opp / 10.0, 1.0),
            min(win_opportunities_me / 20.0, 1.0),
            min(win_opportunities_opp / 20.0, 1.0)
        ])
        
        # Lose patterns analysis (10 features)
        lose_dangers_me = 0
        lose_dangers_opp = 0
        
        for pattern in self.board.lose_patterns:
            my_count = sum(1 for r, c in pattern if board[r][c] == self.player)
            opp_count = sum(1 for r, c in pattern if board[r][c] == self.opponent)
            empty_count = sum(1 for r, c in pattern if board[r][c] == 0)
            
            if opp_count == 0 and my_count == 2 and empty_count == 1:
                lose_dangers_me += 1
            if my_count == 0 and opp_count == 2 and empty_count == 1:
                lose_dangers_opp += 1
        
        features.extend([
            min(lose_dangers_me / 15.0, 1.0),
            min(lose_dangers_opp / 15.0, 1.0)
        ])
        
        # Positional features (10 features)
        center_control_me = sum(1 for move in self.center_moves 
                               if board[(move//10)-1][(move%10)-1] == self.player)
        center_control_opp = sum(1 for move in self.center_moves 
                                if board[(move//10)-1][(move%10)-1] == self.opponent)
        
        # Corner control
        corners = [(0,0), (0,4), (4,0), (4,4)]
        corner_control_me = sum(1 for r, c in corners if board[r][c] == self.player)
        corner_control_opp = sum(1 for r, c in corners if board[r][c] == self.opponent)
        
        # Edge control
        edges = [(0,1), (0,2), (0,3), (1,0), (1,4), (2,0), (2,4), 
                 (3,0), (3,4), (4,1), (4,2), (4,3)]
        edge_control_me = sum(1 for r, c in edges if board[r][c] == self.player)
        edge_control_opp = sum(1 for r, c in edges if board[r][c] == self.opponent)
        
        features.extend([
            center_control_me / 9.0,
            center_control_opp / 9.0,
            corner_control_me / 4.0,
            corner_control_opp / 4.0,
            edge_control_me / 12.0,
            edge_control_opp / 12.0
        ])
        
        # Game phase (5 features)
        total_pieces = sum(1 for row in board for cell in row if cell != 0)
        game_progress = total_pieces / 25.0
        
        my_pieces = sum(1 for row in board for cell in row if cell == self.player)
        opp_pieces = sum(1 for row in board for cell in row if cell == self.opponent)
        
        features.extend([
            game_progress,
            my_pieces / 25.0,
            opp_pieces / 25.0,
            (my_pieces - opp_pieces) / 25.0,
            len(self.board.get_valid_moves()) / 25.0
        ])
        
        # Connectivity features (15 features)
        # Adjacent pairs, triplets analysis
        adjacent_pairs_me = 0
        adjacent_pairs_opp = 0
        
        # Check horizontal, vertical, diagonal adjacencies
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for r in range(5):
            for c in range(5):
                if board[r][c] != 0:
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 5 and 0 <= nc < 5:
                            if board[r][c] == board[nr][nc]:
                                if board[r][c] == self.player:
                                    adjacent_pairs_me += 1
                                else:
                                    adjacent_pairs_opp += 1
        
        features.extend([
            min(adjacent_pairs_me / 20.0, 1.0),
            min(adjacent_pairs_opp / 20.0, 1.0)
        ])
        
        # Pad to exactly 75 features
        while len(features) < 75:
            features.append(0.0)
        
        return np.array(features[:75], dtype=np.float32)

    def order_moves(self, moves: List[int]) -> List[int]:
        """Order moves for better alpha-beta pruning"""
        center_first = []
        others = []
        
        for move in moves:
            if move in self.center_moves:
                center_first.append(move)
            else:
                others.append(move)
        
        return center_first + others

    def evaluate_position(self, player: int) -> float:
        """Combined DQN + heuristic evaluation"""
        # Check terminal states first
        result = self.board.get_game_result(player)
        if result is not None:
            if result == 1:
                return 10000  # Win
            elif result == -1:
                return -10000  # Loss
            else:
                return 0  # Draw

        dqn_score = 0.0
        heuristic_score = self._evaluate_patterns(player)
        
        # Get DQN evaluation if available
        if self.use_dqn and self.dqn_agent is not None:
            try:
                state = self.board_to_enhanced_state()
                
                # Adjust perspective for the evaluating player
                if player != self.player:
                    # Flip the state perspective
                    state_copy = state.copy()
                    # Flip basic board state (first 25 features)
                    for i in range(25):
                        if state_copy[i] != 0:
                            state_copy[i] = -state_copy[i]
                    # Flip other relevant features
                    for i in range(25, len(state_copy)):
                        if i in [25, 27, 29, 31, 33, 35]:  # Indices of "opponent" features
                            state_copy[i], state_copy[i+1] = state_copy[i+1], state_copy[i]
                    state = state_copy
                
                dqn_value = self.dqn_agent.get_value(state)
                dqn_score = dqn_value * 3000  # Scale appropriately
                
            except Exception as e:
                print(f"DQN evaluation failed: {e}")
                dqn_score = 0.0
        
        # Combine scores with appropriate weighting
        if self.is_training:
            # During training, rely more on heuristics early on
            dqn_weight = 0.3
            heuristic_weight = 0.7
        else:
            # During actual play, trust the trained DQN more
            dqn_weight = 0.7
            heuristic_weight = 0.3
        
        combined_score = (dqn_weight * dqn_score) + (heuristic_weight * heuristic_score)
        return combined_score

    def _evaluate_patterns(self, player: int) -> float:
        """Enhanced heuristic evaluation"""
        score = 0
        board = self.board.board
        opponent = 3 - player

        # Win patterns evaluation with better weights
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
                    score += 500  # Stronger weight for near-wins
                elif player_count == 2:
                    score += 50
                elif player_count == 1:
                    score += 5
            elif player_count == 0:
                if opponent_count == 3:
                    score -= 500
                elif opponent_count == 2:
                    score -= 50
                elif opponent_count == 1:
                    score -= 5

        # Lose patterns evaluation with stronger penalties
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
                    score -= 200  # Heavy penalty for potential loses
                elif player_count == 1:
                    score -= 10
            elif player_count == 0:
                if opponent_count == 2:
                    score += 200
                elif opponent_count == 1:
                    score += 10

        return score

    def calculate_intermediate_reward(self, player: int, move: int) -> float:
        """Calculate intermediate rewards for better training"""
        old_eval = self.evaluate_position(player)
        
        # Make the move temporarily
        self.board.set_move(move, player)
        new_eval = self.evaluate_position(player)
        self.board.undo_move(move)
        
        # Reward improvement
        improvement = (new_eval - old_eval) / 1000.0
        
        # Additional specific rewards
        reward = improvement
        
        # Check if move creates winning threat
        self.board.set_move(move, player)
        if any(sum(1 for r, c in pattern if self.board.board[r][c] == player) == 3 
               for pattern in self.board.win_patterns):
            reward += 0.5
        
        # Check if move blocks opponent win
        self.board.undo_move(move)
        self.board.set_move(move, 3-player)
        if any(sum(1 for r, c in pattern if self.board.board[r][c] == (3-player)) == 4 
               for pattern in self.board.win_patterns):
            reward += 0.8
        
        # Check if move avoids creating lose pattern
        if any(sum(1 for r, c in pattern if self.board.board[r][c] == (3-player)) == 3 
               for pattern in self.board.lose_patterns):
            reward -= 0.5
        
        self.board.undo_move(move)
        
        return np.clip(reward, -1.0, 1.0)

    def minimax(self, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[float, Optional[int]]:
        """Optimized minimax algorithm"""
        if depth == 0 or self.board.is_terminal():
            return self.evaluate_position(self.player), None

        current_player = self.player if maximizing else self.opponent
        moves = self.order_moves(self.board.get_valid_moves())
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

                if depth == self.max_depth and self.board.check_win(current_player):
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
                
                if depth == self.max_depth and self.board.check_win(current_player):
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
        """Get the best move using minimax algorithm"""
        depth = self.training_depth if training_mode else self.max_depth
        
        _, best_move = self.minimax(depth, float("-inf"), float("inf"), True)

        if best_move is None:
            valid_moves = self.board.get_valid_moves()
            best_move = valid_moves[0]
        
        if not training_mode and not self.is_training:
            time.sleep(0.1)
        
        return best_move

    def get_heuristic_move(self) -> int:
        """Get move using pure heuristic evaluation"""
        old_dqn = self.use_dqn
        self.use_dqn = False
        move = self.get_best_move(training_mode=True)
        self.use_dqn = old_dqn
        return move

    # Enhanced training with curriculum learning
    def train_dqn(self, episodes=5000, save_interval=500):
        """Enhanced DQN training with curriculum learning"""
        if not self.use_dqn or self.dqn_agent is None:
            print("DQN not enabled!")
            return
        
        self.is_training = True
        print(f"Training DQN for {episodes} episodes...")
        print(f"Training on device: {self.dqn_agent.device}")
        
        # Curriculum phases
        random_phase = int(episodes * 0.2)      # 20% random
        heuristic_phase = int(episodes * 0.6)   # 60% vs heuristic  
        self_play_phase = episodes              # 20% self-play
        
        win_rates = []
        
        for episode in tqdm(range(episodes), desc="Training DQN"):
            self.board.reset()
            current_player = 1
            episode_data = []
            game_length = 0
            max_game_length = 40
            
            while not self.board.is_terminal() and game_length < max_game_length:
                state = self.board_to_enhanced_state()
                valid_moves = self.board.get_valid_moves()
                
                # Curriculum learning strategy
                if episode < random_phase:
                    # Early phase: Some random, some heuristic
                    if random.random() < 0.4:
                        move = random.choice(valid_moves)
                    else:
                        old_player = self.player
                        self.player = current_player
                        move = self.get_heuristic_move()
                        self.player = old_player
                
                elif episode < heuristic_phase:
                    # Main training phase: mostly heuristic opponents
                    if random.random() < max(0.3, self.dqn_agent.epsilon):
                        move = random.choice(valid_moves)
                    else:
                        old_player = self.player
                        self.player = current_player
                        move = self.get_heuristic_move()
                        self.player = old_player
                
                else:
                    # Advanced phase: self-play and exploration
                    if random.random() < max(0.1, self.dqn_agent.epsilon):
                        move = random.choice(valid_moves)
                    else:
                        old_player = self.player
                        self.player = current_player
                        move = self.get_best_move(training_mode=True)
                        self.player = old_player
                
                # Calculate intermediate reward
                intermediate_reward = self.calculate_intermediate_reward(current_player, move)
                
                episode_data.append((current_player, state.copy(), move, intermediate_reward))
                self.board.set_move(move, current_player)
                current_player = 3 - current_player
                game_length += 1
            
            # Process episode with improved rewards
            final_result_p1 = self.board.get_game_result(1)
            final_result_p2 = self.board.get_game_result(2)
            
            for i, (player, state, move, intermediate_reward) in enumerate(episode_data):
                # Final game reward
                if player == 1:
                    final_reward = 1.0 if final_result_p1 == 1 else (-1.0 if final_result_p1 == -1 else 0.0)
                else:
                    final_reward = 1.0 if final_result_p2 == 1 else (-1.0 if final_result_p2 == -1 else 0.0)
                
                # Combine intermediate and final rewards
                combined_reward = 0.3 * intermediate_reward + 0.7 * final_reward
                
                # Add small penalty for long games
                combined_reward -= 0.01 * (game_length / max_game_length)
                
                next_state = episode_data[i + 1][1] if i < len(episode_data) - 1 else state
                done = i == len(episode_data) - 1
                
                self.dqn_agent.remember(state, combined_reward, next_state, done)
            
            # Train more frequently with larger batches
            if len(self.dqn_agent.memory) > 128:
                loss = self.dqn_agent.replay(batch_size=64)
            
            # Update target network
            if episode % 100 == 0:
                self.dqn_agent.update_target_network()
            
            # Track performance and save
            if episode % save_interval == 0 and episode > 0:
                self.dqn_agent.save_model(self.model_path)
                
                # Quick performance test
                if episode % (save_interval * 2) == 0:
                    win_rate = self._quick_performance_test()
                    win_rates.append(win_rate)
                    print(f"Episode {episode}: Epsilon={self.dqn_agent.epsilon:.4f}, "
                          f"Win rate vs heuristic: {win_rate:.3f}, Memory={len(self.dqn_agent.memory)}")
        
        self.is_training = False
        self.dqn_agent.save_model(self.model_path)
        print("Training completed!")
        
        if win_rates:
            print(f"Final win rate progression: {win_rates}")
    
    def _quick_performance_test(self, games=10):
        """Quick performance test during training"""
        wins = 0
        
        for _ in range(games):
            test_board = GameBoard()
            old_board = self.board
            self.board = test_board
            
            current_player = 1
            game_length = 0
            
            while not test_board.is_terminal() and game_length < 30:
                if current_player == self.player:
                    move = self.get_best_move(training_mode=True)
                else:
                    move = self.get_heuristic_move()
                
                test_board.set_move(move, current_player)
                current_player = 3 - current_player
                game_length += 1
            
            result = test_board.get_game_result(self.player)
            if result == 1:
                wins += 1
            
            self.board = old_board
        
        return wins / games

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