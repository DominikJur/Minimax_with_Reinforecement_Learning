from typing import List, Optional, Tuple
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
from concurrent.futures import ThreadPoolExecutor

from src.board import GameBoard
from tqdm import tqdm

from src.dqn_agent import DQNAgent

class MinimaxBot:
    """Speed optimized with move ordering, caching, and parallel execution"""

    def __init__(self, player: int, max_depth: int, use_dqn=True, model_path="dqn_model.pth"):
        self.player = player
        self.opponent = 3 - player
        self.max_depth = max_depth
        self.board = GameBoard()
        self.use_dqn = use_dqn
        self.model_path = model_path
        self.is_training = False
        
        # Transposition table for caching position evaluations
        self.transposition_table = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize DQN
        if self.use_dqn:
            self.dqn_agent = DQNAgent()
            if os.path.exists(model_path):
                success = self.dqn_agent.load_model(model_path)
                if not success:
                    print("Starting with fresh model")
        else:
            self.dqn_agent = None

    def get_position_hash(self):
        """Create a hash of the current board position for caching"""
        board_tuple = tuple(tuple(row) for row in self.board.board)
        return hash(board_tuple)

    def board_to_features(self):
        """Convert board to feature vector - UNCHANGED"""
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
        """Cached evaluation function"""
        # Check cache first
        pos_hash = self.get_position_hash()
        cache_key = (pos_hash, player)
        
        if cache_key in self.transposition_table:
            self.cache_hits += 1
            return self.transposition_table[cache_key]
        
        self.cache_misses += 1
        
        # Terminal states
        result = self.board.get_game_result(player)
        if result is not None:
            eval_score = 10000 if result == 1 else (-10000 if result == -1 else 0)
            self.transposition_table[cache_key] = eval_score
            return eval_score

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
                dqn_score = 0.0
        
        # Combine scores
        if self.is_training:
            final_score = 0.3 * dqn_score + 0.7 * heuristic_score
        else:
            final_score = 0.6 * dqn_score + 0.4 * heuristic_score
        
        # Cache the result
        self.transposition_table[cache_key] = final_score
        return final_score

    def _evaluate_heuristic(self, player: int) -> float:
        """UNCHANGED heuristic"""
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

        # Center control
        center_moves = [33, 32, 34, 23, 43, 22, 24, 42, 44]
        my_center = sum(1 for move in center_moves 
                       if board[(move//10)-1][(move%10)-1] == player)
        opp_center = sum(1 for move in center_moves 
                        if board[(move//10)-1][(move%10)-1] == opponent)
        score += (my_center - opp_center) * 10
        
        return score

    def order_moves(self, moves: List[int], current_player: int) -> List[int]:
        """Smart move ordering for better alpha-beta pruning"""
        if len(moves) <= 1:
            return moves
        
        move_scores = []
        
        for move in moves:
            score = 0
            
            # Test the move to see immediate effects
            self.board.set_move(move, current_player)
            
            # HIGHEST priority: Winning moves
            if self.board.check_win(current_player):
                score += 15000
                self.board.undo_move(move)
                move_scores.append((move, score))
                continue  # Skip other checks, this is the best possible move
            
            # HIGHEST penalty: Losing moves (creating 3-in-a-row)
            if self.board.check_lose(current_player):
                score -= 10000  # Heavily penalize losing moves
                self.board.undo_move(move)
                move_scores.append((move, score))
                continue  # Skip other checks, this is terrible
            
            # High priority: Block opponent wins (but ONLY if we don't lose)
            self.board.undo_move(move)
            self.board.set_move(move, 3 - current_player)
            if self.board.check_win(3 - current_player):
                score += 5000  # Good to block, and we know it doesn't make us lose
            
            # Reset for positional analysis
            self.board.undo_move(move)
            
            # Positional scoring
            row, col = (move // 10) - 1, (move % 10) - 1
            
            # Center control
            center_dist = abs(row - 2) + abs(col - 2)
            score += (4 - center_dist) * 10
            
            # Pattern building - count potential 4-in-a-rows
            potential_wins = 0
            for pattern in self.board.win_patterns:
                if (row, col) in pattern:
                    my_count = sum(1 for r, c in pattern if self.board.board[r][c] == current_player)
                    opp_count = sum(1 for r, c in pattern if self.board.board[r][c] == (3 - current_player))
                    
                    if opp_count == 0:
                        potential_wins += my_count
            
            score += potential_wins * 5
            
            # Check for opponent losing patterns we can exploit
            opponent_traps = 0
            for pattern in self.board.lose_patterns:
                if (row, col) in pattern:
                    opp_count = sum(1 for r, c in pattern if self.board.board[r][c] == (3 - current_player))
                    my_count = sum(1 for r, c in pattern if self.board.board[r][c] == current_player)
                    
                    # If opponent has 2 in this losing pattern and we have 0, we can force them to lose
                    if opp_count == 2 and my_count == 0:
                        opponent_traps += 1
            
            score += opponent_traps * 50  # Reward moves that force opponent into traps
            
            move_scores.append((move, score))
        
        # Sort by score (highest first)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in move_scores]

    def minimax(self, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[float, Optional[int]]:
        """Optimized minimax with move ordering and caching"""
        if depth == 0 or self.board.is_terminal():
            return self.evaluate_position(self.player), None

        current_player = self.player if maximizing else self.opponent
        moves = self.board.get_valid_moves()
        
        if not moves:
            return self.evaluate_position(self.player), None
        
        best_move = None

        if self.board.is_empty():
            return 5000, 33
        
        # OPTIMIZATION 1: Move ordering for better pruning
        ordered_moves = self.order_moves(moves, current_player)
        
        if maximizing:
            max_eval = float("-inf")
            for move in ordered_moves:
                self.board.set_move(move, current_player)
                
                if depth == self.max_depth and self.board.check_win(current_player):
                    self.board.undo_move(move)
                    return 150000.0, move

                if self.board.check_win(current_player):
                    self.board.undo_move(move)
                    return 15000.0, move
                
                if self.board.check_lose(current_player):
                    self.board.undo_move(move)
                    continue

                if depth == self.max_depth:
                    self.board.undo_move(move)
                    self.board.set_move(move, 3 - current_player)
                    if self.board.check_win(3 - current_player):
                        self.board.undo_move(move)
                        return 15000.0, move
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
            for move in ordered_moves:
                self.board.set_move(move, current_player)
                
                if self.board.check_win(current_player):
                    self.board.undo_move(move)
                    return 15000.0, move
                
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
                    break  # Alpha-beta pruning

            return min_eval, best_move

    def get_best_move(self, training_mode=False) -> int:
        """UNCHANGED move selection - keeps full depth for intelligence"""
        depth = max(2, self.max_depth - 1) if training_mode else self.max_depth
        _, best_move = self.minimax(depth, float("-inf"), float("inf"), True)

        if best_move is None:
            best_move = self.board.get_valid_moves()[0]
        
        if not training_mode and not self.is_training:
            time.sleep(0.1)
        
        return best_move

    def get_heuristic_move(self) -> int:
        """UNCHANGED"""
        old_dqn = self.use_dqn
        self.use_dqn = False
        move = self.get_best_move(training_mode=True)
        self.use_dqn = old_dqn
        return move

    def clear_cache(self):
        """Clear transposition table to prevent memory buildup"""
        self.transposition_table.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def run_single_game(self, episode_num, total_episodes):
        """Run a single training game - for parallel execution"""
        self.board.reset()
        current_player = 1
        episode_data = []
        
        # KEEP EXACT SAME training phases
        random_phase = int(total_episodes * 0.15)
        heuristic_phase = int(total_episodes * 0.75)
        
        while not self.board.is_terminal():
            state = self.board_to_features()
            
            # KEEP EXACT SAME opponent strategy logic
            if episode_num < random_phase:
                if random.random() < 0.5:
                    move = random.choice(self.board.get_valid_moves())
                else:
                    old_player = self.player
                    self.player = current_player
                    move = self.get_heuristic_move()
                    self.player = old_player
            
            elif episode_num < heuristic_phase:
                if random.random() < self.dqn_agent.epsilon:
                    move = random.choice(self.board.get_valid_moves())
                else:
                    old_player = self.player
                    self.player = current_player
                    move = self.get_heuristic_move()
                    self.player = old_player
            else:
                # Self-play - now optimized with caching and move ordering
                old_player = self.player
                self.player = current_player
                move = self.get_best_move(training_mode=True)
                self.player = old_player
            
            episode_data.append((current_player, state.copy()))
            self.board.set_move(move, current_player)
            current_player = 3 - current_player
        
        # Return game result and episode data
        final_result_p1 = self.board.get_game_result(1)
        final_result_p2 = self.board.get_game_result(2)
        
        return episode_data, final_result_p1, final_result_p2

    def train_dqn(self, episodes=3000, save_interval=300):
        """OPTIMIZATION 3: Parallel training for self-play episodes"""
        if not self.use_dqn:
            print("DQN not enabled!")
            return
        
        self.is_training = True
        print(f"Training DQN for {episodes} episodes (with optimizations)...")
        
        # KEEP EXACT SAME training phases
        random_phase = int(episodes * 0.15)
        heuristic_phase = int(episodes * 0.75)
        selfplay_start = heuristic_phase
        
        print(f"Cache optimization: Active")
        print(f"Move ordering: Active") 
        print(f"Parallel self-play: Active for episodes {selfplay_start}-{episodes}")
        
        for episode in tqdm(range(episodes), desc="Training"):
            
            # Clear cache periodically to prevent memory buildup
            if episode % 500 == 0:
                self.clear_cache()
            
            # OPTIMIZATION 3: Use parallel execution for self-play episodes
            if episode >= selfplay_start and episode % 4 == 0 and episode < episodes - 4:
                # Run 4 self-play games in parallel every 4th episode during self-play
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = []
                    for i in range(4):
                        if episode + i < episodes:
                            future = executor.submit(self.run_single_game, episode + i, episodes)
                            futures.append(future)
                    
                    # Collect results and store experiences
                    for future in futures:
                        episode_data, final_result_p1, final_result_p2 = future.result()
                        self.store_episode_experiences(episode_data, final_result_p1, final_result_p2)
                
                # Skip the next 3 episodes since we processed them in parallel
                episode += 3
                continue
            
            # Regular sequential training for non-parallel episodes
            episode_data, final_result_p1, final_result_p2 = self.run_single_game(episode, episodes)
            self.store_episode_experiences(episode_data, final_result_p1, final_result_p2)
            
            # KEEP EXACT SAME training schedule
            if len(self.dqn_agent.memory) > 64:
                self.dqn_agent.replay(batch_size=32)
            
            if episode % 100 == 0:
                self.dqn_agent.update_target_network()
            
            if episode % save_interval == 0 and episode > 0:
                self.dqn_agent.save_model(self.model_path)
                cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
                print(f"Episode {episode}: ε={self.dqn_agent.epsilon:.3f}, Cache hit rate: {cache_hit_rate:.3f}")
        
        self.is_training = False
        self.dqn_agent.save_model(self.model_path)
        
        # Final stats
        total_cache_accesses = self.cache_hits + self.cache_misses
        if total_cache_accesses > 0:
            cache_hit_rate = self.cache_hits / total_cache_accesses
            print(f"Training completed! Final cache hit rate: {cache_hit_rate:.3f}")
            print(f"Cache saved {self.cache_hits} position evaluations")

    def store_episode_experiences(self, episode_data, final_result_p1, final_result_p2):
        """Store episode experiences in DQN memory"""
        for i, (player, state) in enumerate(episode_data):
            reward = 1.0 if (player == 1 and final_result_p1 == 1) or (player == 2 and final_result_p2 == 1) else \
                    -1.0 if (player == 1 and final_result_p1 == -1) or (player == 2 and final_result_p2 == -1) else 0.0
            
            next_state = episode_data[i + 1][1] if i < len(episode_data) - 1 else state
            done = i == len(episode_data) - 1
            
            self.dqn_agent.remember(state, reward, next_state, done)

    def save_model(self, filepath=None):
        if self.dqn_agent:
            self.dqn_agent.save_model(filepath or self.model_path)

    def load_model(self, filepath=None):
        if self.dqn_agent:
            return self.dqn_agent.load_model(filepath or self.model_path)