from typing import List, Optional, Tuple

import numpy as np
import time

from src.board import GameBoard


class MinimaxBot:
    """Minimax bot with alpha-beta pruning and heuristic evaluation"""

    def __init__(self, player: int, max_depth: int):
        self.player = player
        self.opponent = 3 - player
        self.max_depth = max_depth
        self.board = GameBoard()

    def evaluate_position(self, player: int) -> float:
        """
        Heuristic evaluation function for board position.

        The evaluation considers:
        1. Immediate wins/losses (highest priority)
        2. Threats and opportunities for 4-in-a-row
        3. Control of center positions
        4. Avoiding dangerous 3-in-a-row setups
        5. Creating multiple threats

        Returns: evaluation score from perspective of 'player'
        Positive = good for player, Negative = bad for player
        """
        # Check terminal states first
        result = self.board.get_game_result(player)
        if result is not None:
            if result == 1:
                return 10000  # Win
            elif result == -1:
                return -10000  # Loss
            else:
                return 0  # Draw

        score = 0
         # 1. Evaluate win/lose patterns
        score += self._evaluate_patterns(player)

        # # 2. Center control bonus
        # score += self._evaluate_center_control(player)

        # # 3. Position connectivity bonus
        # score += self._evaluate_connectivity(player)

        # # Add small random noise to break ties and add variety
        # noise = 1.0 + np.clip(
        #     np.random.normal(0, 0.25), -0.5, 0.5
        # )  # Small noise factor with mean 0 and stddev 0.01
        # score *= noise * 10  # Scale noise to avoid too small values

        return score

    def _evaluate_patterns(self, player: int) -> float:
        """Evaluate win and lose patterns"""
        score = 0

        # Evaluate 4-in-a-row potential (win patterns)
        for pattern in self.board.win_patterns:
            player_count = sum(
                1 for r, c in pattern if self.board.board[r][c] == player
            ) - 1
            opponent_count = sum(
                1 for r, c in pattern if self.board.board[r][c] == self.opponent
            ) - 1
             # Count empty spaces
            empty_count = sum(1 for r, c in pattern if self.board.board[r][c] == 0)

            if opponent_count == 0:  # No opponent pieces blocking
                if player_count == 3:
                    score += 100  # One move from winning
                elif player_count == 2:
                    score += 10  # Two moves from winning
                elif player_count == 1:
                    score += 1  # Three moves from winning

            if player_count == 0:  # No player pieces blocking opponent
                if opponent_count == 3:
                    score -= 100  # Opponent one move from winning
                elif opponent_count == 2:
                    score -= 10  # Opponent two moves from winning
                elif opponent_count == 1:
                    score -= 1  # Opponent three moves from winning

        # Evaluate 3-in-a-row danger (lose patterns)
        for pattern in self.board.lose_patterns:
            player_count = sum(
                1 for r, c in pattern if self.board.board[r][c] == player
            )
            opponent_count = sum(
                1 for r, c in pattern if self.board.board[r][c] == self.opponent
            )

            if player_count == 2 and opponent_count == 0:
                score -= 50  # Dangerous: close to losing
            elif player_count == 1 and opponent_count == 0:
                score -= 5  # Somewhat dangerous

            if opponent_count == 2 and player_count == 0:
                score += 50  # Good: opponent close to losing
            elif opponent_count == 1 and player_count == 0:
                score += 5  # Somewhat good

        return score

    def _evaluate_center_control(self, player: int) -> float:
        """Evaluate control of center positions"""
        center_positions = [
            (2, 2),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 1),
            (2, 3),
            (3, 1),
            (3, 2),
            (3, 3),
        ]
        score = 0
        for r, c in center_positions:
            if self.board.board[r][c] == player:
                score += 2
            elif self.board.board[r][c] == self.opponent:
                score -= 2
        return score

    def _evaluate_connectivity(self, player: int) -> float:
        """Evaluate how well connected player's pieces are"""
        score = 0
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        for i in range(5):
            for j in range(5):
                if self.board.board[i][j] == player:
                    # Count adjacent friendly pieces
                    adjacent_count = 0
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if (
                            0 <= ni < 5
                            and 0 <= nj < 5
                            and self.board.board[ni][nj] == player
                        ):
                            adjacent_count += 1
                    score += adjacent_count * 0.5

                elif self.board.board[i][j] == self.opponent:
                    # Penalize opponent connectivity
                    adjacent_count = 0
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if (
                            0 <= ni < 5
                            and 0 <= nj < 5
                            and self.board.board[ni][nj] == self.opponent
                        ):
                            adjacent_count += 1
                    score -= adjacent_count * 0.5

        return score

    def minimax(
        self, depth: int, alpha: float, beta: float, maximizing: bool
    ) -> Tuple[float, Optional[int]]:
        """
        Minimax algorithm with alpha-beta pruning

        Args:
            depth: Current search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: True if maximizing player's turn

        Returns:
            Tuple of (evaluation_score, best_move)
        """
        if depth == 0 or self.board.is_terminal():
            return self.evaluate_position(self.player), None

        current_player = self.player if maximizing else self.opponent
        opponent_player = self.opponent if maximizing else self.player
        moves = self.board.get_valid_moves()
        best_move = None

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
                    break

            return max_eval, best_move

        else:

            min_eval = float("inf")
            

            for move in moves:
                self.board.set_move(move, current_player)
                if depth == self.max_depth and self.board.check_win(current_player):
                    self.board.undo_move(move)
                    return None, move
                # Check if this move causes immediate loss (3 in a row)
                if self.board.check_lose(current_player):
                    self.board.undo_move(move)
                    continue  # Skip losing moves

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

    def get_best_move(self) -> int:
        """Get the best move using minimax algorithm"""
        _, best_move = self.minimax(self.max_depth, float("-inf"), float("inf"), True)

        # Fallback to first valid move if no move found
        if best_move is None:
            #raise ValueError("No valid moves available")
            valid_moves = self.board.get_valid_moves()
            
            # # Evaluate each move and pick the best one
            # best_score = float("-inf") if self.player == 1 else float("inf")
            best_move = valid_moves[0]
            
            # for move in valid_moves:
            #     self.board.set_move(move, self.player)
            #     score = self.evaluate_position(self.player)
            #     self.board.undo_move(move)
            #     if self.player == 1:
            #         if score > best_score:
            #             best_score = score
            #             best_move = move
            #     else:
            #         # For player 2, we want to minimize the score
            #         if score < best_score:
            #             best_score = score
            #             best_move = move
        time.sleep(1)  # Simulate thinking time
        return best_move
