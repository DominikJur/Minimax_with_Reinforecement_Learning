from typing import List, Optional, Tuple


class GameBoard:
    """Represents the 5x5 game board and game logic"""

    def __init__(self):
        self.board = [[0 for _ in range(5)] for _ in range(5)]

        # Win patterns: 4 in a row (copied from board.h)
        self.win_patterns = [
            # Horizontal patterns
            [(0, 0), (0, 1), (0, 2), (0, 3)],
            [(1, 0), (1, 1), (1, 2), (1, 3)],
            [(2, 0), (2, 1), (2, 2), (2, 3)],
            [(3, 0), (3, 1), (3, 2), (3, 3)],
            [(4, 0), (4, 1), (4, 2), (4, 3)],
            [(0, 1), (0, 2), (0, 3), (0, 4)],
            [(1, 1), (1, 2), (1, 3), (1, 4)],
            [(2, 1), (2, 2), (2, 3), (2, 4)],
            [(3, 1), (3, 2), (3, 3), (3, 4)],
            [(4, 1), (4, 2), (4, 3), (4, 4)],
            # Vertical patterns
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 1), (1, 1), (2, 1), (3, 1)],
            [(0, 2), (1, 2), (2, 2), (3, 2)],
            [(0, 3), (1, 3), (2, 3), (3, 3)],
            [(0, 4), (1, 4), (2, 4), (3, 4)],
            [(1, 0), (2, 0), (3, 0), (4, 0)],
            [(1, 1), (2, 1), (3, 1), (4, 1)],
            [(1, 2), (2, 2), (3, 2), (4, 2)],
            [(1, 3), (2, 3), (3, 3), (4, 3)],
            [(1, 4), (2, 4), (3, 4), (4, 4)],
            # Diagonal patterns
            [(0, 1), (1, 2), (2, 3), (3, 4)],
            [(0, 0), (1, 1), (2, 2), (3, 3)],
            [(1, 1), (2, 2), (3, 3), (4, 4)],
            [(1, 0), (2, 1), (3, 2), (4, 3)],
            [(0, 3), (1, 2), (2, 1), (3, 0)],
            [(0, 4), (1, 3), (2, 2), (3, 1)],
            [(1, 3), (2, 2), (3, 1), (4, 0)],
            [(1, 4), (2, 3), (3, 2), (4, 1)],
        ]

        # Lose patterns: exactly 3 in a row (copied from board.h)
        self.lose_patterns = [
            # Horizontal
            [(0, 0), (0, 1), (0, 2)],
            [(0, 1), (0, 2), (0, 3)],
            [(0, 2), (0, 3), (0, 4)],
            [(1, 0), (1, 1), (1, 2)],
            [(1, 1), (1, 2), (1, 3)],
            [(1, 2), (1, 3), (1, 4)],
            [(2, 0), (2, 1), (2, 2)],
            [(2, 1), (2, 2), (2, 3)],
            [(2, 2), (2, 3), (2, 4)],
            [(3, 0), (3, 1), (3, 2)],
            [(3, 1), (3, 2), (3, 3)],
            [(3, 2), (3, 3), (3, 4)],
            [(4, 0), (4, 1), (4, 2)],
            [(4, 1), (4, 2), (4, 3)],
            [(4, 2), (4, 3), (4, 4)],
            # Vertical
            [(0, 0), (1, 0), (2, 0)],
            [(1, 0), (2, 0), (3, 0)],
            [(2, 0), (3, 0), (4, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(1, 1), (2, 1), (3, 1)],
            [(2, 1), (3, 1), (4, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(1, 2), (2, 2), (3, 2)],
            [(2, 2), (3, 2), (4, 2)],
            [(0, 3), (1, 3), (2, 3)],
            [(1, 3), (2, 3), (3, 3)],
            [(2, 3), (3, 3), (4, 3)],
            [(0, 4), (1, 4), (2, 4)],
            [(1, 4), (2, 4), (3, 4)],
            [(2, 4), (3, 4), (4, 4)],
            # Diagonal
            [(0, 2), (1, 3), (2, 4)],
            [(0, 1), (1, 2), (2, 3)],
            [(1, 2), (2, 3), (3, 4)],
            [(0, 0), (1, 1), (2, 2)],
            [(1, 1), (2, 2), (3, 3)],
            [(2, 2), (3, 3), (4, 4)],
            [(1, 0), (2, 1), (3, 2)],
            [(2, 1), (3, 2), (4, 3)],
            [(2, 0), (3, 1), (4, 2)],
            [(0, 2), (1, 1), (2, 0)],
            [(0, 3), (1, 2), (2, 1)],
            [(1, 2), (2, 1), (3, 0)],
            [(0, 4), (1, 3), (2, 2)],
            [(1, 3), (2, 2), (3, 1)],
            [(2, 2), (3, 1), (4, 0)],
            [(1, 4), (2, 3), (3, 2)],
            [(2, 3), (3, 2), (4, 1)],
            [(2, 4), (3, 3), (4, 2)],
        ]

    def reset(self):
        """Reset the board to empty state"""
        self.board = [[0 for _ in range(5)] for _ in range(5)]

    def set_move(self, move: int, player: int) -> bool:
        """Set a move on the board. Move format: row*10 + col (11-55)"""
        if move < 11 or move > 55:
            return False

        row = (move // 10) - 1
        col = (move % 10) - 1

        if row < 0 or row > 4 or col < 0 or col > 4:
            return False
        if self.board[row][col] != 0:
            return False

        self.board[row][col] = player
        return True

    def undo_move(self, move: int):
        """Undo a move"""
        row = (move // 10) - 1
        col = (move % 10) - 1
        self.board[row][col] = 0

    def is_valid_move(self, move: int) -> bool:
        """Check if move is valid"""
        if move < 11 or move > 55:
            return False
        row = (move // 10) - 1
        col = (move % 10) - 1
        if row < 0 or row > 4 or col < 0 or col > 4:
            return False
        return self.board[row][col] == 0

    def get_valid_moves(self) -> List[int]:
        """Get all valid moves"""
        moves = []
        for i in range(5):
            for j in range(5):
                if self.board[i][j] == 0:
                    moves.append((i + 1) * 10 + (j + 1))
        return moves

    def check_win(self, player: int) -> bool:
        """Check if player has won (4 in a row)"""
        for pattern in self.win_patterns:
            if all(self.board[r][c] == player for r, c in pattern):
                return True
        return False

    def check_lose(self, player: int) -> bool:
        """Check if player has lost (exactly 3 in a row)"""
        for pattern in self.lose_patterns:
            if all(self.board[r][c] == player for r, c in pattern):
                return True
        return False

    def is_terminal(self) -> bool:
        """Check if game is in terminal state"""
        # Check if anyone won or lost
        for player in [1, 2]:
            if self.check_win(player) or self.check_lose(player):
                return True
        # Check if board is full
        return len(self.get_valid_moves()) == 0

    def get_game_result(self, player: int) -> Optional[int]:
        """Get game result from perspective of player. 1=win, -1=loss, 0=draw, None=ongoing"""
        if self.check_win(player):
            return 1
        if self.check_lose(player):
            return -1
        if self.check_win(3 - player):
            return -1
        if self.check_lose(3 - player):
            return 1
        if len(self.get_valid_moves()) == 0:
            return 0
        return None

    def is_empty(self) -> bool:
        """Check if the board is empty"""
        return all(cell == 0 for row in self.board for cell in row)
