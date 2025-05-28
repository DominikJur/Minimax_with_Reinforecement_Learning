"""
Training and usage script for DQN-enhanced Minimax Bot
"""

import sys
import os
sys.path.append('.')  # Add current directory to path

from src.board import GameBoard
from src.minimax import MinimaxBot  # Your updated bot
import random
import numpy as np

def train_new_bot():
    """Train a new DQN bot from scratch"""
    print("=== Training New DQN Bot ===")
    
    # Create bot with DQN enabled
    bot = MinimaxBot(
        player=1, 
        max_depth=3,  # Reduced depth since DQN will handle evaluation
        use_dqn=True, 
        model_path="my_trained_bot.pth"
    )
    
    print("Starting training...")
    bot.train_dqn(episodes=300, save_interval=50)
    
    print("Training completed! Testing bot...")
    test_bot_performance(bot)

def load_and_use_trained_bot():
    """Load existing trained bot and use it"""
    print("=== Loading Trained Bot ===")
    
    model_path = "my_trained_bot.pth"
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        print("Run train_new_bot() first!")
        return None
    
    # Create bot and load trained model
    bot = MinimaxBot(
        player=1, 
        max_depth=3, 
        use_dqn=True, 
        model_path=model_path
    )
    
    print("Bot loaded successfully!")
    return bot

def test_bot_performance(bot, games=100):
    """Test bot performance against random player"""
    print(f"\n=== Testing Bot Performance ({games} games) ===")
    
    wins = 0
    losses = 0
    draws = 0
    
    for game in range(games):
        board = GameBoard()
        bot.board = board  # Set board reference
        current_player = 1
        game_length = 0
        max_moves = 50
        
        while not board.is_terminal() and game_length < max_moves:
            valid_moves = board.get_valid_moves()
            
            if current_player == bot.player:
                # Use trained bot
                move = bot.get_best_move()
            else:
                # Random opponent
                move = random.choice(valid_moves)
            
            board.set_move(move, current_player)
            current_player = 3 - current_player
            game_length += 1
        
        # Check game result
        result = board.get_game_result(bot.player)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
        
        if (game + 1) % 20 == 0:
            current_win_rate = wins / (game + 1)
            print(f"Progress: {game + 1}/{games} games, Win rate: {current_win_rate:.3f}")
    
    win_rate = wins / games
    print(f"\nFinal Results: {wins}W-{losses}L-{draws}D")
    print(f"Win Rate: {win_rate:.3f}")
    return win_rate

def play_human_vs_bot():
    """Play a game against the trained bot"""
    print("=== Human vs Bot ===")
    
    bot = load_and_use_trained_bot()
    if bot is None:
        return
    
    board = GameBoard()
    bot.board = board
    
    human_player = int(input("Choose your player (1 or 2): "))
    bot.player = 3 - human_player
    bot.opponent = human_player
    
    current_player = 1
    
    print("\nBoard positions (11-55):")
    print("11 12 13 14 15")
    print("21 22 23 24 25")
    print("31 32 33 34 35")
    print("41 42 43 44 45")
    print("51 52 53 54 55")
    print()
    
    while not board.is_terminal():
        # Display board
        print("Current board:")
        for i, row in enumerate(board.board):
            row_str = ""
            for j, cell in enumerate(row):
                if cell == 0:
                    row_str += ". "
                elif cell == 1:
                    row_str += "X "
                else:
                    row_str += "O "
            print(row_str)
        print()
        
        valid_moves = board.get_valid_moves()
        
        if current_player == human_player:
            # Human turn
            print(f"Your turn (Player {human_player})")
            print(f"Valid moves: {valid_moves}")
            
            while True:
                try:
                    move = int(input("Enter your move: "))
                    if move in valid_moves:
                        break
                    else:
                        print("Invalid move! Try again.")
                except ValueError:
                    print("Please enter a number!")
        else:
            # Bot turn
            print(f"Bot thinking (Player {bot.player})...")
            move = bot.get_best_move()
            print(f"Bot chooses: {move}")
        
        board.set_move(move, current_player)
        current_player = 3 - current_player
    
    # Show final result
    print("\nFinal board:")
    for row in board.board:
        row_str = ""
        for cell in row:
            if cell == 0:
                row_str += ". "
            elif cell == 1:
                row_str += "X "
            else:
                row_str += "O "
        print(row_str)
    
    result = board.get_game_result(human_player)
    if result == 1:
        print("You won! 🎉")
    elif result == -1:
        print("Bot won! 🤖")
    else:
        print("It's a draw! 🤝")

def compare_dqn_vs_heuristic():
    """Compare DQN evaluation vs original heuristic"""
    print("=== DQN vs Heuristic Comparison ===")
    
    if not os.path.exists("my_trained_bot.pth"):
        print("No trained model found! Train a bot first.")
        return
    
    # Create both bots
    dqn_bot = MinimaxBot(player=1, max_depth=3, use_dqn=True, model_path="my_trained_bot.pth")
    heuristic_bot = MinimaxBot(player=2, max_depth=3, use_dqn=False)
    
    wins_dqn = 0
    wins_heuristic = 0
    draws = 0
    games = 50
    
    print(f"Playing {games} games: DQN vs Heuristic")
    
    for game in range(games):
        board = GameBoard()
        dqn_bot.board = board
        heuristic_bot.board = board
        
        current_player = 1
        game_length = 0
        
        while not board.is_terminal() and game_length < 50:
            if current_player == 1:
                move = dqn_bot.get_best_move()
            else:
                move = heuristic_bot.get_best_move()
            
            board.set_move(move, current_player)
            current_player = 3 - current_player
            game_length += 1
        
        result = board.get_game_result(1)  # From DQN bot's perspective
        if result == 1:
            wins_dqn += 1
        elif result == -1:
            wins_heuristic += 1
        else:
            draws += 1
        
        if (game + 1) % 10 == 0:
            print(f"Games played: {game + 1}/{games}")
    
    print(f"\nResults:")
    print(f"DQN Bot: {wins_dqn} wins")
    print(f"Heuristic Bot: {wins_heuristic} wins")
    print(f"Draws: {draws}")
    print(f"DQN Win Rate: {wins_dqn/games:.3f}")

def main():
    """Main menu"""
    while True:
        print("\n" + "="*50)
        print("DQN Minimax Bot Training & Testing")
        print("="*50)
        print("1. Train new DQN bot")
        print("2. Test trained bot vs random")
        print("3. Play against bot")
        print("4. Compare DQN vs Heuristic")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            train_new_bot()
        elif choice == '2':
            bot = load_and_use_trained_bot()
            if bot:
                test_bot_performance(bot, games=100)
        elif choice == '3':
            play_human_vs_bot()
        elif choice == '4':
            compare_dqn_vs_heuristic()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()