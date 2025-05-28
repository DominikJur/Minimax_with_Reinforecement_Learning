"""
Optimized Training and usage script for DQN-enhanced Minimax Bot
"""

import sys
import os
import time
sys.path.append('.')  # Add current directory to path

from src.board import GameBoard
from src.minimax import MinimaxBot  # Your updated bot
import random
import numpy as np

def train_new_bot():
    """Train a new DQN bot from scratch - OPTIMIZED VERSION"""
    print("=== Training New DQN Bot (OPTIMIZED) ===")
    
    # Create bot with DQN enabled - using optimal settings for fast training
    bot = MinimaxBot(
        player=1, 
        max_depth=4,  # Reasonable depth for actual play
        use_dqn=True, 
        model_path="my_trained_bot.pth"
    )
    
    print("Starting optimized training...")
    print("This should take 10-30 minutes instead of 22 hours!")
    
    start_time = time.time()
    bot.train_dqn(episodes=2000, save_interval=200)  # Reduced episodes
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"Training completed in {training_time/60:.1f} minutes!")
    print("Testing bot...")
    test_bot_performance(bot, games=50)

def train_quick_test():
    """Quick training test - just 100 episodes for testing"""
    print("=== Quick Training Test ===")
    
    bot = MinimaxBot(
        player=1, 
        max_depth=3, 
        use_dqn=True, 
        model_path="test_bot.pth"
    )
    
    print("Running quick training test (100 episodes)...")
    start_time = time.time()
    bot.train_dqn(episodes=100, save_interval=50)
    end_time = time.time()
    
    print(f"Quick test completed in {end_time - start_time:.1f} seconds!")
    test_bot_performance(bot, games=20)

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
        max_depth=4,  # Use full depth for actual play
        use_dqn=True, 
        model_path=model_path
    )
    
    print("Bot loaded successfully!")
    return bot

def test_bot_performance(bot, games=50):
    """Test bot performance against random player"""
    print(f"\n=== Testing Bot Performance ({games} games) ===")
    
    wins = 0
    losses = 0
    draws = 0
    total_game_length = 0
    
    start_time = time.time()
    
    for game in range(games):
        board = GameBoard()
        bot.board = board  # Set board reference
        current_player = 1
        game_length = 0
        max_moves = 50
        
        while not board.is_terminal() and game_length < max_moves:
            valid_moves = board.get_valid_moves()
            
            if current_player == bot.player:
                # Use trained bot - fast move for testing
                move = bot.get_fast_move() if hasattr(bot, 'get_fast_move') else bot.get_best_move(training_mode=True)
            else:
                # Random opponent
                move = random.choice(valid_moves)
            
            board.set_move(move, current_player)
            current_player = 3 - current_player
            game_length += 1
        
        total_game_length += game_length
        
        # Check game result
        result = board.get_game_result(bot.player)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
        
        if (game + 1) % 10 == 0:
            current_win_rate = wins / (game + 1)
            elapsed = time.time() - start_time
            print(f"Progress: {game + 1}/{games} games, Win rate: {current_win_rate:.3f}, "
                  f"Time: {elapsed:.1f}s")
    
    end_time = time.time()
    win_rate = wins / games
    avg_game_length = total_game_length / games
    
    print(f"\nFinal Results: {wins}W-{losses}L-{draws}D")
    print(f"Win Rate: {win_rate:.3f}")
    print(f"Average game length: {avg_game_length:.1f} moves")
    print(f"Testing completed in {end_time - start_time:.1f} seconds")
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
            start_time = time.time()
            move = bot.get_best_move()  # Use full depth for human games
            think_time = time.time() - start_time
            print(f"Bot chooses: {move} (thought for {think_time:.1f}s)")
        
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
    games = 30  # Reduced for faster testing
    total_time = 0
    
    print(f"Playing {games} games: DQN vs Heuristic")
    
    start_time = time.time()
    
    for game in range(games):
        board = GameBoard()
        dqn_bot.board = board
        heuristic_bot.board = board
        
        current_player = 1
        game_length = 0
        
        while not board.is_terminal() and game_length < 50:
            if current_player == 1:
                move = dqn_bot.get_best_move(training_mode=True)  # Use training mode for speed
            else:
                move = heuristic_bot.get_best_move(training_mode=True)
            
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
        
        if (game + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Games played: {game + 1}/{games}, Time: {elapsed:.1f}s")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nResults (completed in {total_time:.1f} seconds):")
    print(f"DQN Bot: {wins_dqn} wins")
    print(f"Heuristic Bot: {wins_heuristic} wins")
    print(f"Draws: {draws}")
    print(f"DQN Win Rate: {wins_dqn/games:.3f}")

def benchmark_speeds():
    """Benchmark different configurations to show speed improvements"""
    print("=== Speed Benchmark ===")
    
    # Test different configurations
    configs = [
        ("Random vs Random", None, None),
        ("Heuristic (depth 2)", 2, False),
        ("Heuristic (depth 3)", 3, False),
        ("DQN + Minimax (depth 2)", 2, True),
        ("DQN + Minimax (depth 3)", 3, True),
    ]
    
    for name, depth, use_dqn in configs:
        print(f"\nTesting: {name}")
        
        if depth is None:
            # Pure random
            start_time = time.time()
            for _ in range(10):
                board = GameBoard()
                current_player = 1
                while not board.is_terminal():
                    moves = board.get_valid_moves()
                    if not moves:
                        break
                    move = random.choice(moves)
                    board.set_move(move, current_player)
                    current_player = 3 - current_player
            elapsed = time.time() - start_time
            print(f"  10 random games: {elapsed:.2f}s ({elapsed/10:.3f}s per game)")
        else:
            # Bot testing
            bot = MinimaxBot(player=1, max_depth=depth, use_dqn=use_dqn)
            if use_dqn and os.path.exists("my_trained_bot.pth"):
                bot.load_dqn_model()
            
            start_time = time.time()
            test_bot_performance(bot, games=10)
            elapsed = time.time() - start_time
            print(f"  Total time: {elapsed:.2f}s ({elapsed/10:.3f}s per game)")

def main():
    """Main menu"""
    while True:
        print("\n" + "="*60)
        print("OPTIMIZED DQN Minimax Bot Training & Testing")
        print("="*60)
        print("1. Train new DQN bot (FAST - ~15 minutes)")
        print("2. Quick training test (100 episodes - ~30 seconds)")
        print("3. Test trained bot vs random")
        print("4. Play against bot")
        print("5. Compare DQN vs Heuristic")
        print("6. Speed benchmark")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            train_new_bot()
        elif choice == '2':
            train_quick_test()
        elif choice == '3':
            bot = load_and_use_trained_bot()
            if bot:
                test_bot_performance(bot, games=50)
        elif choice == '4':
            play_human_vs_bot()
        elif choice == '5':
            compare_dqn_vs_heuristic()
        elif choice == '6':
            benchmark_speeds()
        elif choice == '7':
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()