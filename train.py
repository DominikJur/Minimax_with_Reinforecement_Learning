"""
Clean DQN Training Script - ONLY Self-Play Speed Optimized
"""

import sys
import os
import time
sys.path.append('.')

from src.board import GameBoard
from src.minimax import MinimaxBot
import random

def train_bot():
    """Train DQN bot - SAME intelligence with speed optimizations"""
    print("=== Training DQN Bot (Speed Optimized) ===")
    
    model_path = "clean_dqn_bot.pth"
    
    # Handle existing model
    if os.path.exists(model_path):
        choice = input(f"Model {model_path} exists. Delete and start fresh? (y/n): ")
        if choice.lower() == 'y':
            os.remove(model_path)
            print("Starting fresh training...")
    
    # Create and train bot - SAME settings as original
    bot = MinimaxBot(player=1, max_depth=4, use_dqn=True, model_path=model_path)
    
    print("Training strategy (UNCHANGED):")
    print("  15% random exploration")
    print("  75% vs heuristic opponents") 
    print("  10% self-play (NOW OPTIMIZED!)")
    print("  Optimizations: Move ordering + Position caching + Parallel self-play")
    print("  Expected time: 15-25 minutes (self-play ~3x faster)")
    
    start_time = time.time()
    bot.train_dqn(episodes=3000, save_interval=300)  # SAME as original
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time/60:.1f} minutes!")
    
    # Quick test
    print("Testing vs heuristic bot...")
    test_performance(bot)

def quick_train():
    """Quick training for testing - UNCHANGED"""
    print("=== Quick Training Test (500 episodes) ===")
    
    bot = MinimaxBot(player=1, max_depth=3, use_dqn=True, model_path="quick_test.pth")
    
    start_time = time.time()
    bot.train_dqn(episodes=500, save_interval=100)
    training_time = time.time() - start_time
    
    print(f"Quick training done in {training_time:.1f} seconds!")
    test_performance(bot, games=20)

def test_performance(bot, games=30):
    """Test bot performance - UNCHANGED"""
    print(f"\nTesting vs heuristic bot ({games} games)...")
    
    # Create heuristic opponent
    heuristic_bot = MinimaxBot(player=2, max_depth=3, use_dqn=False)
    
    wins = 0
    losses = 0
    draws = 0
    
    for game in range(games):
        board = GameBoard()
        bot.board = board
        heuristic_bot.board = board
        
        current_player = 1
        while not board.is_terminal():
            if current_player == 1:
                move = bot.get_best_move(training_mode=True)
            else:
                move = heuristic_bot.get_best_move(training_mode=True)
            
            board.set_move(move, current_player)
            current_player = 3 - current_player
        
        result = board.get_game_result(1)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
        
        if (game + 1) % 10 == 0:
            print(f"  Progress: {game+1}/{games}, Win rate: {wins/(game+1):.3f}")
    
    win_rate = wins / games
    print(f"\nResults: {wins}W-{losses}L-{draws}D")
    print(f"Win rate: {win_rate:.3f}")
    
    if win_rate >= 0.6:
        print("✅ Good performance! Bot is learning well.")
    elif win_rate >= 0.4:
        print("⚠️  Decent performance, could use more training.")
    else:
        print("❌ Poor performance, needs more training.")
    
    return win_rate

def play_vs_bot():
    """Play against trained bot - UNCHANGED"""
    print("=== Play vs Bot ===")
    
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    if not model_files:
        print("No trained models found!")
        return
    
    print("Available models:")
    for i, model in enumerate(model_files):
        print(f"  {i+1}. {model}")
    
    try:
        choice = int(input("Choose model (number): ")) - 1
        model_path = model_files[choice]
    except:
        print("Invalid choice!")
        return
    
    bot = MinimaxBot(player=2, max_depth=4, use_dqn=True, model_path=model_path)
    board = GameBoard()
    bot.board = board
    
    human_player = 1
    current_player = 1
    
    print("\nBoard positions (11-55):")
    print("11 12 13 14 15")
    print("21 22 23 24 25")
    print("31 32 33 34 35") 
    print("41 42 43 44 45")
    print("51 52 53 54 55")
    
    while not board.is_terminal():
        # Show board
        print("\nCurrent board:")
        for i, row in enumerate(board.board):
            row_str = f"{i+1} "
            for cell in row:
                if cell == 0:
                    row_str += ". "
                elif cell == 1:
                    row_str += "X "
                else:
                    row_str += "O "
            print(row_str)
        
        if current_player == human_player:
            # Human turn
            valid_moves = board.get_valid_moves()
            print(f"\nYour turn! Valid moves: {valid_moves}")
            
            while True:
                try:
                    move = int(input("Enter move: "))
                    if move in valid_moves:
                        break
                    print("Invalid move!")
                except:
                    print("Enter a number!")
        else:
            # Bot turn
            print("\nBot thinking...")
            move = bot.get_best_move()
            print(f"Bot plays: {move}")
        
        board.set_move(move, current_player)
        current_player = 3 - current_player
    
    # Game over
    print("\nFinal board:")
    for i, row in enumerate(board.board):
        row_str = f"{i+1} "
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
        print("\n🎉 You won!")
    elif result == -1:
        print("\n🤖 Bot won!")
    else:
        print("\n🤝 Draw!")

def compare_bots():
    """Compare DQN vs pure heuristic - UNCHANGED"""
    print("=== DQN vs Heuristic Comparison ===")
    
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    if not model_files:
        print("No DQN models found!")
        return
    
    print("Available models:")
    for i, model in enumerate(model_files):
        print(f"  {i+1}. {model}")
    
    try:
        choice = int(input("Choose DQN model: ")) - 1
        model_path = model_files[choice]
    except:
        print("Invalid choice!")
        return
    
    dqn_bot = MinimaxBot(player=1, max_depth=3, use_dqn=True, model_path=model_path)
    heuristic_bot = MinimaxBot(player=2, max_depth=3, use_dqn=False)
    
    games = 30
    wins_dqn = 0
    wins_heuristic = 0
    draws = 0
    
    print(f"\nPlaying {games} games...")
    
    for game in range(games):
        board = GameBoard()
        dqn_bot.board = board
        heuristic_bot.board = board
        
        current_player = 1
        while not board.is_terminal():
            if current_player == 1:
                move = dqn_bot.get_best_move(training_mode=True)
            else:
                move = heuristic_bot.get_best_move(training_mode=True)
            
            board.set_move(move, current_player)
            current_player = 3 - current_player
        
        result = board.get_game_result(1)
        if result == 1:
            wins_dqn += 1
        elif result == -1:
            wins_heuristic += 1
        else:
            draws += 1
        
        if (game + 1) % 10 == 0:
            print(f"  Progress: {game+1}/{games}")
    
    print(f"\nResults:")
    print(f"  DQN Bot: {wins_dqn} wins ({wins_dqn/games:.3f})")
    print(f"  Heuristic Bot: {wins_heuristic} wins ({wins_heuristic/games:.3f})")
    print(f"  Draws: {draws}")
    
    if wins_dqn > wins_heuristic:
        improvement = (wins_dqn - wins_heuristic) / games * 100
        print(f"✅ DQN bot is {improvement:.1f}% better!")
    elif wins_dqn == wins_heuristic:
        print("⚖️  Equal performance")
    else:
        print("❌ DQN bot needs more training")

def clean_models():
    """Delete old model files - UNCHANGED"""
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    
    if not model_files:
        print("No model files found.")
        return
    
    print("Model files:")
    for i, model in enumerate(model_files):
        size = os.path.getsize(model) / 1024 / 1024
        print(f"  {i+1}. {model} ({size:.1f}MB)")
    
    choice = input("\nDelete which models? (numbers like '1,3' or 'all'): ")
    
    if choice.lower() == 'all':
        for model in model_files:
            os.remove(model)
            print(f"Deleted {model}")
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            for i in indices:
                if 0 <= i < len(model_files):
                    os.remove(model_files[i])
                    print(f"Deleted {model_files[i]}")
        except:
            print("Invalid input!")

def main():
    """Simple main menu - UNCHANGED"""
    while True:
        print("\n" + "="*50)
        print("Clean DQN Minimax Bot (Move Ordering + Caching + Parallel)")
        print("="*50)
        print("1. Train new bot (3000 episodes, ~20 min)")
        print("2. Quick test (500 episodes, ~3 min)")
        print("3. Test existing bot")
        print("4. Play vs bot")
        print("5. Compare DQN vs heuristic")
        print("6. Clean old models")
        print("7. Exit")
        
        choice = input("\nChoice (1-7): ").strip()
        
        try:
            if choice == '1':
                train_bot()
            elif choice == '2':
                quick_train()
            elif choice == '3':
                model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
                if model_files:
                    print("Available models:", model_files)
                    model = input("Model name: ").strip()
                    if model in model_files:
                        bot = MinimaxBot(player=1, max_depth=4, use_dqn=True, model_path=model)
                        test_performance(bot)
                    else:
                        print("Model not found!")
                else:
                    print("No models found!")
            elif choice == '4':
                play_vs_bot()
            elif choice == '5':
                compare_bots()
            elif choice == '6':
                clean_models()
            elif choice == '7':
                print("Goodbye!")
                break
            else:
                print("Invalid choice!")
        
        except KeyboardInterrupt:
            print("\nCancelled.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()