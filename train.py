"""
Enhanced Training and usage script for DQN-enhanced Minimax Bot
With curriculum learning and advanced features
"""

import sys
import os
import time
sys.path.append('.')  # Add current directory to path

from src.board import GameBoard
from src.minimax import MinimaxBot  # Your updated bot
import random
import numpy as np
import matplotlib.pyplot as plt

def train_enhanced_bot():
    """Train the enhanced DQN bot with curriculum learning"""
    print("=== Training Enhanced DQN Bot ===")
    print("🚀 Features: Enhanced state representation, curriculum learning, combined evaluation")
    
    model_path = "enhanced_dqn_bot.pth"
    
    # Check for existing model
    if os.path.exists(model_path):
        print(f"Found existing model: {model_path}")
        choice = input("Continue training existing model? (y/n): ").lower().strip()
        if choice != 'y':
            backup_name = f"backup_{int(time.time())}.pth"
            os.rename(model_path, backup_name)
            print(f"Backed up old model to {backup_name}")
    
    # Create enhanced bot
    bot = MinimaxBot(
        player=1, 
        max_depth=4,
        use_dqn=True, 
        model_path=model_path
    )
    
    print("\n📊 Training Strategy:")
    print("  Phase 1 (20%): Random vs Heuristic exploration")
    print("  Phase 2 (60%): Focused training vs Heuristic opponents") 
    print("  Phase 3 (20%): Self-play and advanced tactics")
    print("  Expected time: 30-60 minutes for quality training")
    
    start_time = time.time()
    bot.train_dqn(episodes=5000, save_interval=500)
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"\n✅ Training completed in {training_time/60:.1f} minutes!")
    
    print("\n🧪 Testing trained bot...")
    test_enhanced_performance(bot)

def train_quick_enhanced():
    """Quick enhanced training for testing"""
    print("=== Quick Enhanced Training Test ===")
    
    bot = MinimaxBot(
        player=1, 
        max_depth=3, 
        use_dqn=True, 
        model_path="quick_enhanced_test.pth"
    )
    
    print("Running quick enhanced training (500 episodes)...")
    start_time = time.time()
    bot.train_dqn(episodes=500, save_interval=100)
    end_time = time.time()
    
    print(f"Quick test completed in {end_time - start_time:.1f} seconds!")
    test_enhanced_performance(bot, games=20)

def load_enhanced_bot():
    """Load existing enhanced trained bot"""
    print("=== Loading Enhanced Bot ===")
    
    model_path = "enhanced_dqn_bot.pth"
    if not os.path.exists(model_path):
        print(f"No enhanced model found at {model_path}")
        print("Try training first, or check for other model files:")
        
        # Look for other model files
        model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        if model_files:
            print("Available models:", model_files)
            choice = input("Enter model filename to load (or press Enter to cancel): ").strip()
            if choice and choice in model_files:
                model_path = choice
            else:
                return None
        else:
            print("No model files found!")
            return None
    
    bot = MinimaxBot(
        player=1, 
        max_depth=4,
        use_dqn=True, 
        model_path=model_path
    )
    
    print("Enhanced bot loaded successfully!")
    return bot

def test_enhanced_performance(bot, games=50):
    """Test enhanced bot performance with detailed analysis"""
    print(f"\n=== Enhanced Performance Test ({games} games) ===")
    
    # Test against different opponents
    opponents = [
        ("Random", "random"),
        ("Heuristic (depth 2)", "heuristic_2"),
        ("Heuristic (depth 3)", "heuristic_3"),
        ("Heuristic (depth 4)", "heuristic_4")
    ]
    
    results = {}
    
    for opp_name, opp_type in opponents:
        print(f"\n🆚 Testing vs {opp_name}...")
        wins = 0
        losses = 0
        draws = 0
        total_time = 0
        total_moves = 0
        
        start_time = time.time()
        
        for game in range(games//len(opponents) if len(opponents) > 1 else games):
            board = GameBoard()
            bot.board = board
            
            # Create opponent
            if opp_type == "random":
                opponent_bot = None
            else:
                depth = int(opp_type.split('_')[1])
                opponent_bot = MinimaxBot(player=2, max_depth=depth, use_dqn=False)
                opponent_bot.board = board
            
            # Play game
            game_moves = play_single_game(bot, opponent_bot, board)
            total_moves += game_moves
            
            # Check result
            result = board.get_game_result(bot.player)
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1
        
        end_time = time.time()
        test_games = games//len(opponents) if len(opponents) > 1 else games
        
        win_rate = wins / test_games if test_games > 0 else 0
        avg_moves = total_moves / test_games if test_games > 0 else 0
        
        results[opp_name] = {
            'win_rate': win_rate,
            'record': f"{wins}W-{losses}L-{draws}D",
            'avg_moves': avg_moves,
            'time': end_time - start_time
        }
        
        print(f"  Result: {wins}W-{losses}L-{draws}D (Win rate: {win_rate:.3f})")
        print(f"  Avg moves per game: {avg_moves:.1f}")
        print(f"  Time: {end_time - start_time:.1f}s")
    
    # Summary
    print(f"\n📈 Performance Summary:")
    for opp_name, data in results.items():
        print(f"  vs {opp_name:20s}: {data['record']:12s} ({data['win_rate']:.3f} win rate)")
    
    return results

def play_single_game(bot1, bot2, board):
    """Play a single game between two bots"""
    current_player = 1
    move_count = 0
    max_moves = 50
    
    while not board.is_terminal() and move_count < max_moves:
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            break
        
        if current_player == 1:
            # Bot1's turn
            move = bot1.get_best_move(training_mode=True)
        else:
            # Bot2 or random
            if bot2 is None:
                move = random.choice(valid_moves)
            else:
                move = bot2.get_best_move(training_mode=True)
        
        board.set_move(move, current_player)
        current_player = 3 - current_player
        move_count += 1
    
    return move_count

def play_human_vs_enhanced_bot():
    """Play against the enhanced bot with better interface"""
    print("=== Human vs Enhanced Bot ===")
    
    bot = load_enhanced_bot()
    if bot is None:
        return
    
    board = GameBoard()
    bot.board = board
    
    print("\n🎮 Game Setup")
    human_player = int(input("Choose your player (1=X, 2=O): "))
    bot.player = 3 - human_player
    bot.opponent = human_player
    
    difficulty = input("Choose bot difficulty (easy/medium/hard): ").lower().strip()
    if difficulty == "easy":
        bot.max_depth = 2
    elif difficulty == "medium":
        bot.max_depth = 3
    else:
        bot.max_depth = 4
    
    current_player = 1
    move_history = []
    
    print("\n📍 Board positions:")
    print("11 12 13 14 15")
    print("21 22 23 24 25") 
    print("31 32 33 34 35")
    print("41 42 43 44 45")
    print("51 52 53 54 55")
    print()
    
    while not board.is_terminal():
        # Display board with better formatting
        print("Current board:")
        print("   1 2 3 4 5")
        for i, row in enumerate(board.board):
            row_str = f"{i+1}  "
            for cell in row:
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
            print(f"Your turn (Player {human_player} = {'X' if human_player == 1 else 'O'})")
            print(f"Valid moves: {valid_moves}")
            
            while True:
                try:
                    move_input = input("Enter your move (or 'undo' to undo last move): ").strip()
                    
                    if move_input.lower() == 'undo' and move_history:
                        last_move = move_history.pop()
                        board.undo_move(last_move)
                        if move_history:  # Undo bot's move too
                            last_move = move_history.pop()
                            board.undo_move(last_move)
                        print("Last moves undone!")
                        break
                    
                    move = int(move_input)
                    if move in valid_moves:
                        move_history.append(move)
                        break
                    else:
                        print("Invalid move! Try again.")
                except ValueError:
                    print("Please enter a number or 'undo'!")
        else:
            # Bot turn
            print(f"🤖 Bot thinking (Player {bot.player} = {'X' if bot.player == 1 else 'O'})...")
            think_start = time.time()
            move = bot.get_best_move()
            think_time = time.time() - think_start
            print(f"Bot chooses: {move} (thought for {think_time:.1f}s)")
            move_history.append(move)
        
        board.set_move(move, current_player)
        current_player = 3 - current_player
    
    # Final result with better display
    print("\n🏁 Game Over!")
    print("Final board:")
    print("   1 2 3 4 5")
    for i, row in enumerate(board.board):
        row_str = f"{i+1}  "
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
        print("🎉 You won! Great job!")
    elif result == -1:
        print("🤖 Bot won! Better luck next time!")
    else:
        print("🤝 It's a draw!")
    
    print(f"Game lasted {len(move_history)} moves")

def compare_enhanced_vs_original():
    """Compare enhanced DQN vs original heuristic"""
    print("=== Enhanced DQN vs Original Heuristic ===")
    
    enhanced_model = "enhanced_dqn_bot.pth"
    if not os.path.exists(enhanced_model):
        print("No enhanced model found! Train it first.")
        return
    
    # Create both bots
    enhanced_bot = MinimaxBot(player=1, max_depth=3, use_dqn=True, model_path=enhanced_model)
    heuristic_bot = MinimaxBot(player=2, max_depth=3, use_dqn=False)
    
    games = 30
    wins_enhanced = 0
    wins_heuristic = 0
    draws = 0
    
    print(f"🏟️  Playing {games} games: Enhanced DQN vs Pure Heuristic")
    
    start_time = time.time()
    move_counts = []
    
    for game in range(games):
        board = GameBoard()
        enhanced_bot.board = board
        heuristic_bot.board = board
        
        moves = play_single_game(enhanced_bot, heuristic_bot, board)
        move_counts.append(moves)
        
        result = board.get_game_result(1)  # Enhanced bot's perspective
        if result == 1:
            wins_enhanced += 1
        elif result == -1:
            wins_heuristic += 1
        else:
            draws += 1
        
        if (game + 1) % 5 == 0:
            current_rate = wins_enhanced / (game + 1)
            print(f"  Progress: {game + 1}/{games} games, Enhanced win rate: {current_rate:.3f}")
    
    end_time = time.time()
    
    print(f"\n📊 Final Results:")
    print(f"  Enhanced DQN: {wins_enhanced} wins ({wins_enhanced/games:.3f})")
    print(f"  Pure Heuristic: {wins_heuristic} wins ({wins_heuristic/games:.3f})")
    print(f"  Draws: {draws}")
    print(f"  Average game length: {np.mean(move_counts):.1f} moves")
    print(f"  Total time: {end_time - start_time:.1f}s")
    
    if wins_enhanced > wins_heuristic:
        improvement = (wins_enhanced - wins_heuristic) / games * 100
        print(f"🎯 Enhanced bot shows {improvement:.1f}% improvement!")
    else:
        print("🔧 Enhanced bot needs more training or tuning.")

def analyze_bot_features():
    """Analyze what the enhanced bot has learned"""
    print("=== Bot Feature Analysis ===")
    
    bot = load_enhanced_bot()
    if bot is None:
        return
    
    # Test on various board positions
    test_positions = [
        # Empty board
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],
        # Center control
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0]],
        # Winning threat
        [[1,1,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],
        # Losing danger  
        [[1,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],
    ]
    
    position_names = ["Empty", "Center Control", "Winning Threat", "Losing Danger"]
    
    print("🔍 Analyzing bot's evaluation of different positions:")
    
    for i, (position, name) in enumerate(zip(test_positions, position_names)):
        bot.board.board = [row[:] for row in position]  # Deep copy
        
        state = bot.board_to_enhanced_state()
        dqn_value = bot.dqn_agent.get_value(state) if bot.dqn_agent else 0
        heuristic_value = bot._evaluate_patterns(bot.player)
        combined_value = bot.evaluate_position(bot.player)
        
        print(f"\n  {name}:")
        print(f"    DQN Value: {dqn_value:.3f}")
        print(f"    Heuristic Value: {heuristic_value:.1f}")
        print(f"    Combined Value: {combined_value:.1f}")
        
        # Show some key features
        feature_names = ["Win Threats", "Win Opportunities", "Lose Dangers", 
                        "Center Control", "Game Progress"]
        key_features = [state[25], state[27], state[29], state[33], state[68]]
        
        print(f"    Key Features: " + 
              ", ".join(f"{name}={val:.2f}" for name, val in zip(feature_names, key_features)))

def benchmark_enhanced_speeds():
    """Benchmark the enhanced system"""
    print("=== Enhanced System Speed Benchmark ===")
    
    configs = [
        ("Random vs Random", None, None, None),
        ("Pure Heuristic (depth 3)", 3, False, None),
        ("Enhanced DQN (depth 3)", 3, True, "enhanced_dqn_bot.pth"),
        ("Combined Eval (depth 2)", 2, True, "enhanced_dqn_bot.pth"),
    ]
    
    for name, depth, use_dqn, model_path in configs:
        print(f"\n⚡ Testing: {name}")
        
        if depth is None:
            # Pure random
            start_time = time.time()
            for _ in range(5):
                board = GameBoard()
                current_player = 1
                moves = 0
                while not board.is_terminal() and moves < 30:
                    valid_moves = board.get_valid_moves()
                    if not valid_moves:
                        break
                    move = random.choice(valid_moves)
                    board.set_move(move, current_player)
                    current_player = 3 - current_player
                    moves += 1
            elapsed = time.time() - start_time
            print(f"    5 random games: {elapsed:.2f}s ({elapsed/5:.3f}s per game)")
        
        else:
            # Bot testing
            if model_path and not os.path.exists(model_path):
                print(f"    Model {model_path} not found, skipping...")
                continue
            
            try:
                bot = MinimaxBot(player=1, max_depth=depth, use_dqn=use_dqn, model_path=model_path)
                
                start_time = time.time()
                results = test_enhanced_performance(bot, games=10)
                elapsed = time.time() - start_time
                
                print(f"    10 games completed in {elapsed:.2f}s ({elapsed/10:.3f}s per game)")
                
            except Exception as e:
                print(f"    Error testing {name}: {e}")

def clean_old_models():
    """Clean up old incompatible model files"""
    print("=== Cleaning Old Models ===")
    
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    
    if not model_files:
        print("No model files found.")
        return
    
    print("Found model files:")
    for i, model_file in enumerate(model_files, 1):
        size = os.path.getsize(model_file) / 1024 / 1024  # MB
        mod_time = time.ctime(os.path.getmtime(model_file))
        print(f"  {i}. {model_file} ({size:.1f}MB, modified: {mod_time})")
    
    choice = input("\nDelete models? (Enter numbers like '1,3' or 'all' or 'none'): ").strip()
    
    if choice.lower() == 'none':
        return
    elif choice.lower() == 'all':
        indices = list(range(len(model_files)))
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
        except:
            print("Invalid input!")
            return
    
    deleted = 0
    for i in indices:
        if 0 <= i < len(model_files):
            os.remove(model_files[i])
            print(f"Deleted: {model_files[i]}")
            deleted += 1
    
    print(f"Cleaned up {deleted} model files.")

def main():
    """Enhanced main menu"""
    while True:
        print("\n" + "="*65)
        print("🚀 ENHANCED DQN Minimax Bot - Advanced Training & Testing")
        print("="*65)
        print("1. 🎯 Train Enhanced DQN Bot (5000 episodes, ~45 min)")
        print("2. ⚡ Quick Enhanced Test (500 episodes, ~5 min)")
        print("3. 📊 Test Enhanced Bot Performance")
        print("4. 🎮 Play vs Enhanced Bot")
        print("5. ⚔️  Enhanced vs Original Comparison")
        print("6. 🔍 Analyze Bot Features")
        print("7. ⚡ Speed Benchmarks")
        print("8. 🧹 Clean Old Models")
        print("9. 🚪 Exit")
        
        choice = input(f"\nEnter your choice (1-9): ").strip()
        
        try:
            if choice == '1':
                train_enhanced_bot()
            elif choice == '2':
                train_quick_enhanced()
            elif choice == '3':
                bot = load_enhanced_bot()
                if bot:
                    test_enhanced_performance(bot, games=50)
            elif choice == '4':
                play_human_vs_enhanced_bot()
            elif choice == '5':
                compare_enhanced_vs_original()
            elif choice == '6':
                analyze_bot_features()
            elif choice == '7':
                benchmark_enhanced_speeds()
            elif choice == '8':
                clean_old_models()
            elif choice == '9':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice! Please try again.")
        
        except KeyboardInterrupt:
            print("\n\n⏸️  Operation cancelled by user.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please check your setup and try again.")

if __name__ == "__main__":
    main()