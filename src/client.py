import sys
import socket
import json
import time
from src.minimax import MinimaxBot
from src.board import GameBoard

class GameClient:
    """Client for connecting DQN-enhanced MinimaxBot to game server"""
    
    def __init__(self, server_ip, server_port, player_num, player_name, search_depth):
        self.server_ip = server_ip
        self.server_port = server_port
        self.player_num = player_num
        self.player_name = player_name
        self.search_depth = search_depth
        
        # Initialize our optimized bot
        self.bot = MinimaxBot(
            player=player_num, 
            max_depth=search_depth, 
            use_dqn=True,
            model_path=f"dqn_bot_depth_{search_depth}.pth"
        )
        
        self.socket = None
        self.game_board = GameBoard()
        
    def connect_to_server(self):
        """Establish connection to game server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.server_port))
            print(f"Connected to server {self.server_ip}:{self.server_port}")
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False
    
    def send_message(self, message):
        """Send message to server"""
        try:
            message_str = json.dumps(message) + '\n'
            self.socket.send(message_str.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Failed to send message: {e}")
            return False
    
    def receive_message(self):
        """Receive message from server"""
        try:
            buffer = ""
            while True:
                data = self.socket.recv(1024).decode('utf-8')
                if not data:
                    break
                buffer += data
                if '\n' in buffer:
                    message_str = buffer.split('\n')[0]
                    return json.loads(message_str)
        except Exception as e:
            print(f"Failed to receive message: {e}")
            return None
    
    def register_player(self):
        """Register with the game server"""
        register_message = {
            "type": "register",
            "player_num": self.player_num,
            "player_name": self.player_name
        }
        
        if not self.send_message(register_message):
            return False
            
        response = self.receive_message()
        if response and response.get("status") == "ok":
            print(f"Successfully registered as Player {self.player_num}: {self.player_name}")
            return True
        else:
            print(f"Registration failed: {response}")
            return False
    
    def update_board_from_server(self, board_state):
        """Update local board from server state"""
        if isinstance(board_state, list) and len(board_state) == 5:
            for i in range(5):
                for j in range(5):
                    self.game_board.board[i][j] = board_state[i][j]
            # Update bot's board reference
            self.bot.board = self.game_board
        else:
            print("Invalid board state received from server")
    
    def make_move(self):
        """Get move from bot and send to server"""
        print(f"\nPlayer {self.player_num}'s turn - thinking...")
        
        # Clear bot's cache for fresh game analysis
        if hasattr(self.bot, 'clear_cache'):
            self.bot.clear_cache()
        
        # Get best move from our optimized bot
        start_time = time.time()
        move = self.bot.get_best_move(training_mode=False)
        think_time = time.time() - start_time
        
        print(f"Selected move: {move} (thought for {think_time:.2f}s)")
        
        # Send move to server
        move_message = {
            "type": "move",
            "player": self.player_num,
            "position": move
        }
        
        return self.send_message(move_message)
    
    def display_board(self):
        """Display current board state"""
        print("\nCurrent board:")
        for i, row in enumerate(self.game_board.board):
            row_str = f"{i+1} "
            for cell in row:
                if cell == 0:
                    row_str += ". "
                elif cell == 1:
                    row_str += "X "
                else:
                    row_str += "O "
            print(row_str)
        print()
    
    def play_game(self):
        """Main game loop"""
        print(f"Starting DQN-Enhanced MinimaxBot (depth {self.search_depth})")
        print(f"Player: {self.player_num} ({self.player_name})")
        
        # Connect to server
        if not self.connect_to_server():
            return
        
        # Register player
        if not self.register_player():
            self.socket.close()
            return
        
        # Game loop
        try:
            while True:
                # Wait for message from server
                message = self.receive_message()
                if not message:
                    print("Connection lost")
                    break
                
                message_type = message.get("type")
                
                if message_type == "game_start":
                    print("Game started!")
                    self.game_board.reset()
                    self.bot.board = self.game_board
                    
                elif message_type == "board_update":
                    # Update board state
                    board_state = message.get("board")
                    if board_state:
                        self.update_board_from_server(board_state)
                        self.display_board()
                    
                elif message_type == "your_turn":
                    # Make our move
                    if not self.make_move():
                        print("Failed to send move")
                        break
                        
                elif message_type == "opponent_move":
                    # Opponent made a move
                    move = message.get("position")
                    opponent = 3 - self.player_num
                    print(f"Opponent played: {move}")
                    
                    if move and self.game_board.is_valid_move(move):
                        self.game_board.set_move(move, opponent)
                        self.bot.board = self.game_board
                    
                elif message_type == "game_over":
                    # Game finished
                    result = message.get("result")
                    winner = message.get("winner")
                    
                    self.display_board()
                    
                    if result == "win" and winner == self.player_num:
                        print("🎉 Victory! Our bot won!")
                    elif result == "loss":
                        print("😞 Defeat. Better luck next time!")
                    elif result == "draw":
                        print("🤝 Draw game!")
                    else:
                        print(f"Game over: {result}")
                    
                    break
                    
                elif message_type == "error":
                    error_msg = message.get("message", "Unknown error")
                    print(f"Server error: {error_msg}")
                    break
                    
                elif message_type == "invalid_move":
                    print("Invalid move! This shouldn't happen with our bot...")
                    break
                    
                else:
                    print(f"Unknown message type: {message_type}")
        
        except KeyboardInterrupt:
            print("\nGame interrupted by user")
        except Exception as e:
            print(f"Error during game: {e}")
        finally:
            if self.socket:
                self.socket.close()
                print("Disconnected from server")

def main():
    """Main function"""
    if len(sys.argv) != 6:
        print(
            "Usage: python main.py <server_ip> <port> <player_num> <player_name> <search_depth>"
        )
        print("Example: python main.py 127.0.0.1 8888 1 MinimaxBot 5")
        return

    server_ip = sys.argv[1]
    server_port = int(sys.argv[2])
    player_num = int(sys.argv[3])
    player_name = sys.argv[4]
    search_depth = int(sys.argv[5])

    if player_num not in [1, 2]:
        print("Player number must be 1 or 2")
        return

    if search_depth < 1 or search_depth > 10:
        print("Search depth must be between 1 and 10")
        return

    if len(player_name) > 9:
        print("Player name must be 9 characters or less")
        return

    # Create and run game client with our optimized bot
    client = GameClient(server_ip, server_port, player_num, player_name, search_depth)
    client.play_game()

if __name__ == "__main__":
    main()