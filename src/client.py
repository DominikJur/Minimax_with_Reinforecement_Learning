import socket

from src.board import GameBoard
from src.minimax import MinimaxBot


class GameClient:
    """Socket client for communicating with the game server"""

    def __init__(
        self,
        server_ip: str,
        server_port: int,
        player_num: int,
        player_name: str,
        search_depth: int,
    ):
        self.server_ip = server_ip
        self.server_port = server_port
        self.player_num = player_num
        self.player_name = player_name
        self.search_depth = search_depth
        self.socket = None
        self.bot = MinimaxBot(player_num, search_depth)

    def connect_to_server(self) -> bool:
        """Connect to game server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.server_port))
            print(f"Connected to server at {self.server_ip}:{self.server_port}")
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False

    def send_message(self, message: str) -> bool:
        """Send message to server"""
        try:
            self.socket.send(message.encode())
            return True
        except Exception as e:
            print(f"Failed to send message: {e}")
            return False

    def receive_message(self) -> str:
        """Receive message from server"""
        try:
            data = self.socket.recv(16)
            return data.decode().strip()
        except Exception as e:
            print(f"Failed to receive message: {e}")
            return ""

    def play_game(self):
        """Main game loop"""
        if not self.connect_to_server():
            return

        try:
            # Wait for initial server message (700)
            server_msg = self.receive_message()
            print(f"Server: {server_msg}")

            # Send player info
            player_info = f"{self.player_num} {self.player_name}"
            self.send_message(player_info)
            print(f"Sent: {player_info}")

            # Reset board
            self.bot.board.reset()

            # Game loop
            game_over = False
            while not game_over:
                server_msg = self.receive_message()
                print(f"Server: {server_msg}")

                if not server_msg:
                    break

                msg_code = int(server_msg)
                move = msg_code % 100
                status = msg_code // 100

                # Update board with opponent's move
                if move != 0:
                    self.bot.board.set_move(move, 3 - self.player_num)

                # Check game status
                if status == 0 or status == 6:  # Our turn
                    best_move = self.bot.get_best_move()
                    self.bot.board.set_move(best_move, self.player_num)
                    self.send_message(str(best_move))
                    print(f"Made move: {best_move}")

                else:  # Game over
                    game_over = True
                    if status == 1:
                        print("We won!")
                    elif status == 2:
                        print("We lost!")
                    elif status == 3:
                        print("Draw!")
                    elif status == 4:
                        print("We won due to opponent error!")
                    elif status == 5:
                        print("We lost due to our error!")

        finally:
            if self.socket:
                self.socket.close()
