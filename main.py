import sys

from src.client import GameClient


def main():
    """Main function"""
    if len(sys.argv) != 6:
        print(
            "Usage: python minimax.py <server_ip> <port> <player_num> <player_name> <search_depth>"
        )
        print("Example: python minimax.py 127.0.0.1 8888 1 MinimaxBot 5")
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

    client = GameClient(server_ip, server_port, player_num, player_name, search_depth)
    client.play_game()


if __name__ == "__main__":
    main()
