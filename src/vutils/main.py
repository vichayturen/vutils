
import sys
import argparse
from enum import unique, Enum

from .data.web_data_view import web_data_view


def ping():
    parser = argparse.ArgumentParser()
    parser.usage = "vvcli ping [OPTIONS]"
    parser.add_argument('--name', type=str, default="World", help="Name to greet")
    args = parser.parse_args()
    print(f"Hello {args.name}")


def print_help():
    print("""
Usage:
  vvcli <command> [<args>]

Commands:
  ping           Print a friendly greeting
  web_data_view  Launch the data watching web demo
  help           Get usage
""")


@unique
class Command(str, Enum):
    PING = "ping"
    WEB_DATA_VIEW = "web_data_view"
    HELP = "help"


def main():
    command = sys.argv.pop(1) if len(sys.argv) > 1 else Command.HELP
    if command == Command.PING:
        ping()
    elif command == Command.WEB_DATA_VIEW:
        web_data_view()
    elif command == Command.HELP:
        print_help()
    else:
        raise NotImplementedError(f"Unknown command: {command}.")


if __name__ == '__main__':
    main()
