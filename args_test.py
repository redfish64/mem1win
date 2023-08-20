import argparse

def main():
    # Create the top-level parser
    parser = argparse.ArgumentParser(description="A simple CLI tool with commands")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add the first command and its arguments
    cmd1_parser = subparsers.add_parser("cmd1", help="Command 1 help")
    cmd1_parser.add_argument("--param1", type=str, help="Parameter 1 for Command 1")
    cmd1_parser.add_argument("--param2", type=int, help="Parameter 2 for Command 1")

    # Add the second command and its arguments
    cmd2_parser = subparsers.add_parser("cmd2", help="Command 2 help")
    cmd2_parser.add_argument("--param3", type=str, help="Parameter 3 for Command 2")
    cmd2_parser.add_argument("--param4", type=int, help="Parameter 4 for Command 2")

    # Parse the arguments
    args = parser.parse_args()

    # Process the commands and arguments
    if args.command == "cmd1":
        print(f"Executing Command 1 with param1={args.param1} and param2={args.param2}")
    elif args.command == "cmd2":
        print(f"Executing Command 2 with param3={args.param3} and param4={args.param4}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
