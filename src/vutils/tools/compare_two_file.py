
import argparse

from ..print_color import print_blue as print


def compare_two_file():
    parser = argparse.ArgumentParser()
    parser.usage = "vvcli compare_two_file [OPTIONS]"
    parser.add_argument('--file_path1', type=str, required=True, help="Path to the 1st file")
    parser.add_argument('--file_path2', type=str, required=True, help="Path to the 2nd file")
    parser.add_argument('--mode', type=str, default="bin", help="\"bin\" or \"str\", default to \"bin\"")
    args = parser.parse_args()

    if args.mode == "bin":
        print("### comparing with \"binary\" mode...")
        kwargs = {"mode": "rb"}
    elif args.mode == "str":
        print("### comparing with \"string\" mode...")
        kwargs = {"mode": "r", "encoding": "utf-8"}
    else:
        raise ValueError("mode must be \"bin\" or \"str\"")
    with open(args.file_path1, **kwargs) as f:
        content1 = f.read()
    with open(args.file_path2, **kwargs) as f:
        content2 = f.read()
    if content1 == content2:
        print("The two files are the same.")
    else:
        print("The two files are different.")
