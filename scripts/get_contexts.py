import argparse

from lm_kbc.common.utils import (
    get_contexts
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Get contexts using DuckDuckGo client"
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="./data/raw/lm-kbc/dataset/data/dev.jsonl",
        help='Path to the input (default "./data/raw/lm-kbc/dataset/data/dev.jsonl")',
    )

    parser.add_argument(
        "--contexts_path",
        type=str,
        default="./data/processed/train/contexts/contexts.json",
        help='Path to the contexts (default "./data/processed/train/contexts\
        /contexts.json")',
    )

    args = parser.parse_args()

    return args


def main(args):

  get_contexts(args.input_path,args.contexts_path)


if __name__ == "__main__":
    main(parse_args())
