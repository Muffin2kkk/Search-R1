import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read and print one row from a parquet file."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/nq_hotpotqa_tiny/distillation/test.parquet"),
        help="Path to the parquet file.",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=123,
        help="0-based row index to read.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    file_path = args.file.resolve()

    table = pq.read_table(file_path)
    num_rows = table.num_rows

    if args.row < 0 or args.row >= num_rows:
        raise IndexError(f"Row index {args.row} is out of range. Valid range: 0 to {num_rows - 1}.")

    row = table.slice(args.row, 1).to_pylist()[0]
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
