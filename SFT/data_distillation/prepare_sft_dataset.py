import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keep only the minimal fields needed for SFT training."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Input rollout file in jsonl or parquet format.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Output file in jsonl or parquet format.",
    )
    parser.add_argument(
        "--completed_only",
        action="store_true",
        help="Keep only rows with collector_status=completed.",
    )
    parser.add_argument(
        "--keep_data_source",
        action="store_true",
        help="Keep data_source in the output.",
    )
    parser.add_argument(
        "--keep_ability",
        action="store_true",
        help="Keep ability in the output.",
    )
    parser.add_argument(
        "--keep_extra_info",
        action="store_true",
        help="Keep extra_info in the output.",
    )
    parser.add_argument(
        "--keep_rollout_turns",
        action="store_true",
        help="Keep rollout_turns in the output for debugging or analysis.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict]:
    if path.suffix == ".jsonl":
        rows: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if path.suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=str(path), split="train")
        return list(dataset)

    raise ValueError(f"Unsupported input format: {path.suffix}")


def write_rows(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")
        return

    if path.suffix == ".parquet":
        Dataset.from_list(rows).to_parquet(str(path))
        return

    raise ValueError(f"Unsupported output format: {path.suffix}")


def select_messages(row: Dict) -> List[Dict]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return messages

    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        return prompt

    raise ValueError("Cannot find `messages` or fallback `prompt` in row.")


def build_sft_row(row: Dict, args) -> Dict:
    output = {"messages": select_messages(row)}

    if args.keep_data_source and "data_source" in row:
        output["data_source"] = row["data_source"]
    if args.keep_ability and "ability" in row:
        output["ability"] = row["ability"]
    if args.keep_extra_info and "extra_info" in row:
        output["extra_info"] = row["extra_info"]
    if args.keep_rollout_turns and "rollout_turns" in row:
        output["rollout_turns"] = row["rollout_turns"]

    return output


def main():
    args = parse_args()
    rows = load_rows(args.input_path)

    output_rows: List[Dict] = []
    for row in rows:
        if args.completed_only and row.get("collector_status") != "completed":
            continue
        output_rows.append(build_sft_row(row, args))

    write_rows(args.output_path, output_rows)
    print(f"[save] wrote {len(output_rows)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
