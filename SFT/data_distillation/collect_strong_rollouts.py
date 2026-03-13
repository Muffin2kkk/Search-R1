import argparse
import math
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, load_dataset

try:
    from .strong_model_client import StrongModelClient
    from .strong_rollout_collector import CollectorConfig, StrongRolloutCollector, dumps_jsonl_row
except ImportError:
    from strong_model_client import StrongModelClient
    from strong_rollout_collector import CollectorConfig, StrongRolloutCollector, dumps_jsonl_row


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect strong-model search rollouts for SFT without using verl."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing parquet splits such as train.parquet and test.parquet.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save collected rollout datasets.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,test",
        help="Comma-separated dataset splits to process.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to collect in one rollout batch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows per split.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Optional row offset per split.",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=2,
        help="Maximum number of search turns before forcing a final answer.",
    )
    parser.add_argument(
        "--search_url",
        type=str,
        default="http://127.0.0.1:8000/retrieve",
        help="Local retrieval server URL.",
    )
    parser.add_argument(
        "--retriever_topk",
        type=int,
        default=3,
        help="Number of passages returned for each search query.",
    )
    parser.add_argument(
        "--retrieval_timeout",
        type=int,
        default=60,
        help="Timeout in seconds for the local retrieval server.",
    )
    parser.add_argument(
        "--retrieval_cache_size",
        type=int,
        default=10000,
        help="Simple LRU cache size for repeated retrieval queries.",
    )
    parser.add_argument(
        "--llm_api_base",
        type=str,
        required=True,
        help="Base URL of the OpenAI-compatible LLM API.",
    )
    parser.add_argument(
        "--llm_api_path",
        type=str,
        default="/chat/completions",
        help="Path appended to llm_api_base.",
    )
    parser.add_argument(
        "--llm_api_key",
        type=str,
        default=None,
        help="API key for the remote LLM. If omitted, read LLM_API_KEY from environment.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        required=True,
        help="Remote strong-model name.",
    )
    parser.add_argument(
        "--llm_timeout",
        type=int,
        default=120,
        help="Timeout in seconds for each LLM request.",
    )
    parser.add_argument(
        "--llm_concurrency",
        type=int,
        default=8,
        help="Number of concurrent LLM requests inside one batch.",
    )
    parser.add_argument(
        "--llm_temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--llm_top_p",
        type=float,
        default=0.95,
        help="Top-p for sampling.",
    )
    parser.add_argument(
        "--llm_max_tokens",
        type=int,
        default=512,
        help="Max new tokens for one model turn.",
    )
    parser.add_argument(
        "--llm_presence_penalty",
        type=float,
        default=0.0,
        help="Presence penalty passed to the remote API.",
    )
    parser.add_argument(
        "--llm_frequency_penalty",
        type=float,
        default=0.0,
        help="Frequency penalty passed to the remote API.",
    )
    parser.add_argument(
        "--llm_seed",
        type=int,
        default=None,
        help="Optional sampling seed.",
    )
    parser.add_argument(
        "--llm_extra_body",
        type=str,
        default=None,
        help="Optional JSON string merged into the raw LLM request body.",
    )
    parser.add_argument(
        "--write_parquet",
        action="store_true",
        help="Also save a parquet copy after jsonl is written.",
    )
    return parser.parse_args()


def load_split_rows(split_path: Path, offset: int = 0, limit: int = None) -> List[Dict]:
    dataset = load_dataset("parquet", data_files=str(split_path), split="train")
    total_rows = len(dataset)
    start = min(offset, total_rows)
    end = total_rows if limit is None else min(total_rows, start + limit)
    if start or end != total_rows:
        dataset = dataset.select(range(start, end))
    return list(dataset)


def chunk_rows(rows: List[Dict], batch_size: int):
    total_batches = math.ceil(len(rows) / batch_size) if rows else 0
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        yield batch_idx, rows[start:end], total_batches


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(dumps_jsonl_row(row))
            f.write("\n")


def write_parquet(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(path))


def process_split(split: str, args, collector: StrongRolloutCollector) -> None:
    input_path = args.input_dir / f"{split}.parquet"
    if not input_path.exists():
        print(f"[skip] Missing split file: {input_path}")
        return

    print(f"[load] {split}: {input_path}")
    rows = load_split_rows(input_path, offset=args.offset, limit=args.limit)
    print(f"[info] {split}: loaded {len(rows)} rows")

    collected_rows: List[Dict] = []
    for batch_idx, batch_rows, total_batches in chunk_rows(rows, args.batch_size):
        batch_start = batch_idx * args.batch_size + args.offset
        print(
            f"[collect] split={split} batch={batch_idx + 1}/{total_batches} "
            f"size={len(batch_rows)}"
        )
        collected_rows.extend(
            collector.collect_batch(batch_rows, split=split, start_idx=batch_start)
        )

    output_jsonl = args.output_dir / f"{split}.jsonl"
    write_jsonl(output_jsonl, collected_rows)
    print(f"[save] wrote {output_jsonl}")

    if args.write_parquet:
        output_parquet = args.output_dir / f"{split}.parquet"
        write_parquet(output_parquet, collected_rows)
        print(f"[save] wrote {output_parquet}")


def main():
    args = parse_args()

    model_client = StrongModelClient.from_args(args)
    collector_config = CollectorConfig(
        search_url=args.search_url,
        topk=args.retriever_topk,
        max_turns=args.max_turns,
        llm_concurrency=args.llm_concurrency,
        retrieval_timeout=args.retrieval_timeout,
        cache_size=args.retrieval_cache_size,
    )
    collector = StrongRolloutCollector(model_client=model_client, config=collector_config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    for split in splits:
        process_split(split, args, collector)


if __name__ == "__main__":
    main()
