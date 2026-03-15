import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from datasets import Dataset, load_dataset

try:
    from .strong_model_client import StrongModelClient, is_likely_balance_error
    from .strong_rollout_collector import CollectorConfig, StrongRolloutCollector, dumps_jsonl_row
except ImportError:
    from strong_model_client import StrongModelClient, is_likely_balance_error
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
        help="API key for the remote LLM. If omitted, use the hardcoded default in the client.",
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
        default=0.3,
        help="Sampling temperature. Lower values are more stable for distillation.",
    )
    parser.add_argument(
        "--llm_top_p",
        type=float,
        default=0.8,
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
        help="Optional JSON string merged into the raw LLM request body. Thinking mode is disabled in the client.",
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


def build_sample_id(row: Dict[str, Any], split: str, idx: int) -> str:
    extra_info = row.get("extra_info", {}) if isinstance(row, dict) else {}
    source_idx = extra_info.get("index", idx)
    return f"{split}-{source_idx}"


def load_existing_jsonl(path: Path, split: str) -> Tuple[List[Dict[str, Any]], Set[str], int]:
    if not path.exists():
        return [], set(), 0

    rows: List[Dict[str, Any]] = []
    sample_ids: Set[str] = set()
    skipped_error_rows = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] Skipping invalid JSONL line {line_no} in {path}")
                continue
            if row.get("collector_error"):
                skipped_error_rows += 1
                continue
            rows.append(row)
            sample_id = row.get("sample_id")
            if not sample_id:
                source_idx = row.get("extra_info", {}).get("index") if isinstance(row.get("extra_info"), dict) else None
                if source_idx is not None:
                    sample_id = f"{split}-{source_idx}"
            if sample_id:
                sample_ids.add(str(sample_id))
    return rows, sample_ids, skipped_error_rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(dumps_jsonl_row(row))
            f.write("\n")


def append_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(dumps_jsonl_row(row))
            f.write("\n")
        f.flush()


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

    output_jsonl = args.output_dir / f"{split}.jsonl"
    collected_rows, completed_sample_ids, skipped_error_rows = load_existing_jsonl(output_jsonl, split=split)
    if skipped_error_rows:
        write_jsonl(output_jsonl, collected_rows)
        print(f"[resume] removed {skipped_error_rows} errored rows from {output_jsonl} before retrying")
    if completed_sample_ids:
        print(f"[resume] found {len(completed_sample_ids)} completed rows in {output_jsonl}")
    else:
        write_jsonl(output_jsonl, [])

    pending_items: List[Tuple[int, Dict[str, Any]]] = []
    for row_offset, row in enumerate(rows):
        row_idx = args.offset + row_offset
        sample_id = build_sample_id(row, split, row_idx)
        if sample_id in completed_sample_ids:
            continue
        pending_items.append((row_idx, row))

    if not pending_items:
        print(f"[resume] {split}: nothing left to process")
        if args.write_parquet:
            output_parquet = args.output_dir / f"{split}.parquet"
            write_parquet(output_parquet, collected_rows)
            print(f"[save] wrote {output_parquet}")
        return

    print(f"[resume] {split}: processing {len(pending_items)} remaining rows")
    for batch_idx, batch_items, total_batches in chunk_rows(pending_items, args.batch_size):
        row_indices = [row_idx for row_idx, _ in batch_items]
        batch_rows = [row for _, row in batch_items]
        print(
            f"[collect] split={split} batch={batch_idx + 1}/{total_batches} "
            f"size={len(batch_rows)}"
        )
        batch_results = collector.collect_batch(
            batch_rows,
            split=split,
            start_idx=row_indices[0],
            row_indices=row_indices,
        )
        collected_rows.extend(batch_results)
        append_jsonl(output_jsonl, batch_results)
        print(f"[save] appended {len(batch_results)} rows to {output_jsonl}")

        balance_errors = [
            row.get("collector_error")
            for row in batch_results
            if is_likely_balance_error(row.get("collector_error"))
        ]
        if balance_errors:
            print("[error] Detected possible LLM balance/quota issue. Saved completed rows before stopping.")
            print(f"[error] Example upstream error: {balance_errors[0]}")
            return

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
