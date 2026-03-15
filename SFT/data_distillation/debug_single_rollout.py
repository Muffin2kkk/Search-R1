import argparse
import difflib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

try:
    from .collect_strong_rollouts import load_split_rows
    from .strong_model_client import StrongModelClient
    from .strong_rollout_collector import CollectorConfig, StrongRolloutCollector
except ImportError:
    from collect_strong_rollouts import load_split_rows
    from strong_model_client import StrongModelClient
    from strong_rollout_collector import CollectorConfig, StrongRolloutCollector


INFORMATION_RE = re.compile(r"<information>(.*?)</information>", re.DOTALL)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug one distillation rollout sample.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("/root/Search-R1/data/nq_hotpotqa_tiny/distillation"),
        help="Directory containing parquet splits.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to debug.")
    parser.add_argument("--offset", type=int, default=0, help="Row offset inside the split.")
    parser.add_argument("--search_url", type=str, default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--retriever_topk", type=int, default=3)
    parser.add_argument("--retrieval_timeout", type=int, default=60)
    parser.add_argument("--retrieval_cache_size", type=int, default=10000)
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/root/autodl-tmp/Qwen2.5-3B-Instruct",
        help="Tokenizer used for token-length debugging.",
    )
    parser.add_argument("--max_prompt_length", type=int, default=6144)
    parser.add_argument("--max_obs_length", type=int, default=500)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--llm_api_base", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--llm_api_path", type=str, default="/chat/completions")
    parser.add_argument("--llm_api_key", type=str, default=None)
    parser.add_argument("--llm_model", type=str, default="qwen3.5-plus")
    parser.add_argument("--llm_timeout", type=int, default=120)
    parser.add_argument("--llm_concurrency", type=int, default=1)
    parser.add_argument("--llm_temperature", type=float, default=0.3)
    parser.add_argument("--llm_top_p", type=float, default=0.8)
    parser.add_argument("--llm_max_tokens", type=int, default=512)
    parser.add_argument("--llm_presence_penalty", type=float, default=0.0)
    parser.add_argument("--llm_frequency_penalty", type=float, default=0.0)
    parser.add_argument("--llm_seed", type=int, default=None)
    parser.add_argument("--llm_extra_body", type=str, default='{"enable_thinking": false}')
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Optional path to save the full rollout json.",
    )
    return parser.parse_args()


def default_output_path(split: str, offset: int) -> Path:
    return Path(f"/root/Search-R1/debug/distillation_debug_{split}_{offset}.json")


def extract_information_messages(messages):
    extracted = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = str(message.get("content", ""))
        match = INFORMATION_RE.search(content)
        if match is not None:
            extracted.append(match.group(1).strip())
    return extracted


def build_observation_diff_report(result):
    retrieval_turns = [turn for turn in result.get("rollout_turns", []) if turn.get("type") == "retrieval"]
    info_messages = extract_information_messages(result.get("messages", []))
    lines = []
    if len(retrieval_turns) != len(info_messages):
        lines.append(
            f"[count-mismatch] retrieval_turns={len(retrieval_turns)} information_messages={len(info_messages)}"
        )

    compare_count = min(len(retrieval_turns), len(info_messages))
    all_match = True
    for idx in range(compare_count):
        rl_text = str(retrieval_turns[idx].get("text", "")).strip()
        distill_text = info_messages[idx].strip()
        obs_token_len = retrieval_turns[idx].get("obs_token_len")
        if rl_text == distill_text:
            lines.append(f"[match] turn_{idx + 1} obs_token_len={obs_token_len}")
            continue

        all_match = False
        lines.append(f"[mismatch] turn_{idx + 1} obs_token_len={obs_token_len}")
        diff_lines = list(
            difflib.unified_diff(
                rl_text.splitlines(),
                distill_text.splitlines(),
                fromfile=f"rl_shared_observation_turn_{idx + 1}",
                tofile=f"distillation_message_information_turn_{idx + 1}",
                lineterm="",
            )
        )
        lines.extend(diff_lines if diff_lines else ["[mismatch] diff unavailable"])

    if compare_count == 0:
        lines.append("[info] no retrieval turns found")
    elif all_match and len(retrieval_turns) == len(info_messages):
        lines.append("[summary] all observation texts match exactly")

    return "\n".join(lines).strip() + "\n"


def main():
    args = parse_args()
    split_path = args.input_dir / f"{args.split}.parquet"
    rows = load_split_rows(split_path, offset=args.offset, limit=1)
    if not rows:
        raise ValueError(f"No row found at split={args.split}, offset={args.offset}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    model_client = StrongModelClient.from_args(args)
    collector = StrongRolloutCollector(
        model_client=model_client,
        config=CollectorConfig(
            search_url=args.search_url,
            topk=args.retriever_topk,
            max_turns=args.max_turns,
            llm_concurrency=args.llm_concurrency,
            retrieval_timeout=args.retrieval_timeout,
            cache_size=args.retrieval_cache_size,
            tokenizer=tokenizer,
            max_prompt_length=args.max_prompt_length,
            max_obs_length=args.max_obs_length,
        ),
    )

    result = collector.collect_batch(rows, split=args.split, start_idx=args.offset, row_indices=[args.offset])[0]
    output_path = args.output_path or default_output_path(args.split, args.offset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    diff_path = output_path.parent / f"{output_path.stem}_obs_diff.txt"
    diff_report = build_observation_diff_report(result)
    diff_path.write_text(diff_report, encoding="utf-8")

    print(f"Saved full rollout to: {output_path}")
    print(f"Saved observation diff to: {diff_path}")
    print(f"sample_id: {result.get('sample_id')}")
    print(f"collector_status: {result.get('collector_status')}")
    print(f"collector_error: {result.get('collector_error')}")
    print(f"prompt_token_len: {result.get('prompt_token_len')}")

    retrieval_turns = [turn for turn in result.get("rollout_turns", []) if turn.get("type") == "retrieval"]
    if retrieval_turns:
        print("obs_token_len by retrieval turn:")
        for idx, turn in enumerate(retrieval_turns, start=1):
            print(f"  turn_{idx}: {turn.get('obs_token_len')}")
    else:
        print("obs_token_len by retrieval turn: none")
    print("\nObservation diff summary:")
    print(diff_report)


if __name__ == "__main__":
    main()
