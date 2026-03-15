import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
from transformers import AutoTokenizer


SEARCHES_RE = re.compile(r"<searches>(.*?)</searches>", re.DOTALL | re.IGNORECASE)
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)

DEFAULT_PROMPT_PREFIX = (
    "Answer the given question. Before every action, you should briefly reason inside "
    "<decision> and </decision>. Use the <decision> block to briefly state what "
    "information needs to be verified or concluded, and why the next action should be "
    "a single search, parallel searches, sequential search, or a direct answer. Keep "
    "each <decision> short and action-oriented. If you need external knowledge, you can "
    "call a search engine by <search> query </search>, and it will return the top searched "
    "results between <information> and </information>. If multiple independent pieces of "
    "information need to be looked up simultaneously, you may issue parallel searches in "
    "one turn by wrapping them together: <searches> <search> query1 </search> <search> "
    "query2 </search> </searches>. Use parallel searches only when the sub-queries are "
    "complementary and do not depend on each other's results. Do not use parallel searches "
    "for different rephrasings of the same information need. Use a single search when one "
    "focused query is sufficient. If the next query depends on the result of the current "
    "query, use sequential search instead. You should try to resolve the question in as "
    "few search turns as possible. If no further external knowledge is needed, provide the "
    "final answer inside <answer> and </answer>, without detailed illustrations. For "
    "example, <answer> Beijing </answer>. Question: {question}\n"
)


@dataclass
class TurnResult:
    turn_idx: int
    action_text: str
    queries: List[str]
    info_text: str
    prompt_text: str
    char_len: int
    token_len: Optional[int]


SCENARIOS = {
    "single_then_parallel": {
        "question": "Are Bruno Soares and Serena Williams both tennis players?",
        "actions": [
            (
                "<decision>I need to verify Bruno Soares first, then compare with Serena "
                "Williams, so start with a single focused search.</decision>"
                "<search>Bruno Soares occupation tennis player</search>"
            ),
            (
                "<decision>I now need independent confirmation for Serena Williams and a "
                "direct comparison signal, so parallel searches are appropriate.</decision>"
                "<searches>"
                "<search>Serena Williams occupation tennis player</search>"
                "<search>Bruno Soares and Serena Williams tennis players</search>"
                "</searches>"
            ),
        ],
    },
    "parallel_people_check": {
        "question": "Are Bruno Soares and Serena Williams both tennis players?",
        "actions": [
            (
                "<decision>Both people can be verified independently, so use parallel "
                "searches in one turn.</decision>"
                "<searches>"
                "<search>Bruno Soares occupation tennis player</search>"
                "<search>Serena Williams occupation tennis player</search>"
                "<search>Bruno Soares biography tennis</search>"
                "</searches>"
            ),
        ],
    },
    "two_round_parallel": {
        "question": "Did Marie Curie and Pierre Curie both win Nobel Prizes, and in which fields?",
        "actions": [
            (
                "<decision>I need two independent biographies first, so parallel searches "
                "fit.</decision>"
                "<searches>"
                "<search>Marie Curie Nobel Prize field</search>"
                "<search>Pierre Curie Nobel Prize field</search>"
                "</searches>"
            ),
            (
                "<decision>I should cross-check the prize categories and years in parallel "
                "before answering.</decision>"
                "<searches>"
                "<search>Marie Curie Nobel Prize Physics Chemistry years</search>"
                "<search>Pierre Curie Nobel Prize Physics year</search>"
                "</searches>"
            ),
        ],
    },
    "three_round_parallel": {
        "question": "Were Bruno Soares, Serena Williams, and Roger Federer all professional tennis players, and what evidence supports that?",
        "actions": [
            (
                "<decision>I should verify the occupations of the three people independently, so parallel "
                "searches are the best first step.</decision>"
                "<searches>"
                "<search>Bruno Soares occupation professional tennis player</search>"
                "<search>Serena Williams occupation professional tennis player</search>"
                "<search>Roger Federer occupation professional tennis player</search>"
                "</searches>"
            ),
            (
                "<decision>I should gather supporting biography evidence for each person in parallel before "
                "forming a final conclusion.</decision>"
                "<searches>"
                "<search>Bruno Soares biography tennis doubles player</search>"
                "<search>Serena Williams biography tennis singles player</search>"
                "<search>Roger Federer biography tennis player</search>"
                "</searches>"
            ),
            (
                "<decision>I should retrieve concise career-summary evidence for all three in parallel to "
                "simulate a third search round.</decision>"
                "<searches>"
                "<search>Bruno Soares career tennis summary</search>"
                "<search>Serena Williams career tennis summary</search>"
                "<search>Roger Federer career tennis summary</search>"
                "</searches>"
            ),
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate single-turn or multi-turn search prompts and measure final prompt length."
    )
    parser.add_argument(
        "--search-url",
        default="http://127.0.0.1:8000/retrieve",
        help="Retrieval service endpoint.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path or name. Defaults to BASE_MODEL env-style path if provided manually.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Top-k retrieval results per search query.",
    )
    parser.add_argument(
        "--max-obs-length",
        type=int,
        default=500,
        help="Token budget for each individual search result before concatenation.",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS.keys()),
        default="single_then_parallel",
        help="Built-in simulation scenario.",
    )
    parser.add_argument(
        "--show-full-prompt",
        action="store_true",
        help="Print the full prompt after each turn.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save the per-turn prompt statistics as JSON.",
    )
    parser.add_argument(
        "--save-final-prompt",
        default=None,
        help="Optional path to save the final full prompt as a txt file. Defaults to debug/final_prompt_<scenario>.txt",
    )
    return parser.parse_args()


def default_output_path(filename: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)


def truncate_by_tokens(tokenizer, text: str, max_tokens: int) -> str:
    input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if input_ids.shape[0] <= max_tokens:
        return text.strip()
    return tokenizer.decode(input_ids[:max_tokens], skip_special_tokens=True).strip()


def passages_to_string(retrieval_result: List[dict]) -> str:
    lines = []
    for idx, doc_item in enumerate(retrieval_result, start=1):
        content = doc_item.get("document", {}).get("contents", "").strip()
        if not content:
            continue
        title = content.split("\n")[0].strip()
        text = "\n".join(content.split("\n")[1:]).strip()
        lines.append(f"Doc {idx}(Title: {title}) {text}".strip())
    return "\n".join(lines).strip()


def extract_queries(action_text: str) -> List[str]:
    searches_match = SEARCHES_RE.search(action_text)
    if searches_match is not None:
        return [query.strip() for query in SEARCH_RE.findall(searches_match.group(1)) if query.strip()]

    single_queries = [query.strip() for query in SEARCH_RE.findall(action_text) if query.strip()]
    return single_queries[:1]


def retrieve_queries(
    search_url: str,
    queries: List[str],
    topk: int,
) -> List[List[dict]]:
    if not queries:
        return []

    response = requests.post(
        search_url,
        json={"queries": queries, "topk": topk, "return_scores": True},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["result"]


def format_information_block(
    tokenizer,
    queries: List[str],
    retrieval_results: List[List[dict]],
    max_obs_length: int,
) -> str:
    if tokenizer is None:
        formatted_results = [passages_to_string(result).strip() for result in retrieval_results]
    else:
        formatted_results = [
            truncate_by_tokens(tokenizer, passages_to_string(result), max_obs_length)
            for result in retrieval_results
        ]

    if len(queries) == 1:
        return formatted_results[0]

    sections = []
    for idx, (query, result_text) in enumerate(zip(queries, formatted_results), start=1):
        sections.append(f"[Search {idx}] Query: {query}\n{result_text}".strip())
    return "\n\n".join(sections).strip()


def build_prompt_prefix(question: str) -> str:
    return DEFAULT_PROMPT_PREFIX.format(question=question)


def simulate_scenario(
    tokenizer,
    search_url: str,
    topk: int,
    max_obs_length: int,
    question: str,
    actions: List[str],
) -> List[TurnResult]:
    current_prompt = build_prompt_prefix(question)
    turn_results: List[TurnResult] = []

    for turn_idx, action_text in enumerate(actions, start=1):
        queries = extract_queries(action_text)
        retrieval_results = retrieve_queries(search_url, queries, topk) if queries else []
        info_text = format_information_block(tokenizer, queries, retrieval_results, max_obs_length) if queries else ""

        current_prompt += action_text
        if info_text:
            current_prompt += f"\n\n<information>{info_text}</information>\n\n"

        token_len = None
        if tokenizer is not None:
            token_len = tokenizer(current_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
        turn_results.append(
            TurnResult(
                turn_idx=turn_idx,
                action_text=action_text,
                queries=queries,
                info_text=info_text,
                prompt_text=current_prompt,
                char_len=len(current_prompt),
                token_len=token_len,
            )
        )

    return turn_results


def main() -> None:
    args = parse_args()
    scenario = SCENARIOS[args.scenario]

    tokenizer = None
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    turn_results = simulate_scenario(
        tokenizer=tokenizer,
        search_url=args.search_url,
        topk=args.topk,
        max_obs_length=args.max_obs_length,
        question=scenario["question"],
        actions=scenario["actions"],
    )

    print(f"Scenario: {args.scenario}")
    print(f"Question: {scenario['question']}")
    print(f"Search URL: {args.search_url}")
    print(f"Top-k: {args.topk}")
    if tokenizer is None:
        print("Tokenizer: not provided, using raw retrieval text without token truncation")
    else:
        print(f"Tokenizer: {args.tokenizer}")
        print(f"Per-search max observation tokens: {args.max_obs_length}")
    print()

    serializable = []
    for item in turn_results:
        print(f"=== Turn {item.turn_idx} ===")
        print("Action:")
        print(item.action_text)
        print()
        print(f"Queries ({len(item.queries)}): {item.queries}")
        print(f"Prompt chars: {item.char_len}")
        if item.token_len is not None:
            print(f"Prompt tokens: {item.token_len}")
        if item.info_text:
            print(f"Information chars: {len(item.info_text)}")
            if tokenizer is not None:
                info_tokens = tokenizer(item.info_text, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
                print(f"Information tokens: {info_tokens}")
        if args.show_full_prompt:
            print()
            print("Full prompt:")
            print(item.prompt_text)
        print()

        serializable.append(
            {
                "turn_idx": item.turn_idx,
                "action_text": item.action_text,
                "queries": item.queries,
                "info_text": item.info_text,
                "prompt_text": item.prompt_text,
                "char_len": item.char_len,
                "token_len": item.token_len,
            }
        )

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scenario": args.scenario,
                    "question": scenario["question"],
                    "search_url": args.search_url,
                    "topk": args.topk,
                    "max_obs_length": args.max_obs_length,
                    "turns": serializable,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved JSON report to {args.save_json}")

    final_prompt_path = args.save_final_prompt or default_output_path(f"final_prompt_{args.scenario}.txt")
    if turn_results:
        with open(final_prompt_path, "w", encoding="utf-8") as f:
            f.write(turn_results[-1].prompt_text)
        print(f"Saved final prompt to {final_prompt_path}")


if __name__ == "__main__":
    main()
