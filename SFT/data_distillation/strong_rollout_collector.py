import copy
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from .strong_model_client import GenerationRequest, GenerationResult, StrongModelClient
except ImportError:
    from strong_model_client import GenerationRequest, GenerationResult, StrongModelClient

from search_r1.llm_agent.search_observation_utils import (
    format_search_observation,
    passages_to_string,
    truncate_search_result,
)


INVALID_ACTION_OBSERVATION = (
    "My previous action is invalid. If I want to search, I should put the query "
    "between <search> and </search>. If I want to launch parallel searches, I should "
    "put them inside <searches> ... </searches> with one or more <search> blocks. "
    "If I want to give the final answer, I should put the answer between <answer> "
    "and </answer>. Let me try again."
)

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
SEARCHES_RE = re.compile(r"<searches>(.*?)</searches>", re.IGNORECASE | re.DOTALL)
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)


@dataclass
class CollectorConfig:
    search_url: str
    topk: int = 3
    max_turns: int = 3
    llm_concurrency: int = 8
    retrieval_timeout: int = 60
    cache_size: int = 10000
    tokenizer: Any = None
    max_prompt_length: int = 6144
    max_obs_length: int = 500
    final_answer_prompt: str = (
        "You have reached the maximum number of search turns. Based on the available "
        "information, provide your final answer in <answer> and </answer>."
    )


@dataclass
class ParsedAction:
    action: str
    answer: str = ""
    queries: List[str] = field(default_factory=list)


@dataclass
class SampleState:
    sample_id: str
    source_row: Dict[str, Any]
    messages: List[Dict[str, str]]
    status: str = "running"
    final_answer: str = ""
    final_response: str = ""
    error: Optional[str] = None
    rollout_turns: List[Dict[str, Any]] = field(default_factory=list)
    search_turn_count: int = 0
    invalid_action_count: int = 0
    usage: List[Dict[str, Any]] = field(default_factory=list)
    prompt_token_len: Optional[int] = None


def clone_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"role": str(item["role"]), "content": str(item["content"])} for item in copy.deepcopy(list(messages))]


def build_info_message(content: str) -> Dict[str, str]:
    return {"role": "user", "content": f"\n\n<information>{content}</information>\n\n"}


def build_invalid_message() -> Dict[str, str]:
    return {"role": "user", "content": f"\n{INVALID_ACTION_OBSERVATION}\n"}


def _strip_searches_block(text: str) -> str:
    return SEARCHES_RE.sub("", text)


def parse_model_action(text: str) -> ParsedAction:
    searches_match = SEARCHES_RE.search(text)
    answer_match = ANSWER_RE.search(text)
    stripped_text = _strip_searches_block(text)
    single_search_match = SEARCH_RE.search(stripped_text)

    candidates: List[Tuple[int, str, Any]] = []
    if searches_match is not None:
        candidates.append((searches_match.start(), "searches", searches_match))
    if single_search_match is not None:
        candidates.append((single_search_match.start(), "search", single_search_match))
    if answer_match is not None:
        candidates.append((answer_match.start(), "answer", answer_match))

    if not candidates:
        return ParsedAction(action="invalid")

    _, action_type, match = min(candidates, key=lambda item: item[0])

    if action_type == "answer":
        return ParsedAction(action="answer", answer=match.group(1).strip())

    if action_type == "searches":
        queries = [item.strip() for item in SEARCH_RE.findall(match.group(1)) if item.strip()]
        if queries:
            return ParsedAction(action="search", queries=queries)
        return ParsedAction(action="invalid")

    query = match.group(1).strip()
    if query:
        return ParsedAction(action="search", queries=[query])
    return ParsedAction(action="invalid")


class RetrievalClient:
    def __init__(self, url: str, topk: int, timeout: int, cache_size: int):
        self.url = url
        self.topk = topk
        self.timeout = timeout
        self.cache_size = max(0, cache_size)
        self.cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()

    def _remember(self, query: str, result: List[Dict[str, Any]]) -> None:
        if self.cache_size <= 0:
            return
        self.cache[query] = result
        self.cache.move_to_end(query)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def get_many(self, queries: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        normalized_queries = [query.strip() for query in queries if query and query.strip()]
        unique_queries = list(dict.fromkeys(normalized_queries))
        results_by_query: Dict[str, List[Dict[str, Any]]] = {}
        uncached_queries: List[str] = []

        for query in unique_queries:
            if query in self.cache:
                self.cache.move_to_end(query)
                results_by_query[query] = self.cache[query]
            else:
                uncached_queries.append(query)

        if uncached_queries:
            payload = {
                "queries": uncached_queries,
                "topk": self.topk,
                "return_scores": True,
            }
            response = requests.post(self.url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            response_data = response.json()
            raw_results = response_data["result"]
            if len(raw_results) != len(uncached_queries):
                raise ValueError("Retrieval result count does not match query count.")

            for query, result in zip(uncached_queries, raw_results):
                results_by_query[query] = result
                self._remember(query, result)

        return results_by_query


class StrongRolloutCollector:
    def __init__(self, model_client: StrongModelClient, config: CollectorConfig):
        self.model_client = model_client
        self.config = config
        self.retrieval_client = RetrievalClient(
            url=config.search_url,
            topk=config.topk,
            timeout=config.retrieval_timeout,
            cache_size=config.cache_size,
        )

    def collect_batch(
        self,
        rows: List[Dict[str, Any]],
        split: str,
        start_idx: int = 0,
        row_indices: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        effective_indices = row_indices or [start_idx + idx for idx, _ in enumerate(rows)]
        states = [
            self._build_state(row=row, split=split, idx=row_idx)
            for row, row_idx in zip(rows, effective_indices)
        ]

        for _ in range(self.config.max_turns):
            active_states = [state for state in states if state.status == "running"]
            if not active_states:
                break
            self._run_generation_step(active_states)

        final_active_states = [state for state in states if state.status == "running"]
        if final_active_states:
            self._force_final_answer(final_active_states)

        return [self._serialize_state(state) for state in states]

    def _build_state(self, row: Dict[str, Any], split: str, idx: int) -> SampleState:
        prompt_messages = clone_messages(row["prompt"])
        sample_id = self._get_sample_id(row=row, split=split, idx=idx)
        return SampleState(
            sample_id=sample_id,
            source_row=copy.deepcopy(row),
            messages=self._truncate_messages_to_budget(prompt_messages),
            prompt_token_len=self._count_message_tokens(self._truncate_messages_to_budget(prompt_messages)),
        )

    def _get_sample_id(self, row: Dict[str, Any], split: str, idx: int) -> str:
        extra_info = row.get("extra_info", {}) if isinstance(row, dict) else {}
        source_idx = extra_info.get("index", idx)
        return f"{split}-{source_idx}"

    def _run_generation_step(self, active_states: List[SampleState]) -> None:
        for state in active_states:
            state.messages = self._truncate_messages_to_budget(state.messages)
            state.prompt_token_len = self._merge_token_len(
                state.prompt_token_len,
                self._count_message_tokens(state.messages),
            )
        requests_list = [
            GenerationRequest(sample_id=state.sample_id, messages=state.messages)
            for state in active_states
        ]
        generation_results = self.model_client.generate_batch(
            requests_list,
            max_concurrency=self.config.llm_concurrency,
        )
        result_by_sample_id = {result.sample_id: result for result in generation_results}

        search_jobs: List[Tuple[SampleState, List[str]]] = []
        for state in active_states:
            result = result_by_sample_id[state.sample_id]
            self._record_generation_result(state, result)
            if result.error:
                state.status = "failed"
                state.error = result.error
                continue

            parsed_action = parse_model_action(result.text)
            if parsed_action.action == "answer":
                state.status = "completed"
                state.final_answer = parsed_action.answer
                state.final_response = result.text
                continue

            if parsed_action.action == "search":
                search_jobs.append((state, parsed_action.queries))
                continue

            state.invalid_action_count += 1
            state.rollout_turns.append(
                {
                    "type": "invalid_action_feedback",
                    "text": INVALID_ACTION_OBSERVATION,
                }
            )
            state.messages.append(build_invalid_message())

        if search_jobs:
            self._handle_search_jobs(search_jobs)

    def _record_generation_result(self, state: SampleState, result: GenerationResult) -> None:
        if result.usage:
            state.usage.append(result.usage)

        state.rollout_turns.append(
            {
                "type": "assistant",
                "text": result.text,
                "finish_reason": result.finish_reason,
                "error": result.error,
            }
        )

        if result.text:
            state.messages.append({"role": "assistant", "content": result.text})

    def _handle_search_jobs(self, search_jobs: List[Tuple[SampleState, List[str]]]) -> None:
        all_queries = [query for _, queries in search_jobs for query in queries]
        results_by_query = self.retrieval_client.get_many(all_queries)

        for state, queries in search_jobs:
            info_text, obs_token_len = self._format_parallel_search_observation(queries, results_by_query)
            state.search_turn_count += 1
            state.rollout_turns.append(
                {
                    "type": "retrieval",
                    "queries": queries,
                    "text": info_text,
                    "obs_token_len": obs_token_len,
                }
            )
            state.messages.append(build_info_message(info_text))

    def _force_final_answer(self, active_states: List[SampleState]) -> None:
        for state in active_states:
            state.messages.append({"role": "user", "content": self.config.final_answer_prompt})
            state.messages = self._truncate_messages_to_budget(state.messages)
            state.prompt_token_len = self._merge_token_len(
                state.prompt_token_len,
                self._count_message_tokens(state.messages),
            )

        requests_list = [
            GenerationRequest(sample_id=state.sample_id, messages=state.messages)
            for state in active_states
        ]
        generation_results = self.model_client.generate_batch(
            requests_list,
            max_concurrency=self.config.llm_concurrency,
        )

        for result in generation_results:
            state = next(item for item in active_states if item.sample_id == result.sample_id)
            self._record_generation_result(state, result)
            if result.error:
                state.status = "failed"
                state.error = result.error
                continue

            parsed_action = parse_model_action(result.text)
            if parsed_action.action == "answer":
                state.status = "completed"
                state.final_answer = parsed_action.answer
                state.final_response = result.text
            else:
                state.status = "failed"
                state.error = "Final answer generation did not return an <answer> block."

    def _format_parallel_search_observation(
        self,
        queries: List[str],
        results_by_query: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[str, List[Optional[int]]]:
        formatted_results = [
            truncate_search_result(
                self.config.tokenizer,
                passages_to_string(results_by_query.get(query, [])),
                self.config.max_obs_length,
            )
            for query in queries
        ]
        obs_token_len = [self._count_text_tokens(result_text) for result_text in formatted_results]
        return format_search_observation(queries, formatted_results), obs_token_len

    def _count_text_tokens(self, text: str) -> Optional[int]:
        tokenizer = self.config.tokenizer
        if tokenizer is None:
            return None
        return int(
            tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].shape[0]
        )

    def _render_messages(self, messages: List[Dict[str, str]]) -> List[int]:
        tokenizer = self.config.tokenizer
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )

        fallback_text = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages
        )
        return tokenizer(fallback_text, add_special_tokens=False)["input_ids"] if tokenizer is not None else []

    def _truncate_messages_to_budget(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        tokenizer = self.config.tokenizer
        if tokenizer is None or not messages:
            return messages

        truncated_messages = [copy.deepcopy(message) for message in messages]
        max_prompt_length = self.config.max_prompt_length
        if len(self._render_messages(truncated_messages)) <= max_prompt_length:
            return truncated_messages

        while len(truncated_messages) > 1 and len(self._render_messages(truncated_messages)) > max_prompt_length:
            truncated_messages.pop(0)

        if len(self._render_messages(truncated_messages)) <= max_prompt_length:
            return truncated_messages

        first_message = copy.deepcopy(truncated_messages[0])
        content_ids = tokenizer(
            first_message.get("content", ""),
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]

        low, high = 0, content_ids.shape[0]
        best_content = ""
        while low <= high:
            mid = (low + high) // 2
            candidate_content = tokenizer.decode(content_ids[-mid:], skip_special_tokens=True).strip() if mid > 0 else ""
            trial_messages = [copy.deepcopy(message) for message in truncated_messages]
            trial_messages[0]["content"] = candidate_content
            if len(self._render_messages(trial_messages)) <= max_prompt_length:
                best_content = candidate_content
                low = mid + 1
            else:
                high = mid - 1

        truncated_messages[0]["content"] = best_content
        while len(truncated_messages) > 1 and not truncated_messages[0].get("content", "").strip():
            truncated_messages.pop(0)
        return truncated_messages

    def _count_message_tokens(self, messages: List[Dict[str, str]]) -> Optional[int]:
        tokenizer = self.config.tokenizer
        if tokenizer is None:
            return None
        return len(self._render_messages(messages))

    def _merge_token_len(self, current: Optional[int], candidate: Optional[int]) -> Optional[int]:
        if candidate is None:
            return current
        if current is None:
            return candidate
        return max(current, candidate)

    def _serialize_state(self, state: SampleState) -> Dict[str, Any]:
        row = copy.deepcopy(state.source_row)
        row["sample_id"] = state.sample_id
        row["messages"] = state.messages
        row["rollout_turns"] = state.rollout_turns
        row["final_answer"] = state.final_answer
        row["final_response"] = state.final_response
        row["collector_status"] = state.status
        row["collector_error"] = state.error
        row["search_turn_count"] = state.search_turn_count
        row["invalid_action_count"] = state.invalid_action_count
        row["strong_model_usage"] = state.usage
        row["prompt_token_len"] = state.prompt_token_len
        return row


def dumps_jsonl_row(row: Dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False)
