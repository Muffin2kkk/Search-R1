from typing import Any, Dict, List


def passages_to_string(retrieval_result: List[Dict[str, Any]]) -> str:
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = str(doc_item["document"]["contents"])
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"
    return format_reference


def truncate_search_result(tokenizer, result: str, max_obs_length: int) -> str:
    result_ids = tokenizer(
        result,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    if result_ids.shape[0] <= max_obs_length:
        return result.strip()

    truncated_ids = result_ids[:max_obs_length]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True).strip()


def format_search_observation(queries: List[str], query_results: List[str]) -> str:
    if len(queries) == 1:
        return query_results[0].strip()

    sections = []
    for idx, (query, result) in enumerate(zip(queries, query_results), start=1):
        section = f"[Search {idx}] Query: {query}\n{result.strip()}".strip()
        sections.append(section)

    return "\n\n".join(sections).strip()
