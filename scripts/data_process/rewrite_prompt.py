import argparse
from pathlib import Path

from datasets import load_dataset


PARALLEL_PROMPT_TEMPLATE = """Answer the given question.

You must follow this format strictly:
<decision> brief action-oriented decision </decision>
<search> query </search>

or

<decision> brief action-oriented decision </decision>
<searches> <search> query1 </search> <search> query2 </search> </searches>

or

<decision> brief action-oriented decision </decision>
<answer> final answer </answer>

Rules:
- Before every action, you must output exactly one <decision> block.
- The <decision> block must be brief and action-oriented.
- Use the <decision> block only to state:
  1) what needs to be verified or concluded, and
  2) whether to use a single search, parallel searches, or answer directly.
- Do not include long reasoning, background knowledge, or detailed explanation in <decision>.
- Keep each <decision> short, usually one sentence and at most two sentences.

Search policy:
- For questions involving factual world knowledge, named entities, biographies, occupations, dates, places, organizations, events, records, rankings, relationships, or other external facts, verify the needed facts with search before answering, even if you believe you already know the answer.
- Do not answer factual verification questions directly from memory without first retrieving evidence.
- If one focused query is sufficient, prefer a single <search>.
- If multiple independent pieces of information need to be verified simultaneously, you may issue parallel searches in one turn by wrapping them together:
  <searches> <search> query1 </search> <search> query2 </search> </searches>
- Use parallel searches only when the sub-queries are complementary and do not depend on each other's results.
- Do not issue parallel searches that are merely different rephrasings of the same information need.
- If the next query depends on the result of the current query, use sequential search instead.
- Try to resolve the question in as few search turns as possible.

Answer policy:
- Only when the question can be answered without any external factual knowledge, such as pure reasoning or calculation, may you answer directly without search.
- Once enough evidence has been gathered, provide the final answer inside <answer> and </answer>, without detailed illustrations.

Use the <decision> block to briefly state:
1) what information needs to be verified or concluded, and
2) why the next action should be a single search, parallel searches, sequential search, or a direct answer.

Examples of good <decision> blocks:
<decision> Only one factual attribute needs to be verified, so use a single search. </decision>
<decision> The same attribute needs to be verified for multiple independent entities, so use parallel searches. </decision>
<decision> Multiple independent pieces of information need to be verified, so use parallel searches. </decision>
<decision> The next fact depends on the result of the first search, so use a single search sequentially. </decision>
<decision> The retrieved evidence is sufficient to answer the question, so answer now. </decision> Question: {question}\n"""

QUESTION_MARKER = "Question: "


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rewrite prompts inside Search-R1 parquet files."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/nq_hotpotqa_tiny"),
        help="Directory containing train.parquet and test.parquet.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/nq_hotpotqa_tiny/distillation"),
        help="Directory to save the rewritten dataset.",
    )
    return parser.parse_args()


def extract_question_from_prompt(prompt_text: str) -> str:
    idx = prompt_text.rfind(QUESTION_MARKER)
    if idx == -1:
        raise ValueError(
            f"Cannot find `{QUESTION_MARKER}` in prompt: {prompt_text[:200]!r}"
        )
    return prompt_text[idx + len(QUESTION_MARKER):].strip()


def make_parallel_prompt(question: str) -> str:
    question = question.strip()
    if question and not question.endswith("?"):
        question += "?"
    return PARALLEL_PROMPT_TEMPLATE.format(question=question)


def rewrite_example(example):
    prompt = example.get("prompt")
    if not isinstance(prompt, list) or not prompt or "content" not in prompt[0]:
        raise ValueError("Expected `prompt` to be a non-empty list with a `content` field.")

    question = extract_question_from_prompt(prompt[0]["content"])
    prompt[0]["content"] = make_parallel_prompt(question)
    example["prompt"] = prompt
    return example


def save_split(dataset, output_dir: Path, split: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_dir / f"{split}.parquet"))


def save_dataset_dict(dataset_dict, output_dir: Path):
    for split, dataset in dataset_dict.items():
        save_split(dataset, output_dir, split)


def load_parquet_split(path: Path):
    return load_dataset("parquet", data_files=str(path), split="train")


def main():
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    train_path = input_dir / "train.parquet"
    test_path = input_dir / "test.parquet"

    rewritten = {
        "train": load_parquet_split(train_path).map(rewrite_example),
        "test": load_parquet_split(test_path).map(rewrite_example),
    }

    save_dataset_dict(rewritten, output_dir)
    print(f"Saved rewritten dataset to {output_dir}")


if __name__ == "__main__":
    main()
