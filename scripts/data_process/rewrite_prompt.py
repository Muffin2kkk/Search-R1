import argparse
from pathlib import Path

from datasets import load_dataset


PARALLEL_PROMPT_TEMPLATE = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You should try to resolve the question in as few search turns as possible. \
If the question can be answered with one focused query, prefer a single <search>. \
If multiple independent pieces of information need to be looked up simultaneously, you may issue parallel searches in one turn by wrapping them together: <searches> <search> query1 </search> <search> query2 </search> </searches>. \
Use parallel searches only when the sub-queries are complementary and do not depend on each other's results. \
Do not issue parallel searches that are merely different rephrasings of the same information need. \
If the next query depends on the result of the current query, use sequential search instead. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

QUESTION_MARKER = "Question: "


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rewrite prompts inside Search-R1 parquet files."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing train.parquet and test.parquet.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
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
