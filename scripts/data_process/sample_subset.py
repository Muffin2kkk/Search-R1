import argparse
from pathlib import Path

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly sample a smaller train/test subset from parquet files."
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
        help="Directory to save the sampled subset.",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=10000,
        help="Number of samples to keep in train.parquet.",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=2000,
        help="Number of samples to keep in test.parquet.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling before sampling.",
    )
    return parser.parse_args()


def save_split(dataset, output_dir: Path, split: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_dir / f"{split}.parquet"))


def load_parquet_split(path: Path):
    return load_dataset("parquet", data_files=str(path), split="train")


def main():
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    train_dataset = load_parquet_split(input_dir / "train.parquet")
    test_dataset = load_parquet_split(input_dir / "test.parquet")

    if args.train_size > len(train_dataset):
        raise ValueError(
            f"Requested train_size={args.train_size}, but only {len(train_dataset)} samples exist."
        )
    if args.test_size > len(test_dataset):
        raise ValueError(
            f"Requested test_size={args.test_size}, but only {len(test_dataset)} samples exist."
        )

    sampled_train = train_dataset.shuffle(seed=args.seed).select(range(args.train_size))
    sampled_test = test_dataset.shuffle(seed=args.seed).select(range(args.test_size))

    save_split(sampled_train, output_dir, "train")
    save_split(sampled_test, output_dir, "test")

    print(
        f"Saved sampled subset to {output_dir} "
        f"(train={args.train_size}, test={args.test_size}, seed={args.seed})"
    )


if __name__ == "__main__":
    main()
