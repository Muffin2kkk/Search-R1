import glob
import json
import os
from typing import Optional

import datasets


REQUIRED_COLUMNS = {"id", "contents"}


def _default_datasets_cache() -> str:
    if "HF_DATASETS_CACHE" in os.environ:
        return os.environ["HF_DATASETS_CACHE"]
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(hf_home, "datasets")


def _list_arrow_files(arrow_dir: str):
    return sorted(glob.glob(os.path.join(arrow_dir, "*.arrow")))


def _is_arrow_dir(path: Optional[str]) -> bool:
    return bool(path) and os.path.isdir(path) and bool(_list_arrow_files(path))


def _load_arrow_corpus(arrow_dir: str) -> datasets.Dataset:
    arrow_files = _list_arrow_files(arrow_dir)
    if not arrow_files:
        raise FileNotFoundError(f"No Arrow shards found under {arrow_dir}.")

    shards = [datasets.Dataset.from_file(path) for path in arrow_files]
    corpus = shards[0] if len(shards) == 1 else datasets.concatenate_datasets(shards)

    missing_columns = REQUIRED_COLUMNS.difference(corpus.column_names)
    if missing_columns:
        raise ValueError(
            f"Arrow corpus at {arrow_dir} is missing required columns: {sorted(missing_columns)}"
        )

    return corpus


def _find_arrow_dir_from_cache(corpus_path: str, cache_dir: Optional[str]) -> Optional[str]:
    if not corpus_path:
        return None

    cache_root = cache_dir or _default_datasets_cache()
    info_paths = glob.glob(os.path.join(cache_root, "json", "*", "*", "*", "dataset_info.json"))

    for info_path in sorted(info_paths):
        try:
            with open(info_path, "r") as f:
                dataset_info = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        download_checksums = dataset_info.get("download_checksums", {})
        if corpus_path not in download_checksums:
            continue

        arrow_dir = os.path.dirname(info_path)
        if _is_arrow_dir(arrow_dir):
            return arrow_dir

    return None


def resolve_arrow_dir(
    corpus_path: Optional[str] = None,
    arrow_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Optional[str]:
    if _is_arrow_dir(arrow_dir):
        return arrow_dir

    env_arrow_dir = os.environ.get("SEARCH_R1_ARROW_CORPUS_DIR")
    if _is_arrow_dir(env_arrow_dir):
        return env_arrow_dir

    if _is_arrow_dir(corpus_path):
        return corpus_path

    if corpus_path:
        return _find_arrow_dir_from_cache(corpus_path, cache_dir)

    return None


def load_corpus(
    corpus_path: Optional[str] = None,
    arrow_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    allow_json_fallback: bool = False,
) -> datasets.Dataset:
    resolved_arrow_dir = resolve_arrow_dir(
        corpus_path=corpus_path,
        arrow_dir=arrow_dir,
        cache_dir=cache_dir,
    )
    if resolved_arrow_dir:
        print(f"Loading corpus from Arrow shards: {resolved_arrow_dir}")
        return _load_arrow_corpus(resolved_arrow_dir)

    if allow_json_fallback and corpus_path:
        print(f"Arrow corpus not found, falling back to JSONL: {corpus_path}")
        return datasets.load_dataset(
            "json",
            data_files=corpus_path,
            split="train",
            num_proc=4,
            cache_dir=cache_dir,
        )

    raise FileNotFoundError(
        "Unable to locate an Arrow corpus. "
        "Provide --arrow_corpus_dir, set SEARCH_R1_ARROW_CORPUS_DIR, "
        "or keep corpus_path identical to the path used when the Arrow cache was built."
    )
