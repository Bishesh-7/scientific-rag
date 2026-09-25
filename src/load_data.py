from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


# The normal QASPER loader did not work with my version of datasets because it
# uses an old Python loading script. These are the converted parquet files.
QASPER_REVISION = "ff28ef00406baa196597e25a0b2cb7b0a0e1f96d"
QASPER_BASE_URL = (
    "https://huggingface.co/datasets/allenai/qasper/resolve/"
    f"{QASPER_REVISION}/qasper"
)


def _load_cached_qasper():
    """Return a previously prepared QASPER cache, if all splits are present."""
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "parquet"
    for validation_file in cache_root.glob("*/*/*/parquet-validation.arrow"):
        folder = validation_file.parent
        split_files = {
            "train": folder / "parquet-train.arrow",
            "validation": validation_file,
            "test": folder / "parquet-test.arrow",
        }
        if all(path.exists() for path in split_files.values()):
            return DatasetDict(
                {name: Dataset.from_file(str(path)) for name, path in split_files.items()}
            )
    return None


def load_qasper(prefer_cache=True):
    """Load QASPER, reusing the local Arrow cache when it is available."""
    print("Loading QASPER dataset...")

    if prefer_cache:
        cached = _load_cached_qasper()
        if cached is not None:
            print("Using cached QASPER Arrow files.")
            return cached

    data_files = {
        "train": f"{QASPER_BASE_URL}/qasper-train.parquet",
        "validation": f"{QASPER_BASE_URL}/qasper-validation.parquet",
        "test": f"{QASPER_BASE_URL}/qasper-test.parquet",
    }

    return load_dataset("parquet", data_files=data_files)


if __name__ == "__main__":
    dataset = load_qasper()
    print(dataset)

    print("Training documents:", len(dataset["train"]))
    print("Validation documents:", len(dataset["validation"]))
    print("Test documents:", len(dataset["test"]))

    sample = dataset["train"][0]
    print("\nSample paper")
    print("=" * 80)
    print("ID:", sample["id"])
    print("Title:", sample["title"])
    print("Abstract:", sample["abstract"][:500])
