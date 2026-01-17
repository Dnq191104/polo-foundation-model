import argparse
import os
import sys
from typing import List, Optional

from datasets import Dataset, DatasetDict, Image, load_dataset

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.attribute_extractor import AttributeExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a Parquet dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/data-00000-of-00001.parquet",
        help="Path to input parquet file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed output",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        choices=["hf", "parquet", "csv"],
        default="hf",
        help="Save as Hugging Face dataset (hf) or tabular (parquet/csv)",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the text column",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="Name of the image column if present",
    )
    parser.add_argument(
        "--min_text_len",
        type=int,
        default=1,
        help="Minimum length of text after stripping",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (0 disables split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting/shuffling",
    )
    parser.add_argument(
        "--extract_attributes",
        action="store_true",
        help="Extract fashion attributes (material, pattern, neckline, sleeve) from text",
    )
    parser.add_argument(
        "--schema_path",
        type=str,
        default=None,
        help="Path to attribute schema YAML (default: auto-detect)",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_parquet(path: str) -> Dataset:
    ds = load_dataset("parquet", data_files=path, split="train")
    # If the image column exists but is not Image type, try casting to Image for consistency
    if "image" in ds.column_names and not isinstance(ds.features["image"], Image):
        try:
            ds = ds.cast_column("image", Image())
        except Exception:
            pass
    return ds


def clean_and_filter_text(ds: Dataset, text_column: str, min_text_len: int) -> Dataset:
    if text_column not in ds.column_names:
        return ds

    def has_text(ex: dict) -> bool:
        v = ex.get(text_column, None)
        if v is None:
            return False
        if isinstance(v, str):
            return len(v.strip()) >= min_text_len
        return False

    ds = ds.filter(has_text)

    def strip_text(batch: dict) -> dict:
        texts = batch[text_column]
        # batch can be scalar if batched=False; handle both
        if isinstance(texts, list):
            return {text_column: [t.strip() for t in texts]}
        return {text_column: texts.strip()}

    ds = ds.map(strip_text, batched=True)
    return ds


def maybe_drop_for_tabular(ds: Dataset, drop_columns: Optional[List[str]]) -> Dataset:
    if not drop_columns:
        return ds
    cols_to_drop = [c for c in drop_columns if c in ds.column_names]
    if cols_to_drop:
        ds = ds.remove_columns(cols_to_drop)
    return ds


def extract_attributes(
    ds: Dataset,
    schema_path: Optional[str] = None,
    text_column: str = "text",
    category_column: str = "category2",
) -> Dataset:
    """
    Extract fashion attributes from text descriptions.
    
    Adds columns:
    - attr_material, attr_pattern, attr_neckline, attr_sleeve (lists)
    - attr_material_primary, attr_pattern_primary, etc. (single canonical tag)
    
    Args:
        ds: Dataset to process
        schema_path: Path to attribute schema YAML (optional)
        text_column: Name of text column
        category_column: Name of category column
        
    Returns:
        Dataset with added attribute columns
    """
    print("Initializing attribute extractor...")
    extractor = AttributeExtractor(schema_path)
    attr_names = extractor.get_attribute_names()
    print(f"Extracting attributes: {attr_names}")
    
    def process_example(example: dict) -> dict:
        text = example.get(text_column, "")
        category2 = example.get(category_column, "")
        
        extracted = extractor.extract_with_primary(text, category2)
        example.update(extracted)
        return example
    
    ds = ds.map(process_example, desc="Extracting attributes")
    
    # Print coverage summary
    print("\nAttribute extraction coverage:")
    for attr_name in attr_names:
        primary_col = f"attr_{attr_name}_primary"
        if primary_col in ds.column_names:
            values = ds[primary_col]
            non_unknown = sum(1 for v in values if v != extractor.unknown_value)
            coverage = non_unknown / len(values) * 100
            print(f"  {attr_name}: {coverage:.1f}% tagged ({non_unknown}/{len(values)})")
    
    return ds


def save_dataset(
    ds_or_dict: Dataset | DatasetDict,
    out_dir: str,
    save_format: str,
    drop_image_for_tabular: bool = True,
    image_column: str = "image",
):
    ensure_dir(out_dir)

    if isinstance(ds_or_dict, DatasetDict):
        splits = ds_or_dict
    else:
        splits = DatasetDict({"train": ds_or_dict})

    if save_format == "hf":
        # Save entire dict in a single directory
        path = os.path.join(out_dir, "hf")
        ensure_dir(path)
        splits.save_to_disk(path)
        print(f"Saved Hugging Face dataset to: {path}")
        return

    # For tabular formats, optionally drop image column (not supported in plain parquet/csv)
    if drop_image_for_tabular and image_column:
        splits = DatasetDict(
            {k: maybe_drop_for_tabular(v, [image_column]) for k, v in splits.items()}
        )

    if save_format == "parquet":
        for split_name, split_ds in splits.items():
            path = os.path.join(out_dir, f"{split_name}.parquet")
            split_ds.to_parquet(path)
            print(f"Saved {split_name} split to: {path}")
        return

    if save_format == "csv":
        for split_name, split_ds in splits.items():
            path = os.path.join(out_dir, f"{split_name}.csv")
            split_ds.to_csv(path)
            print(f"Saved {split_name} split to: {path}")
        return


def main() -> None:
    args = parse_args()

    print(f"Loading: {args.input}")
    ds = load_parquet(args.input)
    print(ds)
    print(ds.features)

    ds = clean_and_filter_text(ds, text_column=args.text_column, min_text_len=args.min_text_len)
    print("After cleaning/filtering:", ds)

    # Extract attributes if requested
    if args.extract_attributes:
        ds = extract_attributes(
            ds,
            schema_path=args.schema_path,
            text_column=args.text_column,
        )
        print("After attribute extraction:", ds)
        print(ds.features)

    if args.val_ratio and args.val_ratio > 0.0:
        dsd = ds.train_test_split(test_size=args.val_ratio, seed=args.seed)
        # Rename for clarity
        dsd = DatasetDict(train=dsd["train"], validation=dsd["test"])
        print({k: v.num_rows for k, v in dsd.items()})
        save_dataset(
            dsd,
            out_dir=args.output_dir,
            save_format=args.save_format,
            drop_image_for_tabular=True,
            image_column=args.image_column,
        )
    else:
        save_dataset(
            ds,
            out_dir=args.output_dir,
            save_format=args.save_format,
            drop_image_for_tabular=True,
            image_column=args.image_column,
        )


if __name__ == "__main__":
    main()
