"""
01_prepare_dataset.py
==================
Waste classifier - preprocessing, split.

- Scans source dataset (each class is a subfolder)
- Search for exact duplicates (SHA1 of file bytes) and removes them before splitting
- Stratified split 70 / 15 / 15  (train / val / test)
- Letterbox resize to 224x224 (keeps aspect ratio, no distortion)
- Saves processed images to dataset_split/

Requirements:
    pip install Pillow scikit-learn tqdm numpy

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --src ./my_dataset --dst ./my_split
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    # Source directory - each class is a subfolder
    "src_dir": "./dataset_raw",

    # Output directory for the split dataset
    "dst_dir": "./dataset_split",

    # Target size required by MobileNetV2
    "target_size": (224, 224),

    # Split ratios (must sum to 1.0)
    "train_ratio": 0.70,
    "val_ratio":   0.15,
    "test_ratio":  0.15,

    # Letterbox padding color (neutral gray)
    # Change to (0, 0, 0) for black padding
    "padding_color": (114, 114, 114),

    # Accepted file extensions
    "extensions": {".jpg", ".jpeg", ".png", ".bmp", ".webp"},

    # Random seed - guarantees reproducible split
    "random_seed": 42,

    # Exact dedup: SHA1 of raw file bytes. Catches verbatim copies only
    # (byte-identical files). Re-saved or recompressed copies are not detected.
    "dedup_exact": True,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def letterbox_resize(img, target_size, padding_color):
    """
    Resize image keeping aspect ratio, pad the remaining space.

    Example: 800x200 image -> scaled to 224x56, then centered
    on a 224x224 canvas with padding on top and bottom.

    Better than plain stretch which distorts object shapes.
    """
    img.thumbnail(target_size, Image.LANCZOS)

    new_img = Image.new("RGB", target_size, padding_color)

    offset_x = (target_size[0] - img.width) // 2
    offset_y = (target_size[1] - img.height) // 2
    new_img.paste(img, (offset_x, offset_y))

    return new_img


def scan_dataset(src_dir, extensions):
    """
    Scan source directory and return dict:
    { class_name: [Path, ...] }

    Skips empty folders and unsupported file types.
    """
    dataset = {}

    for class_dir in sorted(src_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        images = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]

        if not images:
            print("  [WARNING] Empty folder skipped: " + class_dir.name)
            continue

        dataset[class_dir.name] = images

    return dataset


def file_sha1(path, chunk_size=65536):
    """
    SHA1 of raw file bytes - fast exact-duplicate check.
    Streams the file in chunks so it works on large images too.
    """
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def deduplicate_dataset(dataset):
    """
    Remove exact-byte duplicate images from the dataset BEFORE splitting.

    Doing this before the split is critical: if a duplicate ends up
    in train AND val (or test), val/test accuracy is silently inflated
    and the model looks better than it is.

    Detection: SHA1 of file bytes. Catches verbatim copies of the same
    file. Does NOT catch re-saved / recompressed versions of the same
    image (those have different bytes but look identical) - a perceptual
    hash would be needed for that.

    For each duplicate cluster, the first file (in sorted order) is kept.
    Cross-class duplicates are flagged separately - they're a data
    quality bug, not just redundancy.

    Returns:
        cleaned_dataset: same dict shape as input, duplicates removed
        report: dict with stats and the list of removed files
    """
    # Flatten to (path, class) list, sorted for determinism
    all_files = []
    for class_name in sorted(dataset.keys()):
        for f in sorted(dataset[class_name]):
            all_files.append((f, class_name))

    total = len(all_files)
    print("  Scanning " + str(total) + " files for exact duplicates...")

    seen_sha1 = {}    # sha1 -> (path, class)
    removed = []      # list of dicts describing each drop
    cross_class = []  # subset of removed where classes differ

    for path, class_name in tqdm(all_files, desc="  dedup", unit="img"):
        try:
            sha = file_sha1(path)
        except OSError as e:
            # Unreadable file - drop it now, the processing step
            # would fail on it anyway.
            removed.append({
                "file": str(path), "class": class_name,
                "reason": "unreadable", "kept": None, "error": str(e),
            })
            continue

        if sha in seen_sha1:
            kept_path, kept_class = seen_sha1[sha]
            entry = {
                "file": str(path), "class": class_name,
                "reason": "exact",
                "kept": str(kept_path), "kept_class": kept_class,
            }
            removed.append(entry)
            if class_name != kept_class:
                cross_class.append(entry)
            continue

        seen_sha1[sha] = (path, class_name)

    # Rebuild dataset dict from survivors
    removed_paths = {r["file"] for r in removed}
    cleaned = {}
    for class_name, files in dataset.items():
        kept = [f for f in files if str(f) not in removed_paths]
        if kept:
            cleaned[class_name] = kept

    exact_removed = sum(1 for r in removed if r["reason"] == "exact")
    unreadable = sum(1 for r in removed if r["reason"] == "unreadable")

    report = {
        "input_total": total,
        "exact_removed": exact_removed,
        "unreadable": unreadable,
        "kept_total": total - len(removed),
        "cross_class_duplicates": len(cross_class),
        "removed": removed,
        "cross_class": cross_class,
    }

    # Console summary
    print("  Exact duplicates removed:       " + str(exact_removed))
    if unreadable:
        print("  Unreadable files dropped:       " + str(unreadable))
    if cross_class:
        print("  [!] Cross-class duplicates:     " + str(len(cross_class))
              + " (same image in different classes - check data quality)")
        for entry in cross_class[:5]:
            print("      " + entry["class"] + " <-> " + entry["kept_class"]
                  + " : " + Path(entry["file"]).name)
        if len(cross_class) > 5:
            print("      ... and " + str(len(cross_class) - 5) + " more")
    print("  Kept after dedup:               " + str(report["kept_total"])
          + " / " + str(total))

    return cleaned, report


def save_dedup_report(report, dst_dir):
    """Save the full list of removed duplicates for audit."""
    path = dst_dir / "dedup_report.json"
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)
    print("  Dedup report saved -> " + str(path))


def stratified_split(dataset, train_ratio, val_ratio, seed):
    """
    Stratified split - each class is represented proportionally
    in every subset (train / val / test).

    Prevents any class from being over-represented in test.
    """
    train_files, val_files, test_files = [], [], []

    for class_name, files in dataset.items():
        # First: separate test from the rest
        rest, test = train_test_split(
            files,
            test_size=(1.0 - train_ratio - val_ratio),
            random_state=seed,
        )
        # Second: split the rest into train and val
        val_size_relative = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            rest,
            test_size=val_size_relative,
            random_state=seed,
        )

        train_files.extend([(f, class_name) for f in train])
        val_files.extend([(f, class_name) for f in val])
        test_files.extend([(f, class_name) for f in test])

    return train_files, val_files, test_files


def process_and_save(file_list, split_name, dst_dir, target_size, padding_color):
    """
    Process images from list and save to destination directory.
    Returns processing statistics.
    """
    stats = {"processed": 0, "skipped": 0, "errors": []}

    for src_path, class_name in tqdm(file_list, desc=("  " + split_name), unit="img"):
        dst_class_dir = dst_dir / split_name / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        dst_path = dst_class_dir / src_path.name

        # Skip if file already exists (allows resuming after interruption)
        if dst_path.exists():
            stats["skipped"] += 1
            continue

        try:
            with Image.open(src_path) as img:
                # Convert to RGB (handles RGBA, grayscale, CMYK, etc.)
                img = img.convert("RGB")
                img = letterbox_resize(img, target_size, padding_color)
                img.save(dst_path, "JPEG", quality=95)
                stats["processed"] += 1

        except Exception as e:
            stats["errors"].append({"file": str(src_path), "error": str(e)})
            stats["skipped"] += 1

    return stats


def print_summary(dataset, train, val, test):
    """Print a readable summary of the dataset split."""

    total = sum(len(v) for v in dataset.values())

    print("\n" + "=" * 52)
    print("  DATASET SUMMARY")
    print("=" * 52)
    print("  Total images:   " + str(total))
    print("  Classes:        " + str(len(dataset)))
    print("  Train:          " + str(len(train)) + "  (" + str(round(len(train)/total*100, 1)) + "%)")
    print("  Val:            " + str(len(val))   + "  (" + str(round(len(val)/total*100,   1)) + "%)")
    print("  Test:           " + str(len(test))  + "  (" + str(round(len(test)/total*100,  1)) + "%)")
    print("-" * 52)
    print("  Class                  Total   Train   Val  Test")
    print("-" * 52)

    class_counts = {cls: len(files) for cls, files in dataset.items()}
    train_counts, val_counts, test_counts = {}, {}, {}
    for _, cls in train:
        train_counts[cls] = train_counts.get(cls, 0) + 1
    for _, cls in val:
        val_counts[cls] = val_counts.get(cls, 0) + 1
    for _, cls in test:
        test_counts[cls] = test_counts.get(cls, 0) + 1

    for cls in sorted(dataset.keys()):
        n = class_counts[cls]
        tr = train_counts.get(cls, 0)
        va = val_counts.get(cls, 0)
        te = test_counts.get(cls, 0)
        print(f"  {cls:<20} {n:>7}  {tr:>6}  {va:>5}  {te:>5}")

    print("=" * 52)


def save_split_manifest(train, val, test, dst_dir):
    """
    Save split manifest (JSON).
    Allows reproducing the exact same split without re-running the script.
    """
    manifest = {
        "train": [{"file": str(f), "class": c} for f, c in train],
        "val":   [{"file": str(f), "class": c} for f, c in val],
        "test":  [{"file": str(f), "class": c} for f, c in test],
    }
    manifest_path = dst_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)
    print("  Manifest saved -> " + str(manifest_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Waste dataset preprocessing and split")
    parser.add_argument("--src",  default=CONFIG["src_dir"],  help="Source directory with class folders")
    parser.add_argument("--dst",  default=CONFIG["dst_dir"],  help="Output directory for the split")
    parser.add_argument("--size", type=int, default=224,      help="Target image size (default: 224)")
    parser.add_argument("--seed", type=int, default=CONFIG["random_seed"])
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable exact-duplicate detection")
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    target_size = (args.size, args.size)

    print("")
    print("  Source:    " + str(src_dir.resolve()))
    print("  Output:    " + str(dst_dir.resolve()))
    print("  Size:      " + str(target_size[0]) + "x" + str(target_size[1]) + " px")
    print("  Seed:      " + str(args.seed))
    print("")

    # 1. Scan dataset
    print(">> Scanning dataset...")
    if not src_dir.exists():
        raise FileNotFoundError("Source directory not found: " + str(src_dir))

    dataset = scan_dataset(src_dir, CONFIG["extensions"])
    if not dataset:
        raise ValueError("No images found. Check the path and file extensions.")

    # 2. Deduplicate BEFORE splitting (prevents leakage between train/val/test)
    if CONFIG["dedup_exact"] and not args.no_dedup:
        print(">> Removing exact duplicates...")
        dataset, dedup_report = deduplicate_dataset(dataset)
        if not dataset:
            raise ValueError("All images were removed during dedup - check inputs.")
    else:
        print(">> Deduplication skipped (--no-dedup)")
        dedup_report = None

    # 3. Stratified split
    print(">> Stratified split 70/15/15...")
    train_files, val_files, test_files = stratified_split(
        dataset,
        CONFIG["train_ratio"],
        CONFIG["val_ratio"],
        args.seed,
    )

    # 4. Print summary before processing
    print_summary(dataset, train_files, val_files, test_files)

    # 5. Save manifest and deduplication report before processing (in case of interruption)
    dst_dir.mkdir(parents=True, exist_ok=True)
    save_split_manifest(train_files, val_files, test_files, dst_dir)
    if dedup_report is not None:
        save_dedup_report(dedup_report, dst_dir)

    # 6. Process and save images
    print("")
    print(">> Processing images (letterbox resize to 224x224)...")
    all_stats = {}
    for split_name, file_list in [
        ("train", train_files),
        ("val",   val_files),
        ("test",  test_files),
    ]:
        stats = process_and_save(
            file_list,
            split_name,
            dst_dir,
            target_size,
            CONFIG["padding_color"],
        )
        all_stats[split_name] = stats

    # 7. Final report
    print("")
    print(">> Results:")
    total_processed = 0
    total_skipped = 0
    for split_name, stats in all_stats.items():
        print("  " + split_name.ljust(5) +
              "  processed: " + str(stats["processed"]).rjust(5) +
              "  skipped: "   + str(stats["skipped"]).rjust(4) +
              "  errors: "    + str(len(stats["errors"])))
        total_processed += stats["processed"]
        total_skipped   += stats["skipped"]

        if stats["errors"]:
            print("  [!] Errors in '" + split_name + "':")
            for err in stats["errors"][:5]:
                print("      " + err["file"] + ": " + err["error"])
            if len(stats["errors"]) > 5:
                print("      ... and " + str(len(stats["errors"]) - 5) + " more")

    print("")
    print("  Total processed: " + str(total_processed) + " images")
    if total_skipped:
        print("  Skipped (already exist or error): " + str(total_skipped))

    print("")
    print("  Done. Dataset ready in: " + str(dst_dir.resolve()))
    print("  Next step -> model training")
    print("")


if __name__ == "__main__":
    main()
