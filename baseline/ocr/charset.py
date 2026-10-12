"""Training-only vocabulary for subsequent CRNN experiments."""

import argparse
from pathlib import Path

from .io import digest, normalize, read_table, write_json

PROVINCES = frozenset("皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新")
ALLOWED = PROVINCES | frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def build_charset(csv_path):
    rows = read_table(csv_path)
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("Vocabulary must be built from a nonempty training CSV only")
    texts = [normalize(row["plate_text"]) for row in rows]
    if any(not text for text in texts):
        raise ValueError("Empty training plate text")
    characters = set("".join(texts))
    if characters - ALLOWED:
        raise ValueError(f"Characters outside CCPD province/Latin/digit alphabet: {sorted(characters - ALLOWED)}")
    return {
        "schema_version": 1,
        "source_sha256": digest(csv_path),
        "characters": sorted(characters),
        "ctc_blank_index": 0,
        "character_to_index": {char: i for i, char in enumerate(sorted(characters), 1)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset"))
    parser.add_argument("--output", type=Path, default=Path("baseline/ocr/charset.json"))
    args = parser.parse_args()
    write_json(args.output, build_charset(args.dataset / "ocr_train.csv"))


if __name__ == "__main__":
    main()
