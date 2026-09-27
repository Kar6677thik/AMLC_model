from __future__ import annotations

import csv
from pathlib import Path

SOURCE_HEADER = ["entity_id", "business_name", "business_address", "country"]
MATCH_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def tsv_rows(path, header):
    csv.field_size_limit(64 * 1024 * 1024)
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t", strict=True)
        actual = next(reader, None)
        if actual != header:
            raise ValueError(f"{path}: expected header {header}, got {actual}")
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"{path}:{reader.line_num}: expected {len(header)} columns")
            yield row


def id_list(value):
    if value == "":
        return []
    ids = value.split(",")
    if any(not item or item != item.strip() or any(c in item for c in '\t\r\n"') for item in ids):
        raise ValueError("Invalid whitespace, quoting, or empty ID in ID list")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate target ID in ID list")
    return ids


def write_header(stream, header):
    stream.write("\t".join(header) + "\n")


def write_row(stream, s1, targets):
    targets = sorted(targets)
    if len(targets) != len(set(targets)):
        raise ValueError(f"Duplicate targets for {s1}")
    stream.write(s1 + "\t" + ",".join(targets) + "\n")


def data_files(dataset, split):
    base = Path(dataset) / split
    files = [base / f"{split}_source{i}.tsv" for i in (1, 2, 3)]
    if split == "train":
        files.append(base / "train_ground_truth.tsv")
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return files
