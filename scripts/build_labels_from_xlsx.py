"""Export only de-identified training labels from the clinical workbook."""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CELL_REF = re.compile(r"([A-Z]+)(\d+)")


def column_index(reference: str) -> int:
    letters = CELL_REF.fullmatch(reference).group(1)
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def read_rows(path: Path) -> list[list[object]]:
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = [
                "".join(node.text or "" for node in item.findall(".//m:t", NS))
                for item in root.findall("m:si", NS)
            ]
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows = []
    for row_node in root.findall(".//m:sheetData/m:row", NS):
        values: dict[int, object] = {}
        for cell in row_node.findall("m:c", NS):
            index = column_index(cell.attrib["r"])
            kind = cell.attrib.get("t")
            value_node = cell.find("m:v", NS)
            inline = cell.find("m:is/m:t", NS)
            if inline is not None:
                value: object = inline.text or ""
            elif value_node is None:
                value = None
            elif kind == "s":
                value = strings[int(value_node.text)]
            else:
                raw = value_node.text or ""
                try:
                    number = float(raw)
                    value = int(number) if number.is_integer() else number
                except ValueError:
                    value = raw
            values[index] = value
        rows.append([values.get(index) for index in range(max(values, default=-1) + 1)])
    return rows


def export_labels(workbook: Path, output: Path, expected_subjects: int) -> None:
    rows = read_rows(workbook)
    header_index = next(
        index
        for index, row in enumerate(rows)
        if "subject_id" in row and "severity" in row
    )
    headers = [str(value).strip() if value is not None else "" for value in rows[header_index]]
    subject_index = headers.index("subject_id")
    severity_index = headers.index("severity")
    labels = []
    for row in rows[header_index + 1 :]:
        if subject_index >= len(row) or row[subject_index] in (None, ""):
            continue
        subject = str(row[subject_index]).strip()
        if not re.fullmatch(r"S\d{2}", subject):
            raise ValueError(f"Invalid de-identified subject ID: {subject!r}")
        severity = int(row[severity_index])
        if severity not in {0, 1, 2, 3}:
            raise ValueError(f"Invalid severity for {subject}: {severity}")
        labels.append((subject, 0 if severity == 0 else 1, severity))

    if len(labels) != expected_subjects or len({item[0] for item in labels}) != len(labels):
        raise ValueError(
            f"Expected {expected_subjects} unique subjects, got {len(labels)} rows and "
            f"{len({item[0] for item in labels})} unique IDs"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "binary_label", "severity_label"])
        writer.writerows(labels)
    counts = Counter(item[2] for item in labels)
    print(f"Wrote {len(labels)} de-identified labels to {output}")
    print("severity_counts", dict(sorted(counts.items())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-subjects", type=int, default=26)
    args = parser.parse_args()
    export_labels(args.workbook.resolve(), args.output.resolve(), args.expected_subjects)


if __name__ == "__main__":
    main()
