#!/usr/bin/env python3
"""Attach de-identified, reviewed head-turn spans to a window manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if letters is None:
        raise ValueError(f"Invalid XLSX cell reference: {reference}")
    result = 0
    for letter in letters.group(0):
        result = result * 26 + ord(letter) - 64
    return result - 1


def read_xlsx_records(path: Path) -> list[dict[str, str]]:
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", NS):
                shared.append("".join(node.text or "" for node in item.iterfind(".//m:t", NS)))
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        sparse_rows: list[dict[int, str]] = []
        for row in root.findall("m:sheetData/m:row", NS):
            values: dict[int, str] = {}
            for cell in row.findall("m:c", NS):
                index = column_index(cell.attrib["r"])
                value_node = cell.find("m:v", NS)
                if cell.attrib.get("t") == "inlineStr":
                    value = "".join(node.text or "" for node in cell.iterfind(".//m:t", NS))
                elif value_node is None:
                    value = ""
                elif cell.attrib.get("t") == "s":
                    value = shared[int(value_node.text)]
                else:
                    value = value_node.text or ""
                values[index] = value
            sparse_rows.append(values)
    if not sparse_rows:
        raise ValueError(f"Workbook has no rows: {path}")
    width = max((max(row, default=-1) for row in sparse_rows), default=-1) + 1
    rows = [[row.get(index, "") for index in range(width)] for row in sparse_rows]
    headers = [value.strip() for value in rows[0]]
    if "subject_id" not in headers:
        raise ValueError("Workbook sheet1 lacks subject_id")
    return [
        dict(zip(headers, row))
        for row in rows[1:]
        if any(value.strip() for value in row)
    ]


def flag(record: dict[str, str], key: str) -> bool:
    return record.get(key, "").strip().lower() in {"1", "true", "yes"}


def finite_number(record: dict[str, str], key: str) -> float:
    raw = record.get(key, "").strip()
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key} for {record.get('subject_id')}: {raw}")
    return value


def select_headturn_values(
    records: list[dict[str, str]], mode: str
) -> dict[str, tuple[float, str]]:
    selected: dict[str, tuple[float, str]] = {}
    for record in records:
        subject = record.get("subject_id", "").strip()
        if not re.fullmatch(r"S(?:0[1-9]|1\d|2[0-6])", subject):
            continue
        if subject in selected:
            raise ValueError(f"Duplicate subject_id in workbook: {subject}")
        if mode == "primary" and flag(record, "headturn_analysis_eligible"):
            selected[subject] = (
                finite_number(record, "headturn_analysis_span_deg"),
                "reviewed_primary",
            )
        elif mode == "primary_or_sensitivity":
            if flag(record, "headturn_analysis_eligible"):
                selected[subject] = (
                    finite_number(record, "headturn_analysis_span_deg"),
                    "reviewed_primary",
                )
            elif flag(record, "headturn_warning_sensitivity_eligible"):
                selected[subject] = (
                    finite_number(record, "headturn_recorded_span_deg"),
                    "reviewed_warning_sensitivity",
                )
        elif mode == "recorded":
            selected[subject] = (
                finite_number(record, "headturn_recorded_span_deg"),
                record.get("headturn_recorded_kind", "recorded").strip(),
            )
    if not selected:
        raise ValueError(f"No reviewed head-turn values selected for mode={mode}")
    return selected


def attach_rows(
    manifest_rows: list[dict[str, str]], selected: dict[str, tuple[float, str]]
) -> list[dict[str, str]]:
    attached: list[dict[str, str]] = []
    for row in manifest_rows:
        subject = row["subject_id"].strip()
        if subject not in selected:
            continue
        value, source = selected[subject]
        output = dict(row)
        output["headturn_span_deg"] = f"{value:.10f}"
        output["headturn_feature_source"] = source
        attached.append(output)
    if not attached:
        raise ValueError("No manifest rows remained after head-turn eligibility filtering")
    return attached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["primary", "primary_or_sensitivity", "recorded"],
        default="primary",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()

    with args.manifest.resolve().open(newline="", encoding="utf-8-sig") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if not manifest_rows:
        raise ValueError(f"Empty manifest: {args.manifest}")
    selected = select_headturn_values(read_xlsx_records(args.workbook.resolve()), args.mode)
    attached = attach_rows(manifest_rows, selected)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(attached[0]))
        writer.writeheader()
        writer.writerows(attached)

    input_subjects = {row["subject_id"] for row in manifest_rows}
    output_subjects = {row["subject_id"] for row in attached}
    audit = {
        "mode": args.mode,
        "input_windows": len(manifest_rows),
        "input_subjects": len(input_subjects),
        "output_windows": len(attached),
        "output_subjects": len(output_subjects),
        "included_subject_ids": sorted(output_subjects),
        "excluded_subject_ids": sorted(input_subjects - output_subjects),
        "feature_source_subject_counts": dict(
            sorted(Counter(selected[subject][1] for subject in output_subjects).items())
        ),
        "severity_subject_counts": dict(
            sorted(
                Counter(
                    next(row["severity_label"] for row in attached if row["subject_id"] == subject)
                    for subject in output_subjects
                ).items()
            )
        ),
    }
    if args.audit_output:
        audit_output = args.audit_output.resolve()
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

