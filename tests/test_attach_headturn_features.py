import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "attach_headturn_features.py"
SPEC = importlib.util.spec_from_file_location("attach_headturn_features", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_primary_and_sensitivity_selection() -> None:
    records = [
        {
            "subject_id": "S01",
            "headturn_analysis_eligible": "1",
            "headturn_analysis_span_deg": "120.5",
            "headturn_warning_sensitivity_eligible": "0",
            "headturn_recorded_span_deg": "121.0",
        },
        {
            "subject_id": "S02",
            "headturn_analysis_eligible": "0",
            "headturn_warning_sensitivity_eligible": "1",
            "headturn_recorded_span_deg": "88.25",
        },
        {
            "subject_id": "S03",
            "headturn_analysis_eligible": "0",
            "headturn_warning_sensitivity_eligible": "0",
            "headturn_recorded_span_deg": "20.0",
        },
    ]
    primary = MODULE.select_headturn_values(records, "primary")
    sensitivity = MODULE.select_headturn_values(records, "primary_or_sensitivity")
    assert primary == {"S01": (120.5, "reviewed_primary")}
    assert set(sensitivity) == {"S01", "S02"}
    assert sensitivity["S02"] == (88.25, "reviewed_warning_sensitivity")


def test_attach_rows_filters_and_adds_only_deidentified_features() -> None:
    rows = [
        {"subject_id": "S01", "severity_label": "0", "window_id": "0"},
        {"subject_id": "S02", "severity_label": "1", "window_id": "0"},
    ]
    attached = MODULE.attach_rows(rows, {"S02": (90.0, "reviewed_primary")})
    assert len(attached) == 1
    assert attached[0]["subject_id"] == "S02"
    assert attached[0]["headturn_span_deg"] == "90.0000000000"

