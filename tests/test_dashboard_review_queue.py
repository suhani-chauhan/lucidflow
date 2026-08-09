import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

from data_access import find_ml_flag_reason, row_to_queue_entry

_FEATURES = {"null_count": 1, "name_len": 12}


def test_find_ml_flag_reason_picks_the_enriched_classifier_entry():
    reasons = [
        {"rule": "url", "message": "bad url", "severity": "error"},
        {"rule": "quarantine_classifier", "message": "flagged", "score": 0.9, "features": _FEATURES},
    ]

    found = find_ml_flag_reason(reasons)

    assert found is not None
    assert found["score"] == 0.9


def test_find_ml_flag_reason_ignores_contract_failures_only():
    reasons = [{"rule": "url", "message": "bad url", "severity": "error"}]

    assert find_ml_flag_reason(reasons) is None


def test_find_ml_flag_reason_ignores_unenriched_ml_flags():
    # An ML flag from before score/features were persisted -- not reviewable.
    reasons = [{"rule": "quarantine_classifier", "message": "flagged", "model_version": "1"}]

    assert find_ml_flag_reason(reasons) is None


def test_row_to_queue_entry_builds_a_full_entry_for_a_reviewable_row():
    raw_data = {"company_id": "1009", "name": "Acme Corp"}
    reasons = [
        {
            "rule": "quarantine_classifier",
            "message": "ML quarantine classifier flagged this row (score=0.9000, threshold=0.5)",
            "severity": "warning",
            "model_version": "3",
            "score": 0.9,
            "features": _FEATURES,
        }
    ]

    entry = row_to_queue_entry(42, raw_data, reasons, "2026-08-09T00:00:00Z")

    assert entry == {
        "record_id": 42,
        "raw_data": raw_data,
        "score": 0.9,
        "message": reasons[0]["message"],
        "model_version": "3",
        "features": _FEATURES,
        "quarantined_at": "2026-08-09T00:00:00Z",
    }


def test_row_to_queue_entry_returns_none_for_a_contract_failure_row():
    raw_data = {"company_id": "1009"}
    reasons = [{"rule": "company_size", "message": "out of range", "severity": "error"}]

    assert row_to_queue_entry(42, raw_data, reasons, "2026-08-09T00:00:00Z") is None
