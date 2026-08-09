import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

from data_access import count_reasons_by_rule, format_coverage_caveat


def test_count_reasons_by_rule_tallies_across_rows():
    reasons_per_row = [
        [{"rule": "url", "message": "bad url", "severity": "error"}],
        [
            {"rule": "url", "message": "bad url", "severity": "error"},
            {"rule": "company_size", "message": "out of range", "severity": "error"},
        ],
        [{"rule": "quarantine_classifier", "message": "flagged", "severity": "warning"}],
    ]

    counts = count_reasons_by_rule(reasons_per_row)

    assert counts == {"url": 2, "company_size": 1, "quarantine_classifier": 1}


def test_count_reasons_by_rule_handles_no_rows():
    assert count_reasons_by_rule([]) == {}


def test_format_coverage_caveat_is_none_when_no_singleton_classes():
    assert format_coverage_caveat(0, 7) is None
    assert format_coverage_caveat(None, 7) is None


def test_format_coverage_caveat_reports_singleton_and_total_class_counts():
    caveat = format_coverage_caveat(27, 53)

    assert "27/80 classes" in caveat
    assert "structurally impossible" in caveat
