from lucidflow.models.quarantine_classifier.retrain_with_reviews import reviewed_row_to_labeled_row
from lucidflow.models.quarantine_classifier.train import row_label

_FEATURES = {"null_count": 0, "name_len": 9}


def test_reviewed_row_to_labeled_row_confirmed_bad_gets_positive_label():
    raw_data = {"company_id": "1009", "name": "Acme Corp"}
    reasons = [{"rule": "quarantine_classifier", "features": _FEATURES, "score": 0.87}]

    row = reviewed_row_to_labeled_row(raw_data, reasons, "confirmed_bad")

    assert row["label"] == 1
    assert row["is_synthetic"] is False  # real, not synthetic -- never claim otherwise
    assert row["corruption_type"] == "human_confirmed_bad"
    assert row["features"] == _FEATURES
    assert row["company_id"] == "1009"


def test_reviewed_row_to_labeled_row_false_positive_gets_negative_label():
    raw_data = {"company_id": "42", "name": "Beta Inc"}
    reasons = [{"rule": "quarantine_classifier", "features": _FEATURES, "score": 0.55}]

    row = reviewed_row_to_labeled_row(raw_data, reasons, "false_positive")

    assert row["label"] == 0
    assert row["is_synthetic"] is False
    assert row["corruption_type"] == "human_false_positive"


def test_reviewed_row_to_labeled_row_returns_none_without_an_ml_flag_reason():
    raw_data = {"company_id": "7", "name": "Gamma LLC"}
    reasons = [{"rule": "url", "message": "bad url", "severity": "error"}]

    assert reviewed_row_to_labeled_row(raw_data, reasons, "confirmed_bad") is None


def test_reviewed_row_to_labeled_row_returns_none_for_unenriched_ml_flag():
    # An ML flag written before Phase 5, Task 3 started persisting "features" -- shouldn't
    # be usable for retraining even if somehow reviewed.
    raw_data = {"company_id": "7", "name": "Gamma LLC"}
    reasons = [{"rule": "quarantine_classifier", "message": "flagged", "score": 0.9}]

    assert reviewed_row_to_labeled_row(raw_data, reasons, "confirmed_bad") is None


def test_row_label_prefers_explicit_label_field():
    assert row_label({"label": 1, "is_synthetic": False}) == 1
    assert row_label({"label": 0, "is_synthetic": True}) == 0  # explicit label wins over is_synthetic


def test_row_label_falls_back_to_is_synthetic_when_label_missing():
    assert row_label({"is_synthetic": True}) == 1
    assert row_label({"is_synthetic": False}) == 0
