from lucidflow.models.quarantine_classifier.features import extract_row_features

_BASE_ROW = {
    "company_id": "1009",
    "name": "Acme Corp",
    "description": "We are a company that builds widgets for everyone.",
    "company_size": "3",
    "state": "NY",
    "country": "US",
    "city": "New York",
    "zip_code": "10001",
    "address": "123 Main St",
    "url": "https://www.linkedin.com/company/acme",
}

_ZIP3_TO_STATE = {"100": "NY", "902": "CA"}


def test_clean_row_has_zero_null_count_and_zero_contract_violations():
    features = extract_row_features(_BASE_ROW, _ZIP3_TO_STATE)

    assert features["null_count"] == 0
    assert features["contract_violation_count"] == 0


def test_null_count_reflects_actual_missing_optional_fields():
    row = dict(_BASE_ROW)
    row["state"] = None
    row["description"] = None

    features = extract_row_features(row, _ZIP3_TO_STATE)

    assert features["null_count"] == 2


def test_suspicious_char_rate_is_zero_for_plain_ascii_text():
    features = extract_row_features(_BASE_ROW, _ZIP3_TO_STATE)

    assert features["suspicious_char_rate"] == 0.0


def test_suspicious_char_rate_is_positive_for_mojibake_text():
    row = dict(_BASE_ROW)
    row["description"] = "We areÂ a companyâ€™s widgets"

    features = extract_row_features(row, _ZIP3_TO_STATE)

    assert features["suspicious_char_rate"] > 0.0


def test_ends_without_terminal_punct_flags_mid_word_endings():
    row_terminal = dict(_BASE_ROW)
    row_terminal["description"] = "A complete sentence."
    row_midword = dict(_BASE_ROW)
    row_midword["description"] = "A sentence cut off mid wo"

    assert extract_row_features(row_terminal, _ZIP3_TO_STATE)["ends_without_terminal_punct"] == 0
    assert extract_row_features(row_midword, _ZIP3_TO_STATE)["ends_without_terminal_punct"] == 1


def test_zip_state_mismatch_is_unknown_for_non_us_rows():
    row = dict(_BASE_ROW)
    row["country"] = "GB"

    features = extract_row_features(row, _ZIP3_TO_STATE)

    assert features["zip_state_mismatch"] == -1


def test_zip_state_mismatch_is_zero_when_consistent():
    features = extract_row_features(_BASE_ROW, _ZIP3_TO_STATE)  # zip3 "100" -> NY, state NY

    assert features["zip_state_mismatch"] == 0


def test_zip_state_mismatch_is_one_when_inconsistent():
    row = dict(_BASE_ROW)
    row["zip_code"] = "90210"  # zip3 "902" -> CA, but state is still NY

    features = extract_row_features(row, _ZIP3_TO_STATE)

    assert features["zip_state_mismatch"] == 1


def test_contract_violation_count_is_positive_for_an_invalid_row():
    row = dict(_BASE_ROW)
    row["company_size"] = "99"  # outside the valid 1-7 range

    features = extract_row_features(row, _ZIP3_TO_STATE)

    assert features["contract_violation_count"] > 0
