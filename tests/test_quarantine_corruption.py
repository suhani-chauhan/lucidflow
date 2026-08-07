import random

from lucidflow.models.quarantine_classifier.corruption import (
    corrupt_encoding,
    corrupt_null_storm,
    corrupt_truncation,
    corrupt_zip_state_mismatch,
)

_BASE_ROW = {
    "company_id": "1009",
    "name": "Acme Corp",
    "description": "We are a company that builds widgets for everyone. Est. 1990.",
    "company_size": "3",
    "state": "NY",
    "country": "US",
    "city": "New York",
    "zip_code": "10001",
    "address": "123 Main St",
    "url": "https://www.linkedin.com/company/acme",
}


def test_corrupt_encoding_changes_description_and_does_not_mutate_original():
    row = dict(_BASE_ROW)
    corrupted, corruption_type = corrupt_encoding(row, random.Random(1))

    assert corruption_type == "encoding"
    assert corrupted["description"] != _BASE_ROW["description"]
    assert row == _BASE_ROW  # original untouched
    assert corrupted["name"] == _BASE_ROW["name"]  # only the targeted field changes


def test_corrupt_encoding_falls_back_to_insertion_when_no_substitutable_chars():
    row = dict(_BASE_ROW)
    row["description"] = "ABCDFGHIJKLMNOPQRSTUVWXYZ0123689"  # no "'\"- " or lowercase "e"

    corrupted, _ = corrupt_encoding(row, random.Random(1))

    assert len(corrupted["description"]) > len(row["description"])
    assert "Ã©â€™" in corrupted["description"]


def test_corrupt_truncation_shortens_and_appends_garble():
    row = dict(_BASE_ROW)
    corrupted, corruption_type = corrupt_truncation(row, random.Random(2))

    assert corruption_type == "truncation"
    assert len(corrupted["description"]) < len(_BASE_ROW["description"])
    assert row == _BASE_ROW


def test_corrupt_truncation_falls_back_to_name_when_description_is_null():
    row = dict(_BASE_ROW)
    row["description"] = None

    corrupted, _ = corrupt_truncation(row, random.Random(2))

    assert corrupted["name"] != _BASE_ROW["name"]
    assert corrupted["description"] is None


def test_corrupt_zip_state_mismatch_swaps_to_a_different_states_zip():
    zip3_to_state = {"100": "NY", "902": "CA", "331": "FL"}
    zip3_pool = {"100": ["10001", "10002"], "902": ["90210"], "331": ["33101"]}
    row = dict(_BASE_ROW)  # state=NY, zip=10001 (zip3 "100" -> NY)

    corrupted, corruption_type = corrupt_zip_state_mismatch(row, random.Random(3), zip3_to_state, zip3_pool)

    assert corruption_type == "zip_state_mismatch"
    new_zip3 = corrupted["zip_code"][:3]
    assert zip3_to_state[new_zip3] != row["state"]
    assert corrupted["state"] == row["state"]  # state left alone -- the mismatch is the point
    assert row == _BASE_ROW


def test_corrupt_null_storm_nulls_most_optional_fields_but_keeps_identity_fields():
    row = dict(_BASE_ROW)
    corrupted, corruption_type = corrupt_null_storm(row, random.Random(4))

    optional_fields = ["company_size", "state", "country", "city", "zip_code", "address", "description"]
    n_nulled = sum(1 for f in optional_fields if corrupted[f] is None)

    assert corruption_type == "null_storm"
    assert 5 <= n_nulled <= len(optional_fields)
    assert corrupted["company_id"] == row["company_id"]
    assert corrupted["name"] == row["name"]
    assert corrupted["url"] == row["url"]
    assert row == _BASE_ROW
