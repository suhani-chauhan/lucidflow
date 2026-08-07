import json

import polars as pl

from lucidflow.drift.reference_profile import (
    build_reference_profile,
    load_reference_profile,
    null_bucket_counts,
    save_reference_profile,
    value_counts,
)

_DF = pl.DataFrame(
    {
        "company_size": ["1", "1", "2", None],
        "state": ["NY", None, "CA", "CA"],
        "description": ["short", None, "a much longer description here", "medium length text"],
    }
)


def test_value_counts_includes_a_null_bucket():
    counts = value_counts(_DF["company_size"])

    assert counts == {"1": 2, "2": 1, "null": 1}


def test_null_bucket_counts():
    counts = null_bucket_counts(_DF["state"])

    assert counts == {"null": 1, "non_null": 3}


def test_build_reference_profile_excludes_null_descriptions_from_length_values():
    profile = build_reference_profile(_DF)

    assert profile["n_rows"] == 4
    assert profile["company_size_counts"] == {"1": 2, "2": 1, "null": 1}
    assert profile["state_null_counts"] == {"null": 1, "non_null": 3}
    assert len(profile["description_len_values"]) == 3  # the None description is excluded
    assert sorted(profile["description_len_values"]) == sorted([len("short"), len("a much longer description here"), len("medium length text")])


def test_save_and_load_reference_profile_roundtrips(tmp_path):
    profile = build_reference_profile(_DF)
    path = tmp_path / "reference_profile.json"

    save_reference_profile(profile, path)
    loaded = load_reference_profile(path)

    assert loaded == profile
    assert json.loads(path.read_text()) == profile
