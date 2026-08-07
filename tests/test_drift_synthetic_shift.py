import random

import polars as pl

from lucidflow.drift.synthetic_shift import (
    shift_company_size,
    shift_description_length,
    shift_state_null_rate,
)

_BASE_DF = pl.DataFrame(
    {
        "company_size": [str(n) for n in ([1, 2, 3, 4, 5] * 20)],
        "state": (["NY", "CA"] * 50),
        "description": [f"A description of length {n}" * 5 for n in range(100)],
    }
)


def test_shift_company_size_zero_magnitude_is_a_no_op():
    shifted = shift_company_size(_BASE_DF, 0.0, random.Random(1))

    assert shifted["company_size"].to_list() == _BASE_DF["company_size"].to_list()


def test_shift_company_size_changes_roughly_the_requested_fraction():
    original = _BASE_DF["company_size"].to_list()
    shifted = shift_company_size(_BASE_DF, 0.5, random.Random(1))["company_size"].to_list()

    n_changed = sum(1 for o, s in zip(original, shifted) if o != s)

    # not every "changed" row is guaranteed to differ (the skewed re-draw can land on the
    # same code by chance), so this is an upper-bound sanity check, not an exact count.
    assert n_changed <= 50
    assert n_changed > 0


def test_shift_state_null_rate_zero_magnitude_is_a_no_op():
    shifted = shift_state_null_rate(_BASE_DF, 0.0, random.Random(1))

    assert shifted["state"].to_list() == _BASE_DF["state"].to_list()


def test_shift_state_null_rate_nulls_the_requested_row_count():
    shifted = shift_state_null_rate(_BASE_DF, 0.3, random.Random(1))

    n_null = sum(1 for v in shifted["state"].to_list() if v is None)

    assert n_null == 30


def test_shift_description_length_zero_magnitude_is_a_no_op():
    shifted = shift_description_length(_BASE_DF, 0.0, random.Random(1))

    assert shifted["description"].to_list() == _BASE_DF["description"].to_list()


def test_shift_description_length_shortens_the_requested_row_count():
    original = _BASE_DF["description"].to_list()
    shifted = shift_description_length(_BASE_DF, 0.4, random.Random(1))["description"].to_list()

    n_shortened = sum(1 for o, s in zip(original, shifted) if len(s) < len(o))

    assert n_shortened == 40


def test_shift_description_length_never_lengthens_a_row():
    original = _BASE_DF["description"].to_list()
    shifted = shift_description_length(_BASE_DF, 1.0, random.Random(1))["description"].to_list()

    assert all(len(s) <= len(o) for o, s in zip(original, shifted))
