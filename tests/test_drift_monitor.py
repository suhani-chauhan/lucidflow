import polars as pl

from lucidflow.drift.monitor import check_drift
from lucidflow.drift.reference_profile import build_reference_profile

_REFERENCE_DF = pl.DataFrame(
    {
        "company_size": ([str(n) for n in [1, 2, 3, 4, 5, 6, 7]] * 100),
        "state": (["NY"] * 690 + [None] * 10),
        "description": ([f"description number {n} padded out a bit" for n in range(700)]),
    }
)


def _profile():
    return build_reference_profile(_REFERENCE_DF)


def test_check_drift_reports_no_shift_against_an_identical_batch():
    report = check_drift(_profile(), _REFERENCE_DF)

    assert report["company_size"]["severity"] == "none"
    assert report["state_null_rate"]["severity"] == "none"
    assert report["description_len"]["severity"] == "none"
    assert report["any_flagged"] is False


def test_check_drift_flags_a_clear_categorical_shift():
    shifted = _REFERENCE_DF.with_columns(pl.lit("1").alias("company_size"))  # every row now code "1"

    report = check_drift(_profile(), shifted)

    assert report["company_size"]["metric"] == "psi"
    assert report["company_size"]["severity"] == "significant"
    assert report["any_flagged"] is True


def test_check_drift_flags_a_clear_null_rate_shift():
    shifted = _REFERENCE_DF.with_columns(pl.lit(None).alias("state"))  # every row now null

    report = check_drift(_profile(), shifted)

    assert report["state_null_rate"]["severity"] == "significant"
    assert report["any_flagged"] is True


def test_check_drift_flags_a_clear_numeric_length_shift():
    shifted = _REFERENCE_DF.with_columns(pl.lit("x").alias("description"))  # every description now length 1

    report = check_drift(_profile(), shifted)

    assert report["description_len"]["metric"] == "ks"
    assert report["description_len"]["severity"] == "significant"
    assert report["any_flagged"] is True
