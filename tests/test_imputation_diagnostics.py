import polars as pl

from lucidflow.models.imputation_selector.diagnostics import diagnose_missingness


def test_missingness_associated_with_a_predictor_is_flagged_mar():
    # target is null exactly when group == "B" -- a textbook MAR pattern.
    df = pl.DataFrame({"group": ["A"] * 30 + ["B"] * 30, "target": [1] * 30 + [None] * 30})

    result = diagnose_missingness(df, "target")

    assert result["verdict"] == "MAR"
    assoc = {a["predictor"]: a for a in result["associations"]}
    assert assoc["group"]["significant"] is True
    assert assoc["group"]["test"] == "chi2"


def test_missingness_independent_of_predictors_falls_back_to_mcar():
    # group alternates on period 2, nulls fall on period 3 -- coprime periods,
    # so missingness lands on group "A" and "B" in equal proportion by construction.
    n = 120
    group = ["A", "B"] * (n // 2)
    target = [None if i % 3 == 0 else "x" for i in range(n)]
    df = pl.DataFrame({"group": group, "target": target})

    result = diagnose_missingness(df, "target")

    assert result["verdict"].startswith("MCAR")
    assoc = {a["predictor"]: a for a in result["associations"]}
    assert assoc["group"]["significant"] is False


def test_high_cardinality_predictor_is_excluded_from_association_tests():
    n = 60
    df = pl.DataFrame({
        "near_unique": [f"id_{i}" for i in range(n)],
        "target": [1] * (n - 5) + [None] * 5,
    })

    result = diagnose_missingness(df, "target")

    tested_predictors = {a["predictor"] for a in result["associations"]}
    assert "near_unique" not in tested_predictors


def test_reports_missing_count_and_ratio():
    df = pl.DataFrame({"other": list(range(10)), "target": [1, 2, 3, None, None, 6, 7, 8, 9, 10]})

    result = diagnose_missingness(df, "target")

    assert result["missing_count"] == 2
    assert result["missing_ratio"] == 0.2
