import polars as pl

from lucidflow.models.imputation_selector.selector import run_missingness_engine


def _synthetic_companies_df() -> pl.DataFrame:
    company_size = [1] * 40 + [5] * 40 + [None] * 20
    country = ["US"] * 45 + ["GB"] * 30 + [None] * 25
    state = ["CA"] * 40 + ["NY"] * 30 + ["LDN"] * 20 + [None] * 10
    city = ["LA"] * 38 + [None] * 2 + ["NYC"] * 30 + ["London"] * 18 + [None] * 2 + [None] * 10
    zip_code = ["90001"] * 38 + [None] * 2 + ["10001"] * 30 + ["SW1"] * 18 + [None] * 2 + [None] * 10
    description = ["a real description"] * 90 + [None] * 10
    address = ["1 Main St"] * 90 + [None] * 10
    extra_unhandled = ["x"] * 95 + [None] * 5

    df = pl.DataFrame({
        "company_size": company_size,
        "country": country,
        "state": state,
        "city": city,
        "zip_code": zip_code,
        "description": description,
        "address": address,
        "extra_unhandled": extra_unhandled,
    })
    return df.with_columns(pl.col("company_size").cast(pl.Int64, strict=False))


def test_benchmarked_columns_get_fully_imputed(tmp_path):
    df = _synthetic_companies_df()

    imputed, _ = run_missingness_engine(df, artifact_dir=tmp_path)

    assert imputed["company_size"].null_count() == 0
    assert imputed["country"].null_count() == 0


def test_group_mode_columns_get_fully_imputed(tmp_path):
    df = _synthetic_companies_df()

    imputed, _ = run_missingness_engine(df, artifact_dir=tmp_path)

    assert imputed["state"].null_count() == 0
    assert imputed["city"].null_count() == 0
    assert imputed["zip_code"].null_count() == 0


def test_skipped_columns_are_left_null(tmp_path):
    df = _synthetic_companies_df()

    imputed, report = run_missingness_engine(df, artifact_dir=tmp_path)

    assert imputed["description"].null_count() == 10
    assert imputed["address"].null_count() == 10
    entries = {e["column"]: e for e in report}
    assert entries["description"]["decision"].startswith("skipped")
    assert entries["address"]["decision"].startswith("skipped")


def test_unconfigured_column_with_nulls_is_flagged_not_silently_handled(tmp_path):
    df = _synthetic_companies_df()

    imputed, report = run_missingness_engine(df, artifact_dir=tmp_path)

    assert imputed["extra_unhandled"].null_count() == 5  # left untouched
    entry = next(e for e in report if e["column"] == "extra_unhandled")
    assert "UNHANDLED" in entry["decision"]


def test_benchmarked_columns_persist_a_fitted_artifact(tmp_path):
    df = _synthetic_companies_df()

    run_missingness_engine(df, artifact_dir=tmp_path)

    assert (tmp_path / "company_size_imputer.joblib").exists()
    assert (tmp_path / "country_imputer.joblib").exists()


def test_group_mode_columns_persist_a_lookup_artifact(tmp_path):
    df = _synthetic_companies_df()

    run_missingness_engine(df, artifact_dir=tmp_path)

    assert (tmp_path / "state_group_mode.joblib").exists()
    assert (tmp_path / "city_group_mode.joblib").exists()
    assert (tmp_path / "zip_code_group_mode.joblib").exists()


def test_benchmarked_column_report_includes_class_coverage_and_limitation_note(tmp_path):
    # company_size gets a third class with a single known row -- structurally
    # untestable (can't both train on and hold out one example), so it should
    # surface as a documented limitation, not silently vanish from the report.
    company_size = [1] * 40 + [5] * 40 + [7] * 1 + [None] * 20
    country = ["US"] * 46 + ["GB"] * 35 + [None] * 20
    df = pl.DataFrame({
        "company_size": company_size,
        "country": country,
        "state": ["CA"] * 101,
        "city": ["LA"] * 101,
        "zip_code": ["90001"] * 101,
    }).with_columns(pl.col("company_size").cast(pl.Int64, strict=False))

    _, report = run_missingness_engine(df, artifact_dir=tmp_path)

    entries = {e["column"]: e for e in report}
    size_coverage = entries["company_size"]["class_coverage"]
    assert size_coverage["total_classes"] == 3
    assert size_coverage["singleton_classes"] == 1
    assert size_coverage["evaluated_classes"] == 2
    assert "known_limitation" in entries["company_size"]
    assert "structural limitation" in entries["company_size"]["known_limitation"].lower()

    # country has no singleton classes in this fixture -> no limitation note
    assert entries["country"]["class_coverage"]["singleton_classes"] == 0
    assert "known_limitation" not in entries["country"]


def test_report_includes_diagnosis_and_decision_for_every_null_column(tmp_path):
    df = _synthetic_companies_df()

    _, report = run_missingness_engine(df, artifact_dir=tmp_path)

    reported_columns = {e["column"] for e in report}
    assert reported_columns == {
        "company_size", "country", "state", "city", "zip_code", "description", "address", "extra_unhandled",
    }
    for entry in report:
        assert "diagnosis" in entry
        assert "decision" in entry
