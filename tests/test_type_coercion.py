import polars as pl

from lucidflow.cleaning.type_coercion import coerce_types


def test_casts_specified_columns():
    df = pl.DataFrame({"company_size": ["3.0", "5.0", None], "name": ["a", "b", "c"]})

    result = coerce_types(df, {"company_size": pl.Float64})

    assert result["company_size"].dtype == pl.Float64
    assert result["company_size"].to_list() == [3.0, 5.0, None]


def test_leaves_unlisted_columns_untouched():
    df = pl.DataFrame({"company_id": [1, 2], "name": ["a", "b"]})

    result = coerce_types(df, {"company_id": pl.Int64})

    assert result["name"].dtype == pl.Utf8


def test_ignores_columns_not_present_in_schema():
    df = pl.DataFrame({"company_id": [1, 2]})

    result = coerce_types(df, {"company_id": pl.Int64, "not_a_real_column": pl.Utf8})

    assert result.columns == ["company_id"]


def test_unparseable_values_become_null_not_an_error():
    df = pl.DataFrame({"company_size": ["not-a-number"]})

    result = coerce_types(df, {"company_size": pl.Int64})

    assert result["company_size"].to_list() == [None]
