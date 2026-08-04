import polars as pl

from lucidflow.cleaning.text_normalizer import normalize_text_columns


def test_collapses_internal_and_trailing_whitespace():
    df = pl.DataFrame({"name": ["  Acme   Corp  "]})

    result = normalize_text_columns(df, ["name"])

    assert result["name"].to_list() == ["Acme Corp"]


def test_normalizes_unicode_to_nfkc():
    decomposed = "café"  # "e" + combining acute accent (U+0301), not precomposed
    precomposed = "café"  # single "e with acute" codepoint (U+00E9)
    df = pl.DataFrame({"name": [decomposed]})

    result = normalize_text_columns(df, ["name"])

    assert result["name"].to_list() == [precomposed]


def test_preserves_casing():
    df = pl.DataFrame({"name": ["IBM"], "country": ["US"]})

    result = normalize_text_columns(df, ["name", "country"])

    assert result["name"].to_list() == ["IBM"]
    assert result["country"].to_list() == ["US"]


def test_blank_after_strip_becomes_null():
    df = pl.DataFrame({"name": ["   "]})

    result = normalize_text_columns(df, ["name"])

    assert result["name"].to_list() == [None]


def test_nulls_pass_through():
    df = pl.DataFrame({"name": [None]})

    result = normalize_text_columns(df, ["name"])

    assert result["name"].to_list() == [None]


def test_ignores_columns_not_present():
    df = pl.DataFrame({"name": ["Acme"]})

    result = normalize_text_columns(df, ["name", "not_a_real_column"])

    assert result.columns == ["name"]
