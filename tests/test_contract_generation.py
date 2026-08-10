import ast
from pathlib import Path

import pytest

from lucidflow.contract_generation.type_mapping import (
    FieldSpec,
    _looks_all_int,
    _looks_like_boolean_text,
    _looks_like_ordinal_candidate,
    _safe_comment,
    build_field_spec,
)

MODEL_PATH = Path("src/lucidflow/models/column_type_classifier/column_type_classifier.joblib")
pytestmark_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model artifact not present")


# --- pure helper functions (type_mapping.py) --------------------------------


def test_looks_all_int_true_for_pure_integers():
    assert _looks_all_int(["1", "2", "-3", "007"])


def test_looks_all_int_false_if_any_value_is_not_integer():
    assert not _looks_all_int(["1", "2.5", "3"])


def test_looks_all_int_false_for_empty():
    assert not _looks_all_int([])


def test_looks_like_ordinal_candidate_matches_company_size_shape():
    values = [str(i) for i in range(1, 8)] * 5  # 1-7, dense, repeated
    assert _looks_like_ordinal_candidate(values)


def test_looks_like_ordinal_candidate_false_for_high_cardinality_integers():
    # e.g. a numeric identifier column shouldn't be flagged as ordinal
    values = [str(i) for i in range(1000, 1100)]
    assert not _looks_like_ordinal_candidate(values)


def test_looks_like_ordinal_candidate_false_for_non_integers():
    assert not _looks_like_ordinal_candidate(["a", "b", "c"])


def test_looks_like_boolean_text_true_false():
    assert _looks_like_boolean_text(["true", "false", "True", "FALSE"])


def test_looks_like_boolean_text_yes_no():
    assert _looks_like_boolean_text(["yes", "no", "yes"])


def test_looks_like_boolean_text_false_for_unrelated_categorical():
    assert not _looks_like_boolean_text(["red", "green", "blue"])


def test_safe_comment_collapses_embedded_newlines():
    # A raw CSV value could contain a newline; embedding it verbatim in a
    # generated `# comment` line would let it break out into a new source line.
    malicious = "normal\nimport os; os.system('x')"
    result = _safe_comment(malicious)
    assert "\n" not in result
    assert "normal" in result and "import os" in result


def test_safe_comment_truncates_long_text():
    long_text = "x" * 500
    result = _safe_comment(long_text)
    assert len(result) <= 161  # 160 + ellipsis char
    assert result.endswith("…")


# --- build_field_spec, per predicted type ------------------------------------


def test_identifier_all_integer_values_maps_to_required_int():
    spec = build_field_spec("company_id", "identifier", 0.95, ["1", "2", "3"])
    assert spec.python_type == "int"
    assert spec.required is True
    assert not spec.risk_flags


def test_identifier_non_integer_values_maps_to_required_str():
    spec = build_field_spec("external_ref", "identifier", 0.95, ["AB-1", "AB-2"])
    assert spec.python_type == "str"
    assert spec.required is True


def test_identifier_with_observed_nulls_flags_risk():
    spec = build_field_spec("company_id", "identifier", 0.95, ["1", None, "3"])
    assert spec.required is True
    assert any("observed null" in flag for flag in spec.risk_flags)


def test_categorical_lists_observed_values_as_comment():
    spec = build_field_spec("size", "categorical", 0.9, ["a", "b", "a", "c"])
    assert spec.python_type == "str"
    assert spec.required is False
    assert any("Observed values" in c for c in spec.comments)


def test_categorical_high_cardinality_does_not_list_all_values():
    values = [f"v{i}" for i in range(50)]
    spec = build_field_spec("col", "categorical", 0.9, values)
    assert any("too many to list" in c for c in spec.comments)


def test_categorical_ordinal_shaped_values_flagged_not_retyped():
    values = [str(i) for i in range(1, 8)] * 5
    spec = build_field_spec("company_size", "categorical", 0.9, values)
    assert spec.python_type == "str"  # type unchanged
    assert any("ordinal" in flag for flag in spec.risk_flags)


def test_categorical_boolean_shaped_values_flagged_not_retyped():
    values = ["true", "false", "true"] * 5
    spec = build_field_spec("status", "categorical", 0.9, values)
    assert spec.python_type == "str"  # type unchanged
    assert any("possible boolean" in flag for flag in spec.risk_flags)


def test_free_text_is_unconstrained_optional_str():
    spec = build_field_spec("description", "free_text", 0.9, ["hello", "world"])
    assert spec.python_type == "str"
    assert spec.required is False
    assert not spec.constraints


def test_geographic_is_unconstrained_optional_str():
    spec = build_field_spec("state", "geographic", 0.9, ["CA", "New York", "??"])
    assert spec.python_type == "str"
    assert not spec.constraints


def test_date_iso_values_map_to_date_type():
    spec = build_field_spec("created_at", "date", 0.9, ["2020-01-01", "2021-06-15"] * 10)
    assert spec.python_type == "date"
    assert spec.required is False


def test_date_non_iso_values_fall_back_to_str():
    spec = build_field_spec("posted", "date", 0.9, ["01/15/2020", "06/01/2021"] * 10)
    assert spec.python_type == "str"
    assert any("MM/DD/YYYY" in flag for flag in spec.risk_flags)


def test_date_unparseable_values_fall_back_to_str_with_suspicion_flag():
    spec = build_field_spec("weird", "date", 0.9, ["not-a-date", "also-not"] * 10)
    assert spec.python_type == "str"
    assert any("treat this prediction with suspicion" in flag for flag in spec.risk_flags)


def test_url_uses_permissive_pattern_from_features_module():
    from lucidflow.models.column_type_classifier.features import URL_PATTERN

    spec = build_field_spec("homepage", "url", 0.9, ["https://example.com"])
    assert spec.constraints["pattern"] == URL_PATTERN


def test_boolean_with_coercible_values_stays_bool_type():
    spec = build_field_spec("is_active", "boolean", 0.95, ["0", "1", "0"])
    assert spec.python_type == "bool"
    assert any("0/1-coded" in flag for flag in spec.risk_flags)


def test_boolean_always_carries_the_blind_spot_risk_flag_even_when_kept_as_bool():
    spec = build_field_spec("is_active", "boolean", 0.95, ["0", "1", "0"])
    assert any("0/1-coded" in flag for flag in spec.risk_flags)


def test_boolean_with_two_distinct_non_coercible_values_falls_back_to_str():
    # 2 distinct values doesn't trip the definitional-contradiction downgrade, but they
    # still aren't Pydantic-bool-parseable, so the boolean branch's own fallback must fire.
    spec = build_field_spec("weird_flag", "boolean", 0.7, ["2", "5", "2", "5"])
    assert spec.python_type == "str"
    assert any("NOT all Pydantic-bool-parseable" in flag for flag in spec.risk_flags)


def test_boolean_prediction_with_more_than_two_distinct_values_downgrades_to_categorical():
    # Real case found in Task 2: company_size (values "1".."7") predicted boolean at low
    # confidence -- a definitional contradiction (boolean allows at most 2 distinct values),
    # not just a low-confidence guess.
    spec = build_field_spec("company_size", "boolean", 0.48, ["1", "2", "3", "4", "5", "6", "7"])
    assert spec.python_type == "str"
    assert any("definitional contradiction" in flag for flag in spec.risk_flags)
    # Model 1's actual raw prediction is preserved for transparency, even though the field
    # was generated using categorical treatment.
    assert spec.predicted_type == "boolean"


def test_boolean_downgrade_lets_ordinal_advisory_fire():
    # The whole point of running the downgrade before the ordinal check: company_size-shaped
    # data (small, dense integer codes) predicted as boolean should still get flagged as a
    # possible ordinal code, exactly as it would if Model 1 had said "categorical" directly.
    values = [str(i) for i in range(1, 8)] * 5
    spec = build_field_spec("company_size", "boolean", 0.48, values)
    assert any("possible ordinal-coded numeric" in flag for flag in spec.risk_flags)


def test_boolean_prediction_with_exactly_two_distinct_values_does_not_downgrade():
    spec = build_field_spec("is_active", "boolean", 0.95, ["0", "1", "0"])
    assert spec.python_type == "bool"
    assert not any("definitional contradiction" in flag for flag in spec.risk_flags)
    assert any("0/1-coded" in flag for flag in spec.risk_flags)  # blind-spot note still present


def test_boolean_with_no_non_null_values_stays_bool_type():
    spec = build_field_spec("is_active", "boolean", 0.95, [None, None])
    assert spec.python_type == "bool"


def test_numeric_continuous_reports_observed_range_as_soft_comment():
    spec = build_field_spec("salary", "numeric_continuous", 0.9, ["100.5", "200", "50.25"])
    assert spec.python_type == "float"
    assert spec.constraints == {}
    assert any("Observed range" in c for c in spec.comments)


def test_low_confidence_prediction_is_flagged_regardless_of_type():
    spec = build_field_spec("mystery", "free_text", 0.3, ["a", "b"])
    assert any("low-confidence" in flag for flag in spec.risk_flags)


def test_unknown_predicted_type_raises():
    with pytest.raises(ValueError):
        build_field_spec("col", "not_a_real_type", 0.9, ["a"])


# --- generator: identifier sanitization + source rendering ------------------


def test_sanitize_identifier_handles_spaces_and_punctuation():
    from lucidflow.contract_generation.generate_contract import _sanitize_identifier

    taken = set()
    assert _sanitize_identifier("company size!", taken) == "company_size_"


def test_sanitize_identifier_dedupes_collisions():
    from lucidflow.contract_generation.generate_contract import _sanitize_identifier

    taken = set()
    a = _sanitize_identifier("a-b", taken)
    b = _sanitize_identifier("a.b", taken)
    assert a != b


def test_sanitize_identifier_avoids_python_keywords():
    from lucidflow.contract_generation.generate_contract import _sanitize_identifier

    taken = set()
    assert _sanitize_identifier("class", taken) == "class_"


def test_render_source_produces_syntactically_valid_python():
    from lucidflow.contract_generation.generate_contract import render_source

    specs = [
        FieldSpec(column="company_id", predicted_type="identifier", confidence=0.9, python_type="int", required=True),
        FieldSpec(
            column="name",
            predicted_type="free_text",
            confidence=0.8,
            python_type="str",
            required=False,
            comments=["some comment"],
        ),
    ]
    source = render_source("Company", specs, aliases={})
    ast.parse(source)  # raises SyntaxError if invalid
    assert "class Company(BaseModel):" in source


def test_render_source_with_malicious_embedded_newline_stays_valid_python():
    from lucidflow.contract_generation.generate_contract import render_source

    evil_alias = 'name\nimport os; os.system("pwn")  # '
    specs = [
        FieldSpec(column="name_col", predicted_type="free_text", confidence=0.8, python_type="str", required=False)
    ]
    source = render_source("Evil", specs, aliases={"name_col": evil_alias})
    ast.parse(source)  # would raise if the alias broke out of its string literal


# --- end-to-end generation against the real trained model -------------------


@pytestmark_model
def test_generate_contract_end_to_end_produces_usable_model():
    from lucidflow.contract_generation.generate_contract import generate_contract

    columns = {
        "id": [str(i) for i in range(1, 51)],
        "homepage": ["https://example.com/a", "http://foo.bar/baz", "www.example.org"] * 17,
        "notes": ["some free text about a company"] * 50,
    }
    result = generate_contract(columns, class_name="Demo")

    ast.parse(result.source)
    assert len(result.field_specs) == 3

    id_spec = next(s for s in result.field_specs if s.column == "id")
    assert id_spec.required is True

    # The model must actually validate a real row -- proves the generated
    # contract isn't just syntactically valid but functionally usable.
    row = {"id": "5", "homepage": "https://example.com", "notes": "hello"}
    validated = result.model.model_validate(row)
    assert validated.id == 5


@pytestmark_model
def test_generate_contract_sanitizes_and_aliases_unusual_headers():
    from lucidflow.contract_generation.generate_contract import generate_contract

    columns = {
        "weird header!": ["a", "b", "c"] * 10,
        "id": [str(i) for i in range(1, 31)],
    }
    result = generate_contract(columns, class_name="Weird")

    assert result.aliases  # at least one column got sanitized + aliased
    sanitized_name, original = next(iter(result.aliases.items()))
    assert original == "weird header!"

    # populate_by_name=True means the sanitized field name works even though
    # its alias is the original (unsanitized) header.
    row = {sanitized_name: "a", "id": "1"}
    validated = result.model.model_validate(row)
    assert getattr(validated, sanitized_name) == "a"
