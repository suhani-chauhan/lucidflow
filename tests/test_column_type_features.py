from lucidflow.models.column_type_classifier.features import extract_column_features


def test_identifier_column_is_unique_numeric():
    values = ["1009", "1016", "1025", "1028", "1033"]

    features = extract_column_features(values)

    assert features["cardinality_ratio"] == 1.0
    assert features["frac_digit"] == 1.0
    assert features["numeric_parse_rate"] == 1.0
    assert features["std_value_length"] == 0.0


def test_categorical_column_has_low_cardinality_fixed_vocabulary():
    values = ["FULL_TIME", "PART_TIME", "FULL_TIME", "CONTRACT", "FULL_TIME", "PART_TIME", "FULL_TIME", "CONTRACT"]

    features = extract_column_features(values)

    assert features["cardinality_ratio"] == 0.375
    assert features["frac_alpha"] > 0.8
    assert features["numeric_parse_rate"] == 0.0
    assert features["avg_num_tokens"] == 1.0


def test_free_text_column_has_high_cardinality_and_many_tokens():
    values = [
        "A cloud infrastructure company focused on developer tooling.",
        "We build machine learning platforms for enterprise customers worldwide.",
        "Manufacturer of industrial sensors and IoT devices for factories.",
    ]

    features = extract_column_features(values)

    assert features["cardinality_ratio"] == 1.0
    assert features["avg_num_tokens"] > 5
    assert features["avg_value_length"] > 40


def test_numeric_continuous_column_parses_fully_with_high_cardinality():
    values = ["52000.0", "61000.5", "48000.25", "73000.0", "55250.75"]

    features = extract_column_features(values)

    assert features["numeric_parse_rate"] == 1.0
    assert features["cardinality_ratio"] == 1.0
    assert features["frac_digit"] > 0.8


def test_geographic_column_is_alpha_short_tokens():
    values = ["New York", "California", "Texas", "Illinois", "Ohio"]

    features = extract_column_features(values)

    assert features["frac_alpha"] > 0.9
    assert features["numeric_parse_rate"] == 0.0
    assert features["url_hit_rate"] == 0.0
    assert features["date_hit_rate"] == 0.0


def test_date_column_hits_the_date_regex():
    values = ["2021-01-05", "2021-03-12", "2021-07-19", "2021-11-02"]

    features = extract_column_features(values)

    assert features["date_hit_rate"] == 1.0
    assert features["numeric_parse_rate"] == 0.0  # ISO dates don't parse as bare floats


def test_url_column_hits_the_url_regex_including_bare_www_domain():
    values = ["https://example.com/a", "http://foo.org/x", "www.bar.net/page", "https://baz.io/path?x=1"]

    features = extract_column_features(values)

    assert features["url_hit_rate"] == 1.0


def test_boolean_column_is_low_cardinality_numeric_0_1():
    values = ["0", "1", "0", "0", "1"]

    features = extract_column_features(values)

    assert features["cardinality_ratio"] == 0.4
    assert features["numeric_parse_rate"] == 1.0
    assert features["avg_value_length"] == 1.0


def test_null_only_column_returns_null_ratio_one_and_zeroed_features():
    features = extract_column_features([None, None, ""])

    assert features["null_ratio"] == 1.0
    assert features["cardinality_ratio"] == 0.0
    assert features["numeric_parse_rate"] == 0.0
