"""Statistical fingerprint extraction for a single column.

Every feature is computed from the column's values as strings, regardless of
how the source file happened to encode them — the point of this classifier is
to infer semantic meaning from the *shape* of the data, not from dtype hints
or header names.
"""

import math
import re
import statistics
from collections import Counter
from itertools import pairwise

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL_RE = re.compile(
    r"^(https?://|www\.)\S+$"  # explicit scheme or www prefix
    r"|^([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(/\S*)?$",  # bare domain, e.g. foo.example.com
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?$"  # ISO date/datetime
    r"|^\d{1,2}/\d{1,2}/\d{2,4}$"  # MM/DD/YYYY
)
_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")

FEATURE_NAMES = [
    "null_ratio",
    "cardinality_ratio",
    "distinct_count_log",
    "frac_alpha",
    "frac_digit",
    "frac_punct",
    "frac_space",
    "frac_upper_of_alpha",
    "avg_value_length",
    "std_value_length",
    "avg_token_length",
    "std_token_length",
    "avg_num_tokens",
    "email_hit_rate",
    "url_hit_rate",
    "date_hit_rate",
    "numeric_regex_hit_rate",
    "numeric_parse_rate",
    "digit_bigram_rate",
]


def _safe_std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def extract_column_features(raw_values: list[str | None]) -> dict:
    """Compute the statistical fingerprint for one column's values.

    `raw_values` should be the column's values as strings (or None for nulls) —
    exactly what you get from an all-Utf8 CSV read, before any type coercion.
    """
    total = len(raw_values)
    values = [v for v in raw_values if v is not None and v != ""]
    non_null = len(values)

    if non_null == 0:
        return {name: 0.0 for name in FEATURE_NAMES} | {"null_ratio": 1.0}

    null_ratio = 1 - (non_null / total) if total else 0.0
    distinct = len(set(values))
    cardinality_ratio = distinct / non_null

    char_counts = Counter()
    for v in values:
        char_counts.update(v)
    total_chars = sum(char_counts.values()) or 1
    alpha_chars = sum(c for ch, c in char_counts.items() if ch.isalpha())
    digit_chars = sum(c for ch, c in char_counts.items() if ch.isdigit())
    space_chars = sum(c for ch, c in char_counts.items() if ch.isspace())
    punct_chars = total_chars - alpha_chars - digit_chars - space_chars
    upper_chars = sum(c for ch, c in char_counts.items() if ch.isalpha() and ch.isupper())

    value_lengths = [len(v) for v in values]
    tokens = [tok for v in values for tok in v.split()]
    token_lengths = [len(t) for t in tokens] or [0]

    digit_bigrams = 0
    total_bigrams = 0
    for v in values:
        for a, b in pairwise(v):
            total_bigrams += 1
            if a.isdigit() and b.isdigit():
                digit_bigrams += 1

    email_hits = sum(1 for v in values if _EMAIL_RE.match(v))
    url_hits = sum(1 for v in values if _URL_RE.match(v))
    date_hits = sum(1 for v in values if _DATE_RE.match(v))
    numeric_regex_hits = sum(1 for v in values if _NUMERIC_RE.match(v))

    numeric_parse_hits = 0
    for v in values:
        try:
            float(v)
            numeric_parse_hits += 1
        except ValueError:
            pass

    return {
        "null_ratio": null_ratio,
        "cardinality_ratio": cardinality_ratio,
        "distinct_count_log": math.log1p(distinct),
        "frac_alpha": alpha_chars / total_chars,
        "frac_digit": digit_chars / total_chars,
        "frac_punct": punct_chars / total_chars,
        "frac_space": space_chars / total_chars,
        "frac_upper_of_alpha": (upper_chars / alpha_chars) if alpha_chars else 0.0,
        "avg_value_length": statistics.fmean(value_lengths),
        "std_value_length": _safe_std(value_lengths),
        "avg_token_length": statistics.fmean(token_lengths),
        "std_token_length": _safe_std(token_lengths),
        "avg_num_tokens": statistics.fmean([len(v.split()) for v in values]),
        "email_hit_rate": email_hits / non_null,
        "url_hit_rate": url_hits / non_null,
        "date_hit_rate": date_hits / non_null,
        "numeric_regex_hit_rate": numeric_regex_hits / non_null,
        "numeric_parse_rate": numeric_parse_hits / non_null,
        "digit_bigram_rate": (digit_bigrams / total_bigrams) if total_bigrams else 0.0,
    }


def sample_values(raw_values: list[str | None], n: int = 5) -> list[str]:
    seen = []
    for v in raw_values:
        if v is not None and v != "" and v not in seen:
            seen.append(v)
        if len(seen) >= n:
            break
    return seen
