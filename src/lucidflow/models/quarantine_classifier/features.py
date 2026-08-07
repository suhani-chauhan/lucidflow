"""Row-level feature extraction for the quarantine classifier.

Deliberately generic quality signals (text shape, character composition,
missingness) rather than one detector per corruption type — the point of
the exercise is to see whether a classifier trained on these can actually
recover types e-h, not to hand it a bespoke feature per type. The one
exception is `zip_state_mismatch`, which is inherently specific to type g
(cross-field geographic consistency isn't derivable from generic text
shape) and is computed via the same empirical reference used to inject
that corruption — see `zip_state_reference.py`.
"""

import re
import string

from pydantic import ValidationError

from lucidflow.models.quarantine_classifier.zip_state_reference import US_STATE_ABBREVS
from lucidflow.validation.pydantic_models import Company

OPTIONAL_FIELDS = ["company_size", "state", "country", "city", "zip_code", "address", "description"]

FEATURE_NAMES = [
    "null_count",
    "name_len",
    "description_len",
    "address_len",
    "city_len",
    "url_len",
    "description_frac_alpha",
    "description_frac_digit",
    "description_frac_punct",
    "name_mojibake_rate",
    "description_mojibake_rate",
    "suspicious_char_rate",
    "ends_without_terminal_punct",
    "trailing_char_class_diversity",
    "zip_state_mismatch",
    "contract_violation_count",
]

# Shape-only features fed to Isolation Forest -- kept separate from FEATURE_NAMES since
# the IF score itself (not these raw inputs) is what gets appended as a classifier feature.
ISO_FOREST_INPUT_FEATURES = [
    "null_count", "name_len", "description_len", "address_len", "city_len", "url_len",
]

_TEXT_WHITELIST = set(string.ascii_letters + string.digits + string.punctuation + string.whitespace)

# UTF-8-as-cp1252 mojibake produces one of two shapes: a 2-byte sequence (Ã/Â followed by
# whatever the continuation byte decodes to under cp1252) or a 3-byte sequence (â€ followed
# by whatever the third byte decodes to). Building the continuation-byte character set from
# the actual cp1252 table -- rather than hardcoding the handful of substitutions this project's
# own corrupt_encoding() happens to use -- means this also fires on real-world mojibake shapes
# (Ã¨, Ã±, Ã¼, â€”, â€¦, ...) it was never shown during injection, not just literal string matches.
_MOJIBAKE_CONTINUATION_CHARS = "".join(
    bytes([b]).decode("cp1252", errors="ignore") for b in range(0x80, 0xC0)
)
_MOJIBAKE_PATTERN = re.compile(
    f"[ÃÂ][{re.escape(_MOJIBAKE_CONTINUATION_CHARS)}]|â€[{re.escape(_MOJIBAKE_CONTINUATION_CHARS)}]"
)


def _frac(s: str, predicate) -> float:
    return sum(1 for c in s if predicate(c)) / len(s) if s else 0.0


def _mojibake_rate(s: str) -> float:
    return len(_MOJIBAKE_PATTERN.findall(s)) / len(s) if s else 0.0


def _contract_violation_count(row: dict) -> int:
    try:
        Company.model_validate(row)
        return 0
    except ValidationError as exc:
        return len(exc.errors())


def extract_row_features(row: dict, zip3_to_state: dict[str, str]) -> dict:
    name = row.get("name") or ""
    description = row.get("description") or ""
    address = row.get("address") or ""
    city = row.get("city") or ""
    url = row.get("url") or ""

    null_count = sum(1 for f in OPTIONAL_FIELDS if row.get(f) is None)

    combined = name + description
    suspicious_char_rate = _frac(combined, lambda c: c not in _TEXT_WHITELIST)

    ends_without_terminal_punct = 0
    if description:
        stripped = description.rstrip()
        ends_without_terminal_punct = 0 if stripped and stripped[-1] in ".!?\")'" else 1

    trailing = description[-8:] if description else ""
    classes = set()
    for c in trailing:
        if c.islower():
            classes.add("lower")
        elif c.isupper():
            classes.add("upper")
        elif c.isdigit():
            classes.add("digit")
        elif not c.isspace():
            classes.add("punct")

    zip_state_mismatch = -1  # unknown / not applicable (non-US, or state/zip3 not in reference)
    state, zip_code, country = row.get("state"), row.get("zip_code"), row.get("country")
    if country == "US" and state and zip_code:
        state_clean = str(state).strip().upper()
        zip3 = str(zip_code)[:3]
        if zip3 in zip3_to_state and state_clean in US_STATE_ABBREVS:
            zip_state_mismatch = 0 if zip3_to_state[zip3] == state_clean else 1

    return {
        "null_count": null_count,
        "name_len": len(name),
        "description_len": len(description),
        "address_len": len(address),
        "city_len": len(city),
        "url_len": len(url),
        "description_frac_alpha": _frac(description, str.isalpha),
        "description_frac_digit": _frac(description, str.isdigit),
        "description_frac_punct": _frac(description, lambda c: c in string.punctuation),
        "name_mojibake_rate": _mojibake_rate(name),
        "description_mojibake_rate": _mojibake_rate(description),
        "suspicious_char_rate": suspicious_char_rate,
        "ends_without_terminal_punct": ends_without_terminal_punct,
        "trailing_char_class_diversity": len(classes),
        "zip_state_mismatch": zip_state_mismatch,
        "contract_violation_count": _contract_violation_count(row),
    }
