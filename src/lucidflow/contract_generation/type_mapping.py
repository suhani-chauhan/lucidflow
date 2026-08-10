"""Task 1: maps Model 1's predicted semantic type for a column to a Pydantic
field template.

Model 1 (`lucidflow.models.column_type_classifier`) was trained on 8 semantic
types: identifier, categorical, free_text, geographic, date, url, boolean,
numeric_continuous. There is no `numeric_ordinal_code` type -- `company_size`
was originally proposed as one during labeling (see `proposed_labels.csv`),
but the human-confirmed label that actually went into training is
`categorical` (its ordinal-ness is instead hard-coded in the imputation
selector, outside the model -- see `models/imputation_selector/selector.py`).
So this mapping only covers the 8 real output classes; a company_size-like
column comes back as `categorical` here too, same as it does everywhere else
downstream of Model 1. What this module adds on top is advisory-only: it
flags a `categorical` column as a *possible* ordinal-code candidate when its
observed values are all small, dense integers -- without changing the
generated type -- so a human can make the same call that was originally made
for company_size, instead of the generator silently asserting an order the
data can't prove. Same treatment for Model 1's other documented blind spot:
a `categorical` column whose values look like true/false text gets flagged
as a possible mislabeled boolean (see
tests/test_column_type_classifier_model.py::test_true_false_text_boolean_column_is_misclassified_as_categorical),
without the generator silently "fixing" the model's output.

A `categorical` prediction is the common path into that ordinal advisory, but
not the only one: a `boolean` prediction with more than 2 distinct non-null
values is a direct contradiction of what "boolean" means, not just a
low-confidence guess (booleans have at most 2 states). `build_field_spec`
catches this before dispatching -- auto-downgrades the field-shape logic to
categorical treatment (the raw Model 1 prediction is still preserved on the
`FieldSpec` and still shown in the generated comment) -- so a case like
company_size (predicted `boolean` at low confidence, 7 distinct integer
codes) gets the same ordinal-candidate advisory a direct `categorical`
prediction would, instead of silently missing it. Found on real data in Task
2 -- see `reports/companies_csv_comparison.md`.

| Model 1 type         | Field template                                              |
|-----------------------|--------------------------------------------------------------|
| identifier             | required; `int` if all observed values parse as integers, else `str` |
| categorical            | `Optional[str]`; observed value set noted as a comment, not a hard `Enum` (an unseen category shouldn't hard-fail validation) |
| free_text              | `Optional[str]`, unconstrained |
| geographic              | `Optional[str]`, unconstrained (real geographic data is genuinely mixed-format -- same reasoning as the `state` column's mode-within-country fallback in the imputation selector, not a strict format) |
| date                   | `date` if observed values parse cleanly as ISO 8601, else `Optional[str]` (date-shaped-but-not-ISO and non-date-shaped both fall back the same way -- the two cases get different risk-flag wording in `build_field_spec`, so the report can tell them apart) |
| url                    | `Optional[str]`, permissive pattern -- reuses `features.URL_PATTERN` verbatim, the same bare-domain-inclusive regex from the Phase 2 fix, so the generated contract isn't stricter than what Model 1 itself was scored against |
| boolean                | `Optional[bool]` if every observed value is one Pydantic actually coerces to bool, else `Optional[str]` -- **flagged every time regardless**: Model 1's only boolean training examples are 0/1-coded, so it has never learned text true/false, and a `boolean` prediction on a text column should be treated with more suspicion than the other types. (The str fallback was added after Task 2 testing showed `bool` is not a soft type the way the others are -- a wrong prediction hard-rejects every row whose value isn't in Pydantic's fixed coercion set, instead of just being "worth a second look".) |
| numeric_continuous     | `Optional[float]`; observed min/max noted as a comment (soft reference), not a hard `ge`/`le` validation failure |

`required=True` is only ever assigned to `identifier` fields, per the mapping
table above -- it's the one type where "must be present" is part of what the
label means, not a hard-coded guess about any particular dataset. Every other
type is `Optional` even if a column happens to have zero observed nulls,
because "this dataset's sample has no nulls yet" isn't the same claim as "this
field is structurally required" (see the `url` field in the hand-written
Phase 1 contract for a case where a human made that judgment call from domain
knowledge Model 1 doesn't have -- documented in the Task 2 comparison report,
not silently reproduced here).
"""

import re
from dataclasses import dataclass, field
from datetime import date as _date

from lucidflow.models.column_type_classifier.features import URL_PATTERN

# Below this confidence, the winning label barely beat the runner-up -- worth a
# human glance regardless of which type it is. Chosen as a reasonable cutover
# point (8 classes -> uniform-random confidence would be ~0.125), not derived
# from any calibration study; treat it as a coarse triage signal, not a proof.
LOW_CONFIDENCE_THRESHOLD = 0.6

# "Small, dense" enough that listing/eyeballing the codes is meaningful --
# matches company_size's real shape (7 codes, 1-7) with headroom.
_MAX_ORDINAL_CANDIDATE_CODES = 20

# How many distinct categorical values to embed as a reference comment before
# falling back to "too many to list". Keeps generated files from ballooning on
# high-cardinality columns Model 1 still (correctly) called categorical.
_MAX_LISTED_CATEGORIES = 20

_INT_RE = re.compile(r"-?\d+")
_TRUE_FALSE_TEXT_SETS = [
    {"true", "false"},
    {"yes", "no"},
    {"y", "n"},
    {"t", "f"},
]

KNOWN_TYPES = frozenset(
    {
        "identifier",
        "categorical",
        "free_text",
        "geographic",
        "date",
        "url",
        "boolean",
        "numeric_continuous",
    }
)


@dataclass
class FieldSpec:
    column: str
    predicted_type: str
    confidence: float
    python_type: str  # "int" | "str" | "float" | "bool" | "date"
    required: bool
    constraints: dict = field(default_factory=dict)  # e.g. {"pattern": ...}
    comments: list[str] = field(default_factory=list)  # human-readable, non-risk annotations
    risk_flags: list[str] = field(default_factory=list)  # things a human should double-check


def _non_null(raw_values: list[str | None]) -> list[str]:
    return [v.strip() for v in raw_values if v is not None and v.strip() != ""]


def _looks_all_int(values: list[str]) -> bool:
    return bool(values) and all(_INT_RE.fullmatch(v) for v in values)


def _looks_like_ordinal_candidate(values: list[str]) -> bool:
    """Advisory-only heuristic: small, dense, all-integer value set.

    Mirrors how company_size's ordinal treatment was actually discovered (a
    human confirming a proposed label) -- flags a candidate for review, never
    changes the generated type itself.
    """
    if not _looks_all_int(values):
        return False
    distinct = set(values)
    return 2 <= len(distinct) <= _MAX_ORDINAL_CANDIDATE_CODES


def _looks_like_boolean_text(values: list[str]) -> bool:
    if not values:
        return False
    distinct_lower = {v.lower() for v in set(values)}
    return any(distinct_lower <= pair for pair in _TRUE_FALSE_TEXT_SETS)


# Pydantic v2's exact string-to-bool coercion set (verified empirically, not from docs --
# see the Task 2 companies.csv report for how this was found to matter in practice).
_PYDANTIC_BOOL_STRINGS = {"0", "1", "true", "false", "yes", "no", "y", "n", "t", "f", "on", "off"}


def _all_pydantic_bool_parseable(values: list[str]) -> bool:
    return all(v.lower() in _PYDANTIC_BOOL_STRINGS for v in values)


def _parses_as_iso_date(v: str) -> bool:
    try:
        _date.fromisoformat(v[:10])
        return True
    except ValueError:
        return False


_US_DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")


def _safe_comment(text: str) -> str:
    """Collapses whitespace/newlines so raw data can never break out of a
    generated `# comment` line into executable source -- e.g. a CSV cell
    containing an embedded newline followed by real Python would otherwise
    become a live statement in the generated file once written to disk.
    """
    collapsed = " ".join(text.split())
    max_len = 160
    return collapsed if len(collapsed) <= max_len else collapsed[: max_len - 1] + "…"


def _low_confidence_flag(confidence: float) -> list[str]:
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return [f"low-confidence prediction ({confidence:.2f}) -- runner-up type was close, review manually"]
    return []


def build_field_spec(
    column: str, predicted_type: str, confidence: float, raw_values: list[str | None]
) -> FieldSpec:
    if predicted_type not in KNOWN_TYPES:
        raise ValueError(
            f"Unknown predicted type {predicted_type!r} for column {column!r} -- "
            f"expected one of {sorted(KNOWN_TYPES)}. Model 1's label set may have changed."
        )

    non_null = _non_null(raw_values)
    null_count = sum(1 for v in raw_values if v is None or (isinstance(v, str) and v.strip() == ""))
    risk_flags = _low_confidence_flag(confidence)
    comments: list[str] = []

    # Type-consistency check, run before the ordinal/boolean-text advisories: a `boolean`
    # column can have at most 2 distinct non-null values by definition, so more than that is
    # a direct contradiction of the prediction itself, not just a low-confidence guess. Found
    # via company_size (predicted boolean, 7 distinct values 1-7) -- auto-downgrading to
    # categorical here, generally, means the small-dense-integer ordinal advisory below gets a
    # chance to fire on cases like it instead of only ever running when Model 1 says
    # "categorical" directly. `predicted_type` (Model 1's actual, raw output) is preserved on
    # the FieldSpec either way -- this only changes which field-shape branch runs.
    effective_type = predicted_type
    if predicted_type == "boolean":
        distinct_non_null = set(non_null)
        if len(distinct_non_null) > 2:
            risk_flags.append(
                f"Model 1 predicted boolean, but {len(distinct_non_null)} distinct non-null values "
                "is a definitional contradiction (boolean allows at most 2) -- auto-downgraded to "
                "categorical treatment below"
            )
            effective_type = "categorical"

    if effective_type == "identifier":
        is_int = _looks_all_int(non_null)
        spec = FieldSpec(
            column=column,
            predicted_type=predicted_type,
            confidence=confidence,
            python_type="int" if is_int else "str",
            required=True,
        )
        if null_count:
            risk_flags.append(
                f"required field but {null_count} observed null(s) -- validation will reject those rows as-is"
            )

    elif effective_type == "categorical":
        distinct = sorted(set(non_null))
        if distinct:
            if len(distinct) <= _MAX_LISTED_CATEGORIES:
                comments.append(_safe_comment(f"Observed values ({len(distinct)}): {distinct}"))
            else:
                comments.append(_safe_comment(f"{len(distinct)} distinct observed values -- too many to list"))
        if _looks_like_ordinal_candidate(non_null):
            risk_flags.append(
                "possible ordinal-coded numeric (small, dense integer codes, like company_size) -- "
                "kept as Optional[str] here; consider a hard-coded Optional[int] with a range if a "
                "human confirms an order, same as company_size"
            )
        if _looks_like_boolean_text(non_null):
            risk_flags.append(
                "possible boolean mislabeled as categorical -- Model 1's only boolean training "
                "examples are 0/1-coded, so it has never learned text true/false (documented "
                "blind spot, see tests/test_column_type_classifier_model.py)"
            )
        spec = FieldSpec(
            column=column,
            predicted_type=predicted_type,
            confidence=confidence,
            python_type="str",
            required=False,
            comments=comments,
        )

    elif effective_type == "free_text":
        spec = FieldSpec(
            column=column, predicted_type=predicted_type, confidence=confidence, python_type="str", required=False
        )

    elif effective_type == "geographic":
        spec = FieldSpec(
            column=column,
            predicted_type=predicted_type,
            confidence=confidence,
            python_type="str",
            required=False,
            comments=[
                (
                    "No format enforced -- real geographic data is genuinely mixed-format "
                    "(see the state column's handling in the imputation selector)."
                ),
            ],
        )

    elif effective_type == "date":
        if non_null:
            iso_rate = sum(_parses_as_iso_date(v) for v in non_null) / len(non_null)
            us_rate = sum(bool(_US_DATE_RE.match(v)) for v in non_null) / len(non_null)
        else:
            iso_rate = us_rate = 0.0
        if iso_rate >= 0.95:
            spec = FieldSpec(
                column=column,
                predicted_type=predicted_type,
                confidence=confidence,
                python_type="date",
                required=False,
                comments=[f"{iso_rate:.0%} of observed values parse as ISO 8601 dates."],
            )
        else:
            spec = FieldSpec(
                column=column,
                predicted_type=predicted_type,
                confidence=confidence,
                python_type="str",
                required=False,
            )
            if us_rate >= 0.95:
                risk_flags.append(
                    f"date-shaped values present ({us_rate:.0%} match MM/DD/YYYY) but not ISO 8601 -- "
                    "kept as Optional[str]; add a custom parser if you need a real `date` type"
                )
            else:
                risk_flags.append(
                    f"Model 1 predicted 'date' but only {iso_rate:.0%} of observed values parse as ISO "
                    "8601 dates -- kept as Optional[str], treat this prediction with suspicion"
                )

    elif effective_type == "url":
        spec = FieldSpec(
            column=column,
            predicted_type=predicted_type,
            confidence=confidence,
            python_type="str",
            required=False,
            constraints={"pattern": URL_PATTERN},
            comments=["Pattern reused verbatim from features.URL_PATTERN (Phase 2's bare-domain fix)."],
        )

    elif effective_type == "boolean":
        risk_flags.append(
            "Model 1's only boolean training examples are 0/1-coded -- it has never learned text "
            "true/false, so treat this prediction with more suspicion than the other types "
            "(and double-check nearby 'categorical' columns for a possible miss, see the "
            "categorical branch above)"
        )
        # Unlike the other soft-fallback types, `bool` is not actually soft: Pydantic only
        # coerces a fixed string set (see _PYDANTIC_BOOL_STRINGS), so a wrong "boolean"
        # prediction doesn't just look questionable -- it hard-fails validation on every row
        # whose real value isn't in that set (discovered exactly this way, scoring the
        # generated companies.csv contract against real rows: company_size predicted
        # boolean at 0.48 confidence, rejecting ~71% of rows outright). Same fallback
        # discipline as `date` above: verify before committing to the strict type.
        if non_null and not _all_pydantic_bool_parseable(non_null):
            risk_flags.append(
                "observed values are NOT all Pydantic-bool-parseable (only "
                f"{sorted(_PYDANTIC_BOOL_STRINGS)} coerce) -- kept as Optional[str] instead of "
                "Optional[bool] so this field doesn't hard-reject real rows"
            )
            spec = FieldSpec(
                column=column, predicted_type=predicted_type, confidence=confidence, python_type="str", required=False
            )
        else:
            spec = FieldSpec(
                column=column, predicted_type=predicted_type, confidence=confidence, python_type="bool", required=False
            )

    else:  # numeric_continuous
        floats = []
        for v in non_null:
            try:
                floats.append(float(v))
            except ValueError:
                pass
        comments = [f"Observed range: [{min(floats):g}, {max(floats):g}] (soft reference, not enforced)"] if floats else []
        spec = FieldSpec(
            column=column,
            predicted_type=predicted_type,
            confidence=confidence,
            python_type="float",
            required=False,
            comments=comments,
        )

    spec.risk_flags = risk_flags
    return spec
