# contract_generation/

Closes a real gap left open since Phase 2: Model 1 (the column-type classifier) was
trained and evaluated, but nothing consumed its output — the schema that actually runs
against `companies.csv` is still `validation/pydantic_models.py`'s hand-written `Company`
contract. This package adds the missing inference-to-schema path: given a new CSV with
unknown columns, run Model 1 on each column's statistical fingerprint and generate a
working Pydantic contract from the predictions.

## Scope — read this before trusting a generated contract

**This makes schema *inference* generalize to new tabular data. It does not make
the trained models generalize.**

- `predict.py` runs the existing, unmodified `column_type_classifier.joblib` — no
  retraining happened for this package, and none of Model 1's real limitations
  (documented in `models/column_type_classifier/README.md` and the tests below) were
  patched around. They're surfaced as risk flags on the generated fields instead.
- The **imputation selector** and **quarantine classifier** are not touched by this
  package at all, and nothing here changes their behavior. Both remain tuned
  specifically to `companies.csv`'s actual columns (e.g. the imputation selector's
  per-column strategy table, the quarantine classifier's corruption-type features) —
  running them against a structurally different CSV is out of scope and untested by
  this package. A generated contract only tells you what *shape* a column looks like
  well enough to validate against; it says nothing about whether those two models would
  produce anything meaningful on that column's actual values.
- Point this at a CSV that looks nothing like `companies.csv` and you'll get a
  syntactically valid Pydantic model every time — that's the mechanism working, not a
  claim the *predictions* are all correct. See "Task 2" below for what confident-vs-wrong
  actually looked like on real data.

## Type-to-field mapping (Task 1)

Model 1's real 8-class label set: `identifier`, `categorical`, `free_text`, `geographic`,
`date`, `url`, `boolean`, `numeric_continuous`. Full mapping table and reasoning is in
`type_mapping.py`'s module docstring — short version:

| Model 1 type | Generated field |
|---|---|
| `identifier` | required; `int` or `str` depending on whether observed values parse as integers |
| `categorical` | `Optional[str]`, observed values listed as a comment (not a hard `Enum`) |
| `free_text` | `Optional[str]`, unconstrained |
| `geographic` | `Optional[str]`, unconstrained (mixed-format data, same reasoning as `state`) |
| `date` | `date` if observed values parse as ISO 8601, else `Optional[str]` |
| `url` | `Optional[str]`, permissive pattern reused verbatim from `features.URL_PATTERN` |
| `boolean` | `Optional[bool]` if every observed value is actually Pydantic-bool-parseable, else `Optional[str]` — always flagged either way: Model 1's boolean examples are all 0/1-coded, never text |
| `numeric_continuous` | `Optional[float]`, observed min/max noted as a soft comment, not enforced |

The `boolean` fallback above was added *after* Task 2 testing against real
`companies.csv` data: `company_size` predicted `boolean` at low confidence, and
`Optional[bool]` hard-rejected 71% of real rows (Pydantic only coerces a fixed string
set to bool). Unlike the other types, a wrong `boolean` prediction isn't just
"questionable," it actively breaks validation — see
`reports/companies_csv_comparison.md` for the full story, including a known remaining
gap (the ordinal-candidate advisory only checks `categorical` predictions, not
`boolean` ones, so `company_size` doesn't get flagged as a possible ordinal code in
this run).

**No `numeric_ordinal_code` type.** `company_size` was originally proposed with that
label during Phase 2's labeling step, but the label a human actually confirmed —
the one the model trained on — is `categorical` (its ordinal-ness is hard-coded
separately in the imputation selector). So this mapping only covers Model 1's real
output classes. What it adds instead: a `categorical` column whose observed values are
all small, dense integers gets an advisory "possible ordinal code — human review
suggested" comment, without changing the generated type. Same pattern for the boolean
blind spot — a `categorical` column that looks like true/false text gets flagged as a
possible miss. Both are report-only signals, mirroring how `company_size`'s ordinal
treatment was actually discovered (a human confirming a proposed label), not the
generator silently asserting something the data can't prove.

Every field also carries a `# Model 1 prediction: <type> (confidence 0.XX)` comment, and
a low-confidence flag (`< 0.6`) when the winning label barely beat the runner-up.

## Usage

```bash
python -m lucidflow.contract_generation.generate_contract data/intake/companies.csv Company
# writes src/lucidflow/contract_generation/generated/Company.py (gitignored)
```

Programmatically:

```python
from lucidflow.contract_generation.generate_contract import generate_from_csv

result = generate_from_csv("data/intake/companies.csv", class_name="Company")
result.source        # human-readable .py text
result.model          # a real, usable pydantic.BaseModel subclass (built via create_model,
                       # not by exec()-ing the generated text — see generate_contract.py)
result.field_specs    # per-column FieldSpec: predicted_type, confidence, risk_flags, ...
```

Column headers that aren't valid Python identifiers (spaces, punctuation, leading
digits, Python keywords) are sanitized and mapped back via a Pydantic `alias`, so the
generated model still validates rows keyed by the original CSV header.

## Task 2 reports

- `reports/companies_csv_comparison.md` — auto-generated schema vs. the hand-written
  Phase 1 `Company` contract, column by column.
- `reports/second_dataset_report.md` — run against a structurally different real CSV:
  how many columns came back confidently typed vs. uncertain, and whether the result is
  actually usable for validation.
