# Task 2: auto-generated schema vs. the hand-written Phase 1 contract

Ran `generate_from_csv("data/intake/companies.csv", "Company")` against the real, full
24,473-row file — the same file the hand-written `Company` contract
(`src/lucidflow/validation/pydantic_models.py`) targets, and the same file Model 1's
`companies/companies.csv` training fingerprints were built from. **This is not a
generalization test** — it's the intended first check from the task: does Model 1's
output, run through the Task 1 mapping, land close to what a human already wrote by
hand for this exact data? Task 2's genuinely-different-file test is the next report.

## Column-by-column

| Column | Model 1 prediction (confidence) | Generated field | Hand-written field | Agreement |
|---|---|---|---|---|
| `company_id` | identifier (0.89) | `int`, required | `int`, required | **Match** |
| `name` | free_text (0.94) | `str \| None` | `str \| None` | **Match** |
| `description` | free_text (0.89) | `str \| None` | `str \| None` | **Match** |
| `company_size` | boolean (0.48) → falls back to str (see below) | `str \| None` | `int \| None`, `ge=1, le=7` | **Differs** |
| `state` | geographic (0.74) | `str \| None`, no format | `str \| None` + `"0"`-sentinel validator | **Differs** (validator) |
| `country` | geographic (0.66) | `str \| None`, no format | `str \| None` + `"0"`-sentinel validator | **Differs** (validator) |
| `city` | geographic (0.78) | `str \| None`, no format | `str \| None` | **Match** |
| `zip_code` | geographic (0.75) | `str \| None`, no format | `str \| None` | **Match** |
| `address` | geographic (0.68) | `str \| None`, no format | `str \| None` | **Match** |
| `url` | url (0.96) | `str \| None`, permissive pattern | `HttpUrl`, **required** | **Differs** (optionality + type strictness) |

**Whole-file validation, generated vs. hand-written**, every one of the 24,473 real rows:

```
Hand-written Company:  24,473 / 24,473 pass (0 failures)
Generated Company:     24,473 / 24,473 pass (0 failures)   <- after the fix below
```

6 of 9 columns land on exactly the same field shape as the hand-written contract with no
intervention. The 3 that differ are informative, not just noise:

### `company_size` — the interesting one, and a real bug it surfaced

Model 1 predicts **`boolean` at 0.48 confidence** for `company_size`, not the
`categorical` label it was actually trained on for this exact column. Confirmed this
isn't a pipeline bug: the freshly-extracted feature vector is byte-for-byte identical to
the one stored in `column_fingerprints.json`, and re-scoring that exact stored vector
gives `{"boolean": 0.477, "categorical": 0.430, ...}` — a near-tie the trained forest
happens to resolve the "wrong" way. `company_size`'s fingerprint (7 distinct all-integer
codes, very low cardinality) sits close to the model's other low-cardinality all-integer
examples, which happen to be its 0/1-coded boolean training columns — a real, structural
ambiguity in the feature space, not a fluke of this run.

This surfaced a genuine gap in Task 1's own field-mapping code, not Model 1: the
`boolean` type was mapped straight to `Optional[bool]` unconditionally, but Pydantic
only coerces a fixed string set to bool (`0/1/true/false/yes/no/y/n/t/f/on/off` —
verified empirically). Validating the *first* generated `Company` contract against all
24,473 real rows failed 17,351 of them (71%) — every row whose `company_size` was
`"2"`–`"7"` — because those strings aren't in that set. That's exactly the "usable vs.
too wrong to be useful" failure mode Task 2 asks to watch for, and it would have been
easy to miss without actually validating rows instead of just eyeballing the generated
source.

Fixed in `type_mapping.py`: a `boolean` prediction now checks whether *all* observed
values are actually Pydantic-bool-parseable before committing to `Optional[bool]`;
otherwise it falls back to `Optional[str]` (same fallback discipline `date` already
had), and keeps every risk-flag comment regardless. After the fix, `company_size`
generates as `str | None` with three stacked RISK comments (low confidence, the
boolean blind spot, and the failed-parseability check) and the file validates clean.

It still doesn't recover `ge=1, le=7` — Model 1 has no ordinal concept, and the
`_looks_like_ordinal_candidate` advisory heuristic only fires on `categorical`
predictions, not `boolean` ones, so it never got a chance to flag "possible ordinal
code" here either. **Known gap, not fixed**: the ordinal-candidate check should
probably also run in the boolean branch, since this run shows the same small-dense-
integer shape can land as either type. Flagging this rather than quietly patching it
in — happy to extend it if wanted.

### `state` / `country` — the sentinel validator gap

The hand-written contract has a `@field_validator` that treats the literal string
`"0"` as null for `state`/`country` (a real quirk of this source data — LinkedIn uses
`"0"` as an "unknown" sentinel instead of an empty cell). Nothing in a column's
statistical fingerprint reveals *why* a value means "unknown" instead of being a real
category — that's domain knowledge from actually looking at the data, not something
inferable from shape alone. The generated contract has no equivalent, and isn't
expected to: this is exactly the kind of judgment call Task 1 said stays human, not
generated.

### `url` — required vs. optional, `HttpUrl` vs. pattern-matched `str`

Two independent differences here, both by design, not bugs:

- **Optionality.** The mapping table makes `identifier` the only required type;
  `url` is `Optional[str]` even though this column has zero observed nulls. The
  hand-written contract's `url: HttpUrl` (required, no default) reflects a human
  judgment — "every real company record has a URL" — that isn't visible in the
  fingerprint either; a future dataset's `url` column could easily have nulls, and
  the generator has no way to know this one, empirically, doesn't.
- **Strictness.** The hand-written contract uses Pydantic's `HttpUrl` type (real
  URL parsing). The generated contract deliberately uses a permissive regex pattern
  instead — reused verbatim from `features.URL_PATTERN`, the same bare-domain-inclusive
  fix from Phase 2. `HttpUrl` would reject the bare-domain values (`example.com`, no
  scheme) that Phase 2 found are common in this real data; matching Model 1's own
  permissive definition of "looks like a URL" is more honest than generating something
  stricter than what the model itself was scored against.

## Bottom line

For the file Model 1 was literally trained on, 6/9 fields regenerate identically to the
hand-written contract with zero intervention, and after fixing the boolean-fallback gap
the generated schema validates all 24,473 real rows cleanly — same as the hand-written
one. The 3 differences are exactly the categories you'd expect the generator to get
"different, not wrong" on: one real ambiguous prediction (company_size, now safely
degraded instead of silently broken), one piece of dataset-specific domain knowledge no
statistical fingerprint could reveal (the `"0"` sentinel), and one deliberate
strictness trade-off stated in the mapping table up front (`url`'s permissive pattern).
None of this required touching Model 1 itself.
