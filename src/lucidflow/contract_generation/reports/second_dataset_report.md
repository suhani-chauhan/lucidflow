# Task 2: generation against a genuinely different file layout

## What this test is, and isn't

Per explicit instruction, this uses one of the sibling files already downloaded with
the Kaggle dataset rather than sourcing new data, given time constraints:
`data/linkedin-job-postings/jobs/salaries.csv` (job compensation data — a different
domain, a different file, a different row shape than `companies.csv`: 8 columns, 40,785
rows, no free text or geography at all).

**Read this precisely**: every one of `salaries.csv`'s 8 columns —
`salary_id, job_id, max_salary, med_salary, min_salary, pay_period, currency,
compensation_type` — already exists as a `(source_file, column)` key in
`confirmed_labels.csv`/`column_fingerprints.json`. That means Model 1 was trained on
human-confirmed labels for these exact columns already. This run validates two real
things:

1. **The end-to-end generation pipeline works on a new file it's never been pointed at
   before** — a different CSV, different row content, a domain (pay data) structurally
   unlike `companies.csv` (no free text, no geography, higher identifier-to-field
   ratio). That's a genuine, useful check the mechanism isn't hard-coded to one file.
2. **It is not a test of blind generalization to unseen column *types*.** Model 1 has
   already seen every column name and semantic type in this file during training.
   This report cannot and does not claim the classifier would perform this well on a
   column shape it has genuinely never encountered (e.g. a phone number, an email-only
   column, a multi-value tag list) — that would require a CSV with real structural
   novelty, which time didn't allow for this pass. Both facts are worth knowing; treat
   this as "the pipeline works end-to-end on a new file," not "the classifier
   generalizes to new kinds of data."

`jobs/benefits.csv` was checked too as a smaller sanity check (3 columns: `job_id`,
`inferred`, `type` — also all in the training set). Same caveat applies; not written up
separately since it adds no new information beyond confirming the same point on a
tinier file (all 3 columns confidently and correctly typed, including `inferred` —
literally one of Model 1's three known boolean training examples, at confidence 0.97).

## Results on `salaries.csv`

| Column | Model 1 prediction (confidence) | Generated field | Confident? |
|---|---|---|---|
| `salary_id` | identifier (0.74) | `int`, required | yes |
| `job_id` | identifier (0.94) | `int`, required | yes |
| `max_salary` | numeric_continuous (0.92) | `float \| None`, range [1, 1.2e+08] | yes |
| `med_salary` | numeric_continuous (0.92) | `float \| None`, range [0, 750000] | yes |
| `min_salary` | numeric_continuous (0.95) | `float \| None`, range [1, 8.5e+07] | yes |
| `pay_period` | categorical (0.79) | `str \| None`, 5 observed values | yes |
| `currency` | categorical (0.56) | `str \| None`, 6 observed values | **no** — below the 0.6 threshold |
| `compensation_type` | categorical (0.82) | `str \| None`, 1 observed value | yes |

**8/8 columns confidently typed at the field level** (7 above the 0.6 confidence
threshold, 1 flagged as low-confidence). All 8 predictions match the human-confirmed
label in `confirmed_labels.csv` exactly — expected, given point 1 above; this isn't
independent evidence the classifier is broadly accurate, just that it reproduces its
own training labels on fresh extraction from the same source file.

**Whole-file validation**: all 40,785 real rows validate successfully against the
generated `Salary` model (0 failures) — no boolean-fallback issue here (this file has
no boolean-predicted columns), no sentinel-value surprises. The generated schema is
directly usable as-is for this file, with the caveat that `max_salary`'s observed range
tops out at $120M and `min_salary` at $85M — real values in this dataset, not generator
artifacts, but worth a human glance before treating the "soft range" comment as a
sanity bound (a $120M single salary figure is very plausibly a data-entry error in the
source, not a schema problem — out of scope for this task to chase down, but worth
flagging exactly the way this project flags every other "found something odd in real
data, didn't silently fix it" moment).

## Honest failure-mode assessment

Nothing here failed. That's a real result, but a limited one given point 2 above: this
run demonstrates the generation *mechanism* — CSV in, correctly-shaped Pydantic model
out, validating real rows — holds up on a file with a genuinely different column count,
domain, and row shape. It does not demonstrate the *classifier* generalizing beyond its
8 trained semantic types or beyond columns resembling ones it has seen, because nothing
in this file tested that. A meaningful stress test of Model 1's actual generalization
would need a CSV with a column type genuinely absent from its training distribution
(the report above already found one real failure mode on `companies.csv` itself —
`company_size`'s boolean/categorical ambiguity — without needing an unseen file to find
it).
