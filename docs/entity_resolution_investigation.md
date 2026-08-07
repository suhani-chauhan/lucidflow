# Entity resolution: investigated, not built

A fourth trained model was scoped for Phase 3 — an LLM-distilled duplicate-pair classifier
(local embeddings for candidate generation, Ollama for training-label adjudication, LightGBM for
the final classifier) to predict whether two company records refer to the same real-world entity.
Before writing any embedding/blocking/classifier code, the premise was checked against the actual
dataset. It didn't hold up across two separate passes, so the model was dropped rather than built
on a weak signal.

## Pass 1 — `companies.csv`, normalized company names

**Method**: lowercase, strip punctuation, drop trailing legal-form tokens (`Inc`, `LLC`, `Ltd`,
`Corp`, `Group`, `Holdings`, `International`, etc.), group by the normalized name, and look for
groups where more than one distinct `company_id` shares the normalized form.

**Result**: 139 groups out of 24,473 rows (288 rows, ~1.18%) — before excluding false positives.

Manual inspection of a sample split the 139 groups into three categories:

**Genuine likely-duplicates** (same real company, two LinkedIn pages, formatting/legal-suffix
variants) — the actual signal, and it's small (~10-15 pairs total):

| example | ids |
|---|---|
| `Johnson & Johnson` / `Johnson & Johnson, Inc.` | 1207 / 896415 |
| `Merck` / `Merck & Co., Inc.` | 1486 / 90569489 |
| `Moody's Corporation` / `Moody's` | 165033 / 306019 |
| `Centene Corporation` / `Centene` | 9703 / 10448504 |
| `Hyatt Hotels Corporation` / `HYATT Hotels` | 220336 / 3663581 |
| `Georgia-Pacific LLC` / `Georgia Pacific` | 3411 / 85677 |
| `TEKsystems` / `Teksystems LLC` | 2152 / 34072437 |
| `Atlas Copco` / `Atlas Copco Group` | 4804 / 93389372 |
| `Persistent Systems` / `Persistent Systems, LLC` | 5034 / 582324 |
| `The North Face` / `The North Face` (identical string, two IDs) | 4667 / 89178877 |

**False positives** (generic/placeholder names shared by genuinely unrelated companies) — this is
most of the 139 groups:

- `Confidential` — **7 distinct `company_id`s**, unrelated employers hiding their identity behind
  a common placeholder
- `Independent Consultant`, `Law Firm`, `Stealth`, `Anonymous` — literal placeholders, not company
  names
- `Atlas`, `Paradigm`, `Pandora`, `Archer`, `Match` — generic dictionary words, unrelated firms
- `Mercer` (2423) / `Mercer International Inc.` (118708) — a genuine trap: HR-consulting Mercer
  (part of Marsh McLennan) and pulp/paper manufacturer Mercer International are different
  companies that happen to collide after suffix-stripping

**Genuinely ambiguous** (would need "unsure" / human-review handling, not a forced label):

- `Baptist Health` (11809 / 14991) — several unrelated regional US hospital systems share this
  exact name
- `Colorado State University` (163149) / `Colorado State University Global` (497729) — related
  but legally distinct institution
- `JBS International, Inc.` (477190) / `JBS USA` (25040991) — a research consultancy vs. the
  meat-processing subsidiary of Brazil's JBS S.A.

**Fallback hypothesis checked and ruled out**: the plan was to fall back to `postings.csv`'s
`company_name` field on the theory that it's free-text and less curated than the linked
`companies.csv` table. Checked directly: within `postings.csv`, `company_name` has **zero
spelling variance per `company_id`** (verified across all 24,472 non-null pairs), and the
`company_id` universe between `postings.csv` and `companies.csv` is essentially identical (off by
exactly 1 row, full overlap otherwise). It's a direct denormalized join from the same canonical
`company_id -> name` mapping, not independently-entered free text — no additional signal there.

## Pass 2 — `postings.csv`, near-duplicate job postings

Before concluding, a second candidate signal was checked: near-duplicate *postings* (not company
records) within the same `company_id` — title/description similarity, cheap exact/near-exact
match first, before reaching for embeddings.

**Method**: group postings by `(company_id, description)` for exact matches (122,126 rows with
non-null `company_id` and `description`).

**Result**: 5,957 groups, 21,574 rows (17.67%) — a much bigger number on paper. Splitting by
`location` changes the picture:

| bucket | groups | rows | share |
|---|---|---|---|
| same description, different locations | 3,448 | 15,011 | 69.6% |
| same description, same location | 2,509 | 6,563 | 30.4% |

**Different-location bucket (69.6% of the duplicate-row mass) is a false-positive pattern**, the
same shape as `companies.csv`'s generic-name problem — a company posting the same boilerplate
description for genuinely different openings across many stores:

- `company_id=73013724`: "Sales Manager" — 474 postings, different stores
- `company_id=163761`: "ASSISTANT STORE MANAGER" — 167 postings, different locations
- `company_id=4128`: "Sales Associate — Building Materials / Flooring / Flexible" — 68 postings,
  different departments/locations

**Same-location bucket is the closer candidate, but still mostly explains away**:

- `company_id=54814820` (a New Orleans restaurant): Line Cook x30, Server x19, Dishwasher x9+x9 —
  legitimate recurring hiring for high-turnover service roles, not accidental duplication
- `company_id=163139`: "Data Center Operations Manager-Houston Texas" x9, identical
  title+description+location — the strongest candidate for genuine duplication found

To go further, `listed_time` was pulled in: **the entire dataset spans only 26 days**
(2024-03-24 to 2024-04-20 — a single scrape snapshot), and 54.5% of same-location groups cluster
within a single day. That's ambiguous either way — consistent with genuine accidental
duplication, but equally consistent with normal job-board relisting behavior (a live posting
being refreshed to stay visible in search), which isn't a "same real thing labeled as different"
problem at all. With only 26 days of data and no ground truth on LinkedIn's relisting mechanics,
these can't be told apart with any confidence.

## Conclusion

Both passes come back thin or unverifiable for the same underlying reason: the visible volume is
dominated by legitimate patterns (different locations = genuinely different real openings;
same-location repeats = legitimate recurring hiring or benign relisting), not a well-defined
duplicate-entity signal a classifier could learn and be evaluated against. Forcing an
embedding/blocking/LLM-labeling/LightGBM pipeline onto ~10-15 genuine company duplicates, or onto
a residual slice of postings that can't be confirmed as duplicates rather than relisting noise,
would produce a model that looks complete but isn't backed by a real, measurable problem in this
dataset.

Entity resolution is dropped from the plan on this basis. `src/lucidflow/resolution/` remains an
empty stub — see its `README.md`. Phase 3's trained model is the quarantine classifier instead
(see `src/lucidflow/models/quarantine_classifier/`).
