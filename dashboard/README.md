# dashboard/

Phase 5, Tasks 2 and 3 done: read-only observability, plus human-in-the-loop quarantine review
and a real retraining hook.

```bash
set -a; source .env; set +a
streamlit run dashboard/app.py
```

- `app.py` — Pipeline Summary (landing page): most recent run's clean/quarantine counts and
  quarantine-reason breakdown, read from Postgres.
- `pages/1_Model_Results.py` — latest MLflow run per model: column-type classifier confusion
  matrix, imputation benchmark table (with class-coverage caveats), quarantine classifier
  per-corruption-type precision/recall.
- `pages/2_Drift_Status.py` — the three-batch (none/moderate/significant) PSI/KS
  characterization from Phase 4, Task 2, read from `src/lucidflow/drift/last_check_results.json`.
- `pages/3_Entity_Resolution.py` — renders `docs/entity_resolution_investigation.md` directly,
  so the dashboard can't drift out of sync with the actual investigation writeup.
- `pages/4_Quarantine_Review.py` — the human-in-the-loop review queue, one real quarantined
  record at a time (ML-classifier flags only, not deterministic contract failures), plus a
  gated "Retrain Model 3" action once enough decisions exist. See below.
- `data_access.py` — all data access (Postgres, MLflow, drift results, the doc), plus Task 3's
  review-queue read/write and the retrain trigger. "Most recent pipeline run" is derived from
  Postgres write timestamps without any schema change or run_id column -- see the module
  docstring for why that works.

For Task 2's read-only pages, no new backend logic, with one small exception:
`last_check_results.json` didn't previously exist (Phase 4's drift check only printed results)
-- `build_batches.py` now also persists what it already computes, same "save the aggregate, not
raw text" pattern as `reference_profile.json`.

## Task 3 — human-in-the-loop quarantine review

The quarantine classifier was trained entirely on synthetic corruption (Task 0: real
`companies.csv` has zero organic contract failures). This page is what gives it real-world
labels: reviewers see one real quarantined record at a time -- raw fields, the flagged reason,
anomaly score, and the exact feature vector the classifier scored (persisted at classification
time in `pipeline_flow.py`'s `quarantine_classify_task`, never recomputed) -- and mark it
`Confirmed bad` or `False positive`. Decisions are written to
`quarantine.quarantine_reviews` (`record_id UNIQUE`, so a record is reviewed at most once and
then drops out of the queue).

**Honesty check, run for real**: reviewed the first 20 real ML-flagged records from an actual
pipeline run against `companies.csv`. 19 were legitimate, complete, well-formed company records
the classifier over-flagged (false positives) -- consistent with Task 0's grounding that real
positives are genuinely thin. 1 was a real, verified defect: a company description containing
`â€"`, the exact byte signature of an em-dash misread through cp1252 (confirmed via codepoint
inspection, not assumed from how it rendered) -- a genuine example of the "encoding" corruption
type the classifier was built to catch, occurring organically in the wild.

Once 20 reviews exist (`MIN_REVIEWS_FOR_RETRAIN` in `data_access.py` -- picked as more than a
token handful while realistically reachable against current real quarantine volume; deliberately
a total-count threshold with no per-class minimum, since requiring N confirmed-bad specifically
could make the gate permanently unreachable for a legitimately correct reason), the page's
"Retrain now" button unlocks. It runs
`models/quarantine_classifier/retrain_with_reviews.py`, which combines the synthetic-corruption
dataset with the reviewed rows, evaluates on the *same fixed synthetic held-out split*
`train.py` itself uses (so the resulting PR-AUC is directly comparable to whatever's currently
registered), and only registers a new version if that PR-AUC doesn't regress by more than 0.01 --
requiring strict improvement would be measuring noise with this few human-reviewed examples, not
signal. The human-reviewed rows also get a small separate held-out eval, reported but never used
for the registration decision.

**Verified end-to-end for real**, not simulated: reviewed 20 real records (1 confirmed_bad, 19
false_positive), clicked retrain, and it worked exactly as designed --

```
Synthetic-test PR-AUC (candidate):            0.9871
Synthetic-test PR-AUC (previously registered): 0.9864
Human-reviewed rows used: 20 (1 confirmed_bad, 19 false_positive)
Small supplementary human-reviewed eval (n=4, NOT used for the registration decision):
  precision=0.0000, recall=0.0000
Registered as new version 3.
```

Independently confirmed after the fact (not just trusting the UI): the MLflow run
(`e29c88c50f99425c89c3465272fa6e15`, tagged `retrain_trigger=human_review`) and the
`quarantine.quarantine_reviews` table both match exactly.

**What this does and doesn't prove**: this demonstrates the full loop works -- review, persist,
retrain, evaluate, conditionally register -- with a real (if modest) PR-AUC improvement on this
one run. It does **not** prove the model has learned anything robust from human labels yet: with
only 1 confirmed-bad example, the human-reviewed eval's 0.0000/0.0000 reflects an all-negative
4-row test slice (no positive class to recall), not a meaningful precision/recall estimate --
that number should be read as "not enough data yet," not "the model is bad at this." Real signal
here requires reviewing more real quarantine volume over time, ideally across multiple pipeline
runs with genuinely different flagged rows.
