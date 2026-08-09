# drift/

Phase 4, Task 2 done: PSI (categorical) and KS-test (numeric) drift monitoring, built from
scratch (no external drift-monitoring library) against a reference profile persisted from a
baseline sample of `companies.csv`.

`companies.csv` is one static snapshot with no timestamp column, so there's no real longitudinal
drift to detect here. `build_batches.py` demonstrates the detector against three batches built
from the same underlying sample, with a documented *synthetic* shift injected at two magnitudes:

```bash
python -m lucidflow.drift.build_batches
```

| batch | shift magnitude | company_size (PSI) | state null rate (PSI) | description length (KS) |
|---|---|---|---|---|
| A | none (negative control) | none | none | none |
| B | full | significant | significant | significant |
| C | half | moderate | moderate | moderate |

This proves the PSI/KS wiring is correct and sensitive to the kinds of shift it's designed to
catch — it does **not** validate real-world drift-detection performance or a "correct" threshold
for this pipeline in production, since there's no real drift in this dataset to calibrate either
against. See `monitor.py` and `synthetic_shift.py` docstrings for the full reasoning.

Wired into the Prefect flow (Task 3, done) — see `src/lucidflow/flows/README.md`: the pipeline's
optional `--check-drift` flag runs this exact check against a full-magnitude synthetic shift of
the just-ingested batch, and triggers a retraining flow when it flags.
