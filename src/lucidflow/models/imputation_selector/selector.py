"""Missingness engine: diagnoses, imputes, and reports on every column with
nulls in companies.csv. Column-by-column treatment (decided with the user,
see PR discussion / project notes rather than guessed):

  company_size  -> full benchmark (median/KNN/MICE/LightGBM), hard-coded as
                   an ordinal column regardless of what the column-type
                   classifier (Model 1) labels it as, since ordinal-ness is
                   structural knowledge the classifier can't infer from
                   stats alone.
  state         -> mode-within-country fallback. Originally slated for the
                   full benchmark suite on the assumption of a clean
                   ~31-rows/class distribution; the real data has hundreds
                   of singleton/dirty free-text values (non-ASCII province
                   names, malformed entries), so stratified benchmarking
                   couldn't run meaningfully. Downgraded after auditing the
                   actual distribution, not before.
  city          -> mode-within-(state, country) fallback. Near-identifier
                   cardinality-wise, but genuinely recoverable via
                   geographic correlation, so a group-mode fallback beats
                   leaving it null.
  zip_code      -> mode-within-(state, country) fallback, same reasoning.
  address       -> left null, no fallback attempted. ~19,478/24,454 rows
                   are distinct addresses (essentially unique per row); a
                   group-mode fallback there would be statistically
                   meaningless, not just noisy.
  description   -> left null, no fallback attempted. Free text, ~99%
                   unique values; none of the benchmarked strategies
                   predict open-ended text.

city and zip_code are imputed *after* state, so their group-mode lookups
can use a state column that's already been filled in.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
import polars as pl

from lucidflow.models.imputation_selector.benchmark import benchmark_column
from lucidflow.models.imputation_selector.diagnostics import diagnose_missingness
from lucidflow.models.imputation_selector.group_mode import apply_group_mode, fit_group_mode
from lucidflow.models.imputation_selector.strategies import build_strategies

logger = logging.getLogger("lucidflow.models.imputation_selector")

ARTIFACT_DIR = Path(__file__).parent

BENCHMARKED_COLUMNS = {
    "company_size": {"predictors": ["state", "country", "city"], "kind": "ordinal"},
    # 81 clean ISO-2 codes (verified no near-duplicate encodings like state had),
    # ~302 rows/class on average -> a real stratified-split target, unlike state.
    "country": {"predictors": ["state", "city", "zip_code", "company_size"], "kind": "categorical"},
}
GROUP_MODE_COLUMNS = {
    "state": ["country"],
    "city": ["state", "country"],
    "zip_code": ["state", "country"],
}
SKIPPED_COLUMNS = ["address", "description"]

# country must resolve before state/city/zip_code, which group-mode fallback against it.
# company_size has no dependents so it can go first or anywhere before state/city/zip_code.
PROCESSING_ORDER = ["company_size", "country", "state", "city", "zip_code", "description", "address"]


def run_missingness_engine(df: pl.DataFrame, artifact_dir: Path = ARTIFACT_DIR) -> tuple[pl.DataFrame, list[dict]]:
    """Diagnoses and imputes every column with nulls in `df`. Returns the
    imputed DataFrame and a report (one entry per column with any missing
    values) suitable for logging/printing.

    `artifact_dir` is where fitted imputers get persisted — override it (e.g.
    in tests) to avoid overwriting the real artifacts trained on the full
    dataset.
    """
    pdf = df.to_pandas()
    report = []

    columns_with_nulls = [c for c in df.columns if df[c].null_count() > 0]
    ordered = [c for c in PROCESSING_ORDER if c in columns_with_nulls]
    ordered += [c for c in columns_with_nulls if c not in PROCESSING_ORDER]

    for col in ordered:
        diagnosis = diagnose_missingness(df, col)

        if col in BENCHMARKED_COLUMNS:
            entry = _run_benchmarked_column(pdf, col, diagnosis, artifact_dir)
        elif col in GROUP_MODE_COLUMNS:
            entry = _run_group_mode_column(pdf, col, diagnosis, artifact_dir)
        elif col in SKIPPED_COLUMNS:
            entry = _run_skipped_column(diagnosis)
        else:
            entry = _run_unhandled_column(diagnosis)

        report.append(entry)

    return pl.from_pandas(pdf), report


def _run_benchmarked_column(pdf: pd.DataFrame, col: str, diagnosis: dict, artifact_dir: Path) -> dict:
    spec = BENCHMARKED_COLUMNS[col]
    known = pdf[pdf[col].notna()]

    result = benchmark_column(known, col, spec["predictors"], spec["kind"])
    winner_name = result["winner"]

    strategies = {s.name: s for s in build_strategies(spec["kind"])}
    winner = strategies[winner_name]
    winner.fit(known, col, spec["predictors"])

    missing_mask = pdf[col].isna()
    if missing_mask.any():
        predictions = winner.predict(pdf.loc[missing_mask], spec["predictors"])
        pdf.loc[missing_mask, col] = predictions

    artifact_path = artifact_dir / f"{col}_imputer.joblib"
    joblib.dump(winner, artifact_path)

    coverage = result["class_coverage"]
    entry = {
        "column": col,
        "decision": "benchmarked",
        "diagnosis": diagnosis["verdict"],
        "benchmark_scores": result["scores"],
        "winner": winner_name,
        "winner_score": result["scores"][winner_name],
        "class_coverage": coverage,
        "artifact": str(artifact_path),
    }
    if coverage["singleton_classes"] > 0:
        entry["known_limitation"] = (
            f"Structural limitation, not an implementation gap: {coverage['singleton_classes']}/"
            f"{coverage['total_classes']} classes ({100 * (1 - coverage['coverage_ratio']):.1f}%) had "
            "exactly 1 known example and went to training only -- a class with a single example can't "
            "also be held out to test against, so it's structurally impossible to verify recovery on it. "
            "The macro-F1 above reflects recovery only for the "
            f"{coverage['evaluated_classes']} classes with enough examples to test; performance on the "
            "untested singleton classes is unknown, not verified-good."
        )
    logger.info(
        "missingness_engine: %s -> benchmarked, winner=%s (macro-F1=%.4f), scores=%s, class_coverage=%s",
        col, winner_name, result["scores"][winner_name], result["scores"], coverage,
    )
    return entry


def _run_group_mode_column(pdf: pd.DataFrame, col: str, diagnosis: dict, artifact_dir: Path) -> dict:
    group_cols = GROUP_MODE_COLUMNS[col]
    lookup = fit_group_mode(pdf, col, group_cols)

    missing_mask = pdf[col].isna()
    if missing_mask.any():
        pdf.loc[missing_mask, col] = apply_group_mode(pdf.loc[missing_mask], col, lookup)

    artifact_path = artifact_dir / f"{col}_group_mode.joblib"
    joblib.dump(lookup, artifact_path)

    entry = {
        "column": col,
        "decision": f"group-mode fallback (grouped by {group_cols})",
        "diagnosis": diagnosis["verdict"],
        "artifact": str(artifact_path),
    }
    logger.info("missingness_engine: %s -> group-mode fallback grouped by %s", col, group_cols)
    return entry


def _run_skipped_column(diagnosis: dict) -> dict:
    entry = {
        "column": diagnosis["column"],
        "decision": "skipped (left null, no fallback attempted)",
        "diagnosis": diagnosis["verdict"],
    }
    logger.info("missingness_engine: %s -> skipped, left null", diagnosis["column"])
    return entry


def _run_unhandled_column(diagnosis: dict) -> dict:
    # A column has nulls but isn't in any of the three configured buckets above —
    # surfaced rather than silently imputed or silently ignored.
    entry = {
        "column": diagnosis["column"],
        "decision": "UNHANDLED — no strategy configured for this column",
        "diagnosis": diagnosis["verdict"],
    }
    logger.warning("missingness_engine: %s has nulls but no configured strategy", diagnosis["column"])
    return entry


def print_report(report: list[dict]) -> None:
    print("=== Missingness Engine Report ===")
    for entry in report:
        print(f"\ncolumn: {entry['column']}")
        print(f"  diagnosis: {entry['diagnosis']}")
        print(f"  decision:  {entry['decision']}")
        if "benchmark_scores" in entry:
            print("  benchmark scores (macro-F1):")
            for method, score in entry["benchmark_scores"].items():
                marker = " <- winner" if method == entry["winner"] else ""
                print(f"    {method:10s} {score:.4f}{marker}")
            coverage = entry["class_coverage"]
            print(
                f"  class coverage: {coverage['evaluated_classes']}/{coverage['total_classes']} classes "
                f"evaluated in test ({coverage['coverage_ratio']:.1%}), "
                f"{coverage['singleton_classes']} trained-only (singleton)"
            )
        if "known_limitation" in entry:
            print(f"  known limitation: {entry['known_limitation']}")
