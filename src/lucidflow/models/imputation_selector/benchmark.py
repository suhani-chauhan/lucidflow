"""Benchmarks the four candidate strategies for a classification-style
target column (company_size, state) by masking a held-out fraction of
known values and scoring recovery via macro-F1.
"""

import pandas as pd
from sklearn.metrics import f1_score

from lucidflow.models.column_type_classifier.split import stratified_min1_split
from lucidflow.models.imputation_selector.strategies import build_strategies

MASK_FRACTION = 0.2
RANDOM_STATE = 42


def benchmark_column(known_df: pd.DataFrame, target_col: str, predictor_cols: list[str], column_kind: str) -> dict:
    """`known_df` must contain only rows where target_col is non-null.

    Uses a guaranteed-min-1-per-class split (real-world class distributions
    here are long-tailed enough that sklearn's strict stratified split
    rejects them outright over classes with a single member) — singleton
    classes land entirely in train and are never scored, same as Task 1's
    boolean class. This is a structural limitation, not an implementation
    gap: a class with exactly one known example cannot be both trained on
    and held out, so macro-F1 below reflects recovery only for classes with
    enough examples to test. See `class_coverage` for how much of the label
    space that actually covers.

    Returns {"scores": {method_name: macro_f1}, "winner": method_name,
    "class_coverage": {...}}.
    """
    class_counts = known_df[target_col].value_counts()
    total_classes = len(class_counts)
    evaluated_classes = int((class_counts >= 2).sum())
    class_coverage = {
        "total_classes": total_classes,
        "evaluated_classes": evaluated_classes,
        "singleton_classes": total_classes - evaluated_classes,
        "coverage_ratio": evaluated_classes / total_classes if total_classes else 0.0,
    }

    train_idx, test_idx, _, _ = stratified_min1_split(
        list(known_df.index), known_df[target_col].tolist(), test_size=MASK_FRACTION, random_state=RANDOM_STATE
    )
    train_rows, test_rows = known_df.loc[train_idx], known_df.loc[test_idx]

    scores = {}
    for strategy in build_strategies(column_kind):
        strategy.fit(train_rows, target_col, predictor_cols)
        y_pred = strategy.predict(test_rows, predictor_cols)
        scores[strategy.name] = f1_score(
            test_rows[target_col], y_pred, average="macro", zero_division=0
        )

    winner = max(scores, key=scores.get)
    return {"scores": scores, "winner": winner, "class_coverage": class_coverage}
