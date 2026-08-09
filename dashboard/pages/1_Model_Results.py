"""LucidFlow dashboard -- Model Results.

Reads the latest MLflow run per experiment (Phase 4, Task 1's tracking) and
renders the same metrics each train.py already prints -- nothing here
recomputes a score or re-runs a benchmark.
"""

import pandas as pd
import streamlit as st
from data_access import (
    get_column_type_classifier_results,
    get_imputation_benchmark_results,
    get_quarantine_classifier_results,
)

st.set_page_config(page_title="Model Results — LucidFlow", page_icon="🧠", layout="wide")
st.title("Model Results")
st.caption("Latest MLflow run per model. See root README.md for the full narrative behind each.")

st.header("Semantic column-type classifier")
column_type = get_column_type_classifier_results()
if column_type is None:
    st.info("No MLflow run found. Train it with `python -m lucidflow.models.column_type_classifier.train`.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Macro-F1 (held-out)", f"{column_type['macro_f1']:.4f}" if column_type["macro_f1"] else "n/a")
    col2.metric("Test columns", column_type["n_test"] or "n/a")
    col3.metric("Misclassified", column_type["n_misclassified"] or "0")
    if column_type["confusion_matrix_text"]:
        st.text("Confusion matrix (rows=true, cols=predicted):")
        st.code(column_type["confusion_matrix_text"], language=None)
    st.caption(
        "Known limitation: the only `boolean` examples in this dataset are numeric-coded "
        '("0"/"1") -- the model has never seen a text-coded ("true"/"false") boolean. '
        "Documented and regression-tested, not silently wrong."
    )

st.divider()
st.header("Learned imputation selector")
imputation = get_imputation_benchmark_results()
if imputation is None:
    st.info("No MLflow run found. Run the pipeline once (`python run_pipeline.py`) to benchmark it.")
else:
    for column, result in imputation["benchmarked"].items():
        st.subheader(f"`{column}`")
        scores_df = pd.DataFrame(
            [{"method": m, "macro_f1": s} for m, s in result["scores"].items()]
        ).sort_values("macro_f1", ascending=False)
        st.dataframe(scores_df, hide_index=True, width="stretch")
        st.write(f"**Winner:** `{result['winner']}`")
        if result["coverage_ratio"] is not None:
            st.write(
                f"Class coverage: {int(result['evaluated_classes'])}/"
                f"{int(result['evaluated_classes']) + int(result['singleton_classes'])} classes "
                f"evaluated ({result['coverage_ratio']:.1%})"
            )
        if result["caveat"]:
            st.warning(result["caveat"])

    if imputation["group_mode"]:
        st.subheader("Group-mode fallback columns")
        group_df = pd.DataFrame(
            [
                {"column": col, "grouped_by": r["group_cols"], "group_key_hit_rate": r["hit_rate"]}
                for col, r in imputation["group_mode"].items()
            ]
        )
        st.dataframe(group_df, hide_index=True, width="stretch")
        st.caption(
            "Hit rate = fraction of missing values whose group key had a known mode; the "
            "rest fell through to the column's global mode."
        )

    if imputation["skipped"]:
        st.caption(
            f"Skipped (left null, no fallback attempted): {', '.join(f'`{c}`' for c in imputation['skipped'])}"
        )

st.divider()
st.header("Quarantine classifier")
quarantine = get_quarantine_classifier_results()
if quarantine is None:
    st.info("No MLflow run found. Train it with `python -m lucidflow.models.quarantine_classifier.train`.")
else:
    col1, col2 = st.columns(2)
    col1.metric("Aggregate PR-AUC", f"{quarantine['pr_auc']:.4f}" if quarantine["pr_auc"] else "n/a")
    col2.metric(
        "Clean-row false-positive rate",
        f"{quarantine['clean_false_positive_rate']:.2%}" if quarantine["clean_false_positive_rate"] is not None else "n/a",
    )
    st.write(
        f"Aggregate precision/recall/F1 @ threshold: "
        f"**{quarantine['precision']:.4f} / {quarantine['recall']:.4f} / {quarantine['f1']:.4f}**"
    )
    per_type_df = pd.DataFrame(
        [
            {"corruption_type": t, **metrics}
            for t, metrics in quarantine["per_type"].items()
        ]
    ).sort_values("recall", ascending=False)
    st.dataframe(per_type_df, hide_index=True, width="stretch")
    st.caption(
        "Per-type precision differences share a common false-positive pool from clean rows; "
        "recall is the cleaner per-type signal (see root README.md for the full breakdown)."
    )
