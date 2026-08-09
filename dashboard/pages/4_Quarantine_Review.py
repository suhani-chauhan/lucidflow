"""LucidFlow dashboard -- Human-in-the-Loop Quarantine Review (Phase 5, Task 3).

The quarantine classifier was trained entirely on synthetic corruption
(Task 0 grounding: real companies.csv has 0 organic contract failures, and
the model has never seen a confirmed-real corrupt row). This page is what
eventually gives it real-world labels: reviewers look at real quarantined
records the model actually flagged in production and mark each one
'Confirmed bad' or 'False positive'.

One-at-a-time review queue, one decision per record (locked after first
review -- see docker/init/001_init.sql's UNIQUE(record_id) constraint).
Only rows the ML classifier flagged are shown, not deterministic
Pydantic-contract failures -- those aren't model predictions, and Phase 3's
own reasoning throughout this project is that training on them teaches
nothing new (the gate already catches them with perfect precision).

Honesty note, upfront: real companies.csv was already established (Task 0,
Phase 3) to have zero organic corruption. So an honest reviewer should
expect most, possibly all, real flags to be legitimate false positives --
that is itself valid, useful signal (it teaches the model what it's
currently over-flagging on real data), not a failed review process. Don't
expect a stream of 'confirmed bad' results.
"""

import streamlit as st
from data_access import (
    MIN_REVIEWS_FOR_RETRAIN,
    get_registered_quarantine_model_metrics,
    get_review_progress,
    get_review_queue,
    submit_review,
    trigger_retrain,
)

st.set_page_config(page_title="Quarantine Review — LucidFlow", page_icon="🔎", layout="wide")
st.title("Human-in-the-Loop Quarantine Review")

st.warning(
    "**Expectation-setting**: Task 0 established real `companies.csv` has zero organic "
    "contract failures. An honest review of real ML flags is expected to skew heavily "
    "toward *false positive* -- that's a legitimate, useful finding about the model's "
    "current behavior on real data, not a sign the review is going wrong."
)

if "review_queue" not in st.session_state:
    st.session_state.review_queue = get_review_queue()
    st.session_state.review_index = 0

queue = st.session_state.review_queue
progress = get_review_progress()

st.divider()
col1, col2, col3 = st.columns(3)
col1.metric("Reviewed so far", progress["total"])
col2.metric("Confirmed bad", progress["confirmed_bad"])
col3.metric("False positive", progress["false_positive"])
st.progress(min(progress["total"] / MIN_REVIEWS_FOR_RETRAIN, 1.0))
st.caption(f"{progress['total']} / {MIN_REVIEWS_FOR_RETRAIN} reviews needed to unlock retraining.")

st.divider()

if not queue:
    st.info("No unreviewed quarantined records right now. Run the pipeline to generate more.")
elif st.session_state.review_index >= len(queue):
    st.success(f"Reviewed all {len(queue)} records in this queue. Reload the page to fetch any new ones.")
else:
    record = queue[st.session_state.review_index]
    st.subheader(f"Record {st.session_state.review_index + 1} of {len(queue)}")
    st.caption(f"quarantine.records.id = {record['record_id']}  |  quarantined at {record['quarantined_at']}")

    left, right = st.columns([3, 2])
    with left:
        st.write("**Raw record**")
        st.json(record["raw_data"], expanded=True)
    with right:
        st.metric("Anomaly score", f"{record['score']:.4f}")
        st.caption(record["message"])
        st.caption(f"Model version at flag time: {record['model_version'] or 'n/a'}")
        with st.expander("Feature vector the classifier scored"):
            st.json(record["features"])

    st.write("")
    button_col1, button_col2, button_col3 = st.columns(3)
    if button_col1.button("✅ Confirmed bad", key=f"bad_{record['record_id']}", width="stretch"):
        submit_review(record["record_id"], "confirmed_bad")
        st.session_state.review_index += 1
        st.rerun()
    if button_col2.button("❌ False positive", key=f"fp_{record['record_id']}", width="stretch"):
        submit_review(record["record_id"], "false_positive")
        st.session_state.review_index += 1
        st.rerun()
    if button_col3.button("⏭️ Skip (don't record a decision)", key=f"skip_{record['record_id']}", width="stretch"):
        st.session_state.review_index += 1
        st.rerun()

st.divider()
st.header("Retrain Model 3 with human-reviewed labels")

registered = get_registered_quarantine_model_metrics()
if registered:
    st.caption(
        f"Currently registered: version {registered['version']}, "
        f"synthetic-test PR-AUC {registered['pr_auc']:.4f}" if registered["pr_auc"] is not None
        else f"Currently registered: version {registered['version']} (no PR-AUC metric found on its run)"
    )
else:
    st.caption("No quarantine classifier version is currently registered.")

if progress["total"] < MIN_REVIEWS_FOR_RETRAIN:
    st.button(
        f"Retrain (locked -- {progress['total']}/{MIN_REVIEWS_FOR_RETRAIN} reviews so far)",
        disabled=True,
    )
else:
    st.caption(
        "Retraining combines the synthetic-corruption dataset with all human-reviewed rows "
        "above, evaluates on the same fixed synthetic held-out split train.py uses (so PR-AUC "
        "is directly comparable to the currently-registered version), and only registers a "
        "new version if that PR-AUC doesn't regress by more than 0.01 -- with this few "
        "human-reviewed examples, requiring strict improvement would be measuring noise, not "
        "signal. The human-reviewed rows also get a small separate eval, reported but never "
        "used for that registration decision."
    )
    if st.button("🔁 Retrain now", type="primary"):
        with st.spinner("Retraining -- rebuilding the synthetic dataset and fitting a new model, roughly a minute..."):
            result = trigger_retrain()

        st.success(f"Retrain run logged: `{result['run_id']}`")
        col1, col2 = st.columns(2)
        col1.metric("Synthetic-test PR-AUC (candidate)", f"{result['synthetic_pr_auc']:.4f}")
        col2.metric(
            "Synthetic-test PR-AUC (previously registered)",
            f"{result['registered_pr_auc']:.4f}" if result["registered_pr_auc"] is not None else "n/a",
        )
        st.write(
            f"Human-reviewed rows used: {result['n_human_reviewed']} "
            f"({result['n_human_confirmed_bad']} confirmed_bad, {result['n_human_false_positive']} false_positive)"
        )
        if result["human_eval"]:
            st.caption(
                f"Small supplementary human-reviewed eval (n={result['human_eval']['n']}, NOT used for "
                f"the registration decision above): precision={result['human_eval']['precision']:.4f}, "
                f"recall={result['human_eval']['recall']:.4f}"
            )
        else:
            st.caption("Too few human-reviewed rows to hold any out for a separate eval this time.")

        if result["should_register"]:
            st.success(f"Registered as new version {result['registered_new_version']}.")
        else:
            st.warning(
                "Not registered: synthetic-test PR-AUC regressed by more than the 0.01 tolerance "
                "versus the currently-registered version. The run is still logged to MLflow for "
                "inspection, but the previously-registered model is still what's deployed."
            )
