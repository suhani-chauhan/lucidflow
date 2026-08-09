"""LucidFlow dashboard -- Pipeline Summary (landing page).

Phase 5, Task 2: read-only observability. Reads Postgres directly; no new
backend logic, no recomputation -- see data_access.py's module docstring
for how "most recent run" is derived without a schema change.

    streamlit run dashboard/app.py
"""

import streamlit as st
from data_access import get_latest_quarantine_reasons, get_latest_run_summary
from sqlalchemy.exc import OperationalError

st.set_page_config(page_title="LucidFlow Dashboard", page_icon="📊", layout="wide")

st.title("LucidFlow — Pipeline Summary")
st.caption(
    "Most recent pipeline run, derived from Postgres write timestamps "
    "(no run_id column needed -- see data_access.py)."
)

try:
    summary = get_latest_run_summary()
except OperationalError as exc:
    st.error(
        f"Could not connect to Postgres: {exc}\n\n"
        "Start it with `docker compose up -d postgres` and set the connection "
        "env vars (see .env.example) before running the dashboard."
    )
    st.stop()

if summary is None:
    st.info("No pipeline run has written to Postgres yet. Run `python run_pipeline.py` first.")
    st.stop()

col1, col2, col3 = st.columns(3)
total = summary["clean_count"] + summary["quarantine_count"]
col1.metric("Rows written clean", f"{summary['clean_count']:,}")
col2.metric("Rows quarantined", f"{summary['quarantine_count']:,}")
col3.metric(
    "Quarantine rate",
    f"{100 * summary['quarantine_count'] / total:.2f}%" if total else "n/a",
)

st.caption(
    f"Clean write timestamp: {summary['clean_written_at']}  |  "
    f"Quarantine write timestamp: {summary['quarantine_written_at']}"
)
st.caption(
    "These two timestamps differ slightly because clean.analytics_data and "
    "quarantine.records are written in two separate Postgres transactions within "
    "the same pipeline run (see run_pipeline.py / pipeline_flow.py)."
)

st.divider()
st.subheader("Why rows were quarantined (most recent run)")
reasons_df = get_latest_quarantine_reasons()
if reasons_df.empty:
    st.write("No quarantined rows in the most recent run.")
else:
    st.bar_chart(reasons_df.set_index("rule"))
    st.caption(
        "`quarantine_classifier` = flagged by the trained ML model (passed the Pydantic "
        "contract); all other rule names are contract-validation failures (Phase 1)."
    )
    st.dataframe(reasons_df, hide_index=True, width="stretch")

st.divider()
st.page_link("pages/1_Model_Results.py", label="Model Results →", icon="🧠")
st.page_link("pages/2_Drift_Status.py", label="Drift Status →", icon="📈")
st.page_link("pages/3_Entity_Resolution.py", label="Entity Resolution (investigated, not built) →", icon="🔍")
