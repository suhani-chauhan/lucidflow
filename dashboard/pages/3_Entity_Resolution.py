"""LucidFlow dashboard -- Entity Resolution (investigated, not built).

Surfaces the documented negative result from Phase 3 directly -- renders
docs/entity_resolution_investigation.md verbatim rather than re-summarizing
it, so the dashboard can't drift out of sync with the actual writeup.
"""

import streamlit as st
from data_access import get_entity_resolution_doc

st.set_page_config(page_title="Entity Resolution — LucidFlow", page_icon="🔍", layout="wide")
st.title("Entity Resolution: Investigated, Not Built")

st.info(
    "A fourth trained model (an LLM-distilled duplicate-pair classifier) was scoped for "
    "Phase 3, investigated against the real dataset across two separate passes, and "
    "**deliberately dropped** -- not because it wasn't attempted, but because the data "
    "didn't support it. Documented here rather than silently dropped from scope."
)

doc_text = get_entity_resolution_doc()
if doc_text is None:
    st.error("docs/entity_resolution_investigation.md not found.")
else:
    st.markdown(doc_text)
