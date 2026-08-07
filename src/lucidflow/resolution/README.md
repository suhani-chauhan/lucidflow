# resolution/

Investigated, not built. Entity resolution (an LLM-distilled duplicate-pair classifier) was
scoped for Phase 3, but the premise didn't hold up against the actual dataset — see
[`docs/entity_resolution_investigation.md`](../../../docs/entity_resolution_investigation.md) for
the full investigation, real examples, and counts. Phase 3's trained model is the quarantine
classifier instead (`models/quarantine_classifier/`).
