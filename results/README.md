# Published Experiment Artifacts

`from_scratch_26cohort_complete25_walk_v1.tar.gz` is the complete
privacy-preserving export of the independent walk-only baseline. It contains
the split manifests, training histories, run configurations, test predictions,
logs, audits, and aggregate metrics for all five binary and five four-class
subject-independent holdouts.

The adjacent `from_scratch_26cohort_complete25_walk_v1` directory exposes the
summary, provenance, integrity audit, and hash inventories for browser-based
inspection. The archive is the authoritative complete text-artifact bundle.

Internal participant IDs were replaced by stable HMAC-SHA256-derived public
codes using a private salt that is not included. Private absolute paths were
replaced by placeholders. `ARTIFACTS.sha256` covers every file inside the
de-identified export, while `CHECKPOINTS.sha256` records the ten excluded model
checkpoint hashes and sizes. Checkpoint binaries and source clinical data are
not published.

These artifacts document a negative from-scratch sensitivity baseline. They do
not establish clinical performance and must not be combined numerically with
the manuscript's archived pre-trained window-level results.
