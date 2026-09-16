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

`from_scratch_walk_headturn_primary20_v1.tar.gz` is the complete
privacy-preserving export of a paired comparison between the walk-only model
and the same model with one reviewed head-turn range token. The corresponding
browser-readable directory contains aggregate metrics, the eligibility audit,
export metadata, and integrity inventories for 20 participants and 806 aligned
walking windows. The archive contains all de-identified split manifests,
training histories, predictions, logs, and provenance. The 20 excluded
checkpoints are represented by hashes and sizes only.

This second experiment is a preliminary 50%-overlap, same-cohort smoke
analysis. Its five repeated holdouts contain only four test participants each,
so the reported changes are descriptive, are not final manuscript results, and
do not establish clinical efficacy. See
[`docs/HEADTURN_EXPERIMENT.md`](../docs/HEADTURN_EXPERIMENT.md) for the protocol,
results, and interpretation boundaries.

`from_scratch_walk_headturn_dense_8gpu_v2` contains the concise acceptance
summary for the final 80%-overlap matrix: 40 successful runs, 40 metric files,
40 non-empty checkpoints, and no partition-integrity failures. The full AML
artifacts are intentionally not duplicated here because the checkpoints alone
are large. The aggregate values and limitations are documented in
[`docs/DENSE_HEADTURN_EXPERIMENT.md`](../docs/DENSE_HEADTURN_EXPERIMENT.md).

