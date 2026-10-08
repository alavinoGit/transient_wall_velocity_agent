# Transient Velocity ML Package

This isolated package reads the repository's existing transient profile CSV and emits training data only for the sustained time window whose profile-wide maximum absolute error against the analytical solution is strictly below 5% of the reference wall speed.

## Run

From the repository root:

```bash
to_ML/run_to_ML.sh
```

The runner verifies its location, installs/verifies only dependencies listed in `to_ML/requirements.txt`, builds the tables, runs package tests, and performs independent validation. It never edits simulation source or writes generated outputs outside `to_ML/`.

## Package contents

- `config/postprocess_config.yaml`: discovered reference-case values, explicit postprocessing policies, and provenance for each value.
- `scripts/build_ml_dataset.py`: input validation, sustained-window selection, feature derivation, summaries, and table generation.
- `scripts/validate_ml_dataset.py`: independent read-only checks of generated tables, source integrity, and physical/numerical invariants.
- `tests/test_build_ml_dataset.py`: deterministic tests for data validation, metrics, acceptance-window logic, field preservation, and output confinement.
- `data/structured_data_2ML.parquet`: typed pointwise training dataset, limited to the sustained accepted window.
- `data/structured_data_2ML.csv`: readable pointwise mirror of the Parquet data.
- `data/structured_data_2ML_profile_summary.csv`: one record for each exported accepted profile.
- `data/structured_data_2ML_case_summary.json`: case-level eligibility and isolated/sustained acceptance window information.
- `data/structured_data_2ML_metadata.json`: case setup, definitions, assumptions, and selected time window.
- `data/structured_data_2ML_schema.json`: field types, units, derivations, provenance, null policy, ranges, and ML roles.
- `data/structured_data_2ML_manifest.csv`: source/output file inventory and SHA-256 digests (the manifest does not hash itself).
- `validation/discovery_report.md`: input discovery and metadata evidence.
- `validation/validation_report.md` and `validation/validation_metrics.json`: independent validation findings.
- `validation/checksums.sha256`: source and generated artifact checksums.
- `logs/postprocess.log`: captured one-command run output.

## Acceptance-window rule

A profile is accepted only when `max_y(abs(u_numerical-u_analytical))/max(abs(U_reference), epsilon) < 0.05`. Because the t=0 profile is an isolated exact match and the 5 s and 10 s profiles fail, it is not part of the exported contiguous training window. The current window is expected to begin at `Fo_nu=0.15` (15 s); the workflow recomputes this from the CSV each run and exports only profiles from the first sustained accepted profile through the last saved profile, provided every later saved profile remains accepted.

The gradient/curvature and wall-distance labels are resolution heuristics, not evidence of mesh convergence. The current simulation has one grid and one timestep, so it cannot establish mesh sensitivity or justify a coarser mesh. Numerical velocity and error fields are quality/diagnostic fields, not input features for predicting the physical analytical profile.
