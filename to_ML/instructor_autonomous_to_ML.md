# Autonomous instructor: convert transient CFD CSV data into an isolated ML-ready package

## How to invoke this file
From the repository root, ask the coding assistant:

> Read `instructor_autonomous_to_ML.md` and execute every phase in order. Continue autonomously until the completion gate passes. Do not ask me to populate configuration files. Inspect the repository and populate them from authoritative existing files; use `null` plus a warning where a value cannot be established unambiguously.

## Objective
Create a self-contained downstream post-processing workflow that transforms the existing transient velocity-profile CSV into structured data for an ML engineer.

The existing simulation workflow is read-only. Do not edit, move, rename, delete, reformat, import from, or overwrite any pre-existing simulation source, test, configuration, raw output, figure, report, `implementer.md`, or `validator.md`.

Create everything new under `to_ML/`, except this instructor file. The simulation must not call the post-processor. The post-processor must be run separately.

## Non-negotiable isolation rules
1. Take an initial snapshot with `git status --short`, `git diff --stat`, and a SHA-256 checksum of the selected raw CSV.
2. Record pre-existing uncommitted changes and do not modify or revert them.
3. Resolve every output path before writing. Refuse any output path outside `<repository-root>/to_ML/`.
4. Read existing source/configuration files only to discover case metadata. Never edit them.
5. Do not guess missing metadata. Store `null`, identify the missing field, and add a warning.
6. Do not overwrite an existing `to_ML/` package blindly. Preserve its prior data or create a timestamped backup inside `to_ML/backups/` before replacing generated artefacts.
7. The final validator must compare the initial and final repository states and fail if this task changed any pre-existing file outside `to_ML/`.

## Phase 1: inspect and discover inputs
Perform these tasks autonomously:

1. Identify the repository root.
2. Read `implementer.md` and `validator.md` if present.
3. Locate candidate transient CSV files in results/ folder. Prefer a file containing all required columns:
   - `time_s`
   - `fourier_viscous_number`
   - `y_m`
   - `numerical_velocity_m_per_s`
   - `analytical_velocity_m_per_s`
4. If exactly one candidate satisfies the schema, select it.
5. If several candidates satisfy the schema, select the most recently modified only when it is clearly the current simulation output; otherwise process each as a separate case under `to_ML/data/<case_id>/` and record the decision.
6. Locate the simulation source and existing configuration files that define, when available:
   - channel gap;
   - kinematic viscosity;
   - lower- and upper-wall velocities;
   - timestep;
   - spatial scheme;
   - temporal scheme;
   - numerical diffusion number `r` or the quantities needed to calculate it.
7. Treat the source CSV and simulation files as read-only.
8. Write the discovery result to `to_ML/validation/discovery_report.md`.

Do not ask the user to fill a YAML file. Continue with discovered values. Use `null` and a warning for unresolved values.

## Phase 2: create the isolated package structure
Create:

```text
to_ML/
├── README.md
├── requirements.txt
├── run_to_ML.sh
├── config/
│   └── postprocess_config.yaml
├── scripts/
│   └── build_ml_dataset.py
├── tests/
│   └── test_build_ml_dataset.py
├── data/
├── validation/
└── logs/
```

Populate `postprocess_config.yaml` automatically from Phase 1. Include a `provenance` section stating the source of every populated value, for example `raw_csv`, `simulation_source`, `existing_config`, `derived_from_coordinates`, or `unresolved`.

Use these defaults only as declared post-processing policy, not as discovered physics:

- maximum acceptable normalised profile error: `0.05`;
- sustained acceptance required: `true`;
- numerical epsilon: `1.0e-12`;
- Parquet output: `true`;
- CSV mirror: `true`;
- gradient threshold: `0.10`;
- curvature threshold: `0.10`;
- wall-exclusion fraction: `0.05`.

Use the upper-wall speed magnitude as the reference velocity when it is explicitly available and non-zero. Otherwise use the maximum absolute explicitly declared wall velocity. If neither is available, leave the reference velocity unresolved and fail before calculating a percentage acceptance threshold rather than inventing one.

## Phase 3: implement the transformer
Use the implementation guidance in `implementer.md` where available. Create `to_ML/scripts/build_ml_dataset.py`.

The transformer must:

1. Accept `--input`, `--config`, and `--output-root` arguments.
2. Read the raw CSV without modifying it.
3. Validate required columns, numeric types, finite required values, and profile structure.
4. Group profiles using `time_s` and `fourier_viscous_number`.
5. Sort spatial locations by `y_m` inside each profile.
6. Preserve all source values and add:
   - `case_id`;
   - `trajectory_id`;
   - `profile_id`;
   - `y_over_gap`;
   - distance to the nearest wall in dimensional and nondimensional form;
   - recomputed signed and absolute error;
   - normalised absolute error;
   - squared error;
   - local `delta_y`;
   - first and second spatial derivatives calculated within each profile only;
   - normalised gradient and curvature;
   - numerical diffusion number `r`, if present or unambiguously derivable;
   - fine/coarse-grid heuristic labels;
   - profile-acceptance and accepted-window labels.
7. Calculate profile-level maximum absolute error, MAE, RMSE, relative L2 error, location of maximum error, maximum gradient and curvature, fine-grid fraction, and largest contiguous coarse candidate region.
8. Define profile acceptance as:

```text
max_y(abs(u_numerical - u_analytical))
------------------------------------------------  < 0.05
max(abs(reference_velocity), epsilon)
```

9. Report separately:
   - first isolated accepted profile;
   - first profile after which every later saved profile remains accepted;
   - accepted-window start and end.
10. Document that gradient/curvature mesh labels are heuristics, not proof of mesh convergence.
11. Write only under `to_ML/`.

## Phase 4: generate the ML package
Generate these files automatically:

### Primary pointwise data
- `to_ML/data/structured_data_2ML.parquet`
- `to_ML/data/structured_data_2ML.csv`

Each row represents one case, one saved profile, and one spatial location. Parquet is the principal typed ML file; CSV is the readable audit mirror.

### Profile and case summaries
- `to_ML/data/structured_data_2ML_profile_summary.csv`
- `to_ML/data/structured_data_2ML_case_summary.json`

The profile summary contains one row per saved time. The case summary records training eligibility and isolated/sustained acceptance windows.

### Interpretation and provenance
- `to_ML/data/structured_data_2ML_metadata.json`
- `to_ML/data/structured_data_2ML_schema.json`
- `to_ML/data/structured_data_2ML_manifest.csv`

The schema must define field name, type, units, dimensional status, description, source, derivation, null policy, valid range when applicable, and ML role (`feature`, `target`, `sample_weight`, `grouping_key`, `quality_label`, or `provenance_only`).

### Operational evidence
- `to_ML/logs/postprocess.log`
- `to_ML/validation/checksums.sha256`

## Phase 5: create persistent tests
Create `to_ML/tests/test_build_ml_dataset.py`.

At minimum, test:

1. output-path confinement beneath `to_ML/`;
2. required-column validation;
3. raw and structured row-count equality;
4. unique `profile_id × y_m` records;
5. monotonically increasing coordinates within each profile;
6. exact preservation of source numerical and analytical velocity values;
7. correct recomputation of pointwise errors;
8. derivatives calculated separately within profiles;
9. correct MAE, RMSE, maximum-error, and relative-L2 summaries;
10. correct isolated-versus-sustained acceptance using a synthetic pattern;
11. null-plus-warning treatment for absent `r`;
12. numerical equivalence of shared Parquet and CSV columns;
13. manifest and checksum consistency;
14. no modification of the source CSV.

Run:

```bash
pytest -q to_ML/tests/test_build_ml_dataset.py
```

## Phase 6: run independent validation
Use the existing `validator.md` as an independent review role where available. The validator must not rewrite the transformer unless a specific failure is demonstrated.

Validate:

- physical meaning and units of every derived field;
- error normalisation and 5% threshold;
- `Fo_nu` handling;
- `r` handling and provenance;
- no derivative calculation across separate times;
- mesh-indicator limitations;
- absence of target leakage in suggested ML roles;
- pointwise/profile/case-summary consistency;
- package completeness;
- source checksum stability;
- repository isolation.

Write:

- `to_ML/validation/validation_report.md`
- `to_ML/validation/validation_metrics.json`

## Phase 7: document and provide one-command execution
Create `to_ML/README.md` explaining every file and its intended user.

Create `to_ML/requirements.txt` containing only required dependencies.

Create executable `to_ML/run_to_ML.sh` that:

1. verifies execution from the repository root;
2. installs or verifies dependencies without modifying simulation source;
3. runs the transformer using the auto-populated configuration;
4. runs the test suite;
5. invokes or performs validation;
6. prints the paths of all generated outputs and the final pass/fail status.

Do not make the existing simulation invoke this shell script.

## Phase 8: final integrity check
Repeat:

```bash
git status --short
git diff --stat
```

Compare with the initial snapshot. Distinguish pre-existing changes from changes created by this task.

Recalculate the source CSV SHA-256 hash and compare it with the initial hash.

The task must fail if:

- the source checksum changed;
- a pre-existing file outside `to_ML/` changed because of this task;
- row counts differ unexpectedly;
- tests fail;
- required package files are absent;
- percentage acceptance is calculated without a defensible reference velocity.

## Phase 9: completion response
Do not merely say that implementation is complete. Report:

1. selected raw CSV path;
2. discovered metadata and provenance;
3. unresolved metadata and warnings;
4. raw rows, profiles, and spatial nodes;
5. first isolated and first sustained accepted `Fo_nu` and times;
6. all generated file paths;
7. automated-test result;
8. independent-validation result;
9. source checksum result;
10. list of files changed outside `to_ML/`, which must be empty apart from this instructor file if newly added.

Continue autonomously through all phases. Ask the user only if no usable CSV can be identified or a required physical definition is genuinely ambiguous and prevents a scientifically valid result. Do not ask the user to populate configuration files manually.
