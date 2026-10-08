# Instructor 2: ML Postprocessing for Transient Velocity Profiles

## Purpose and output design

Post-process the existing transient Couette-flow results into an auditable, ML-ready dataset. The primary deliverable should be `to_ML/transient_velocity_ml_dataset.json`: one self-describing JSON document with a schema version, run metadata, one summary per stored profile, and a flat `samples` array with one record per `(time, y)` point. A flat sample table keeps each training example easy to load while the metadata and summaries retain the physics and provenance. An optional companion CSV is acceptable only if it is also written under `to_ML` and contains the same sample fields.

The current raw input is `results/transient_velocity_profiles.csv`. Its columns are `time_s`, `fourier_viscous_number`, `y_m`, `numerical_velocity_m_per_s`, `analytical_velocity_m_per_s`, and `error_m_per_s`. The current solver is `src/transient_velocity_profile.py` (read-only for this task); inspect it to obtain the actual numerical scheme and calculate the diffusion number. Do not assume an input column exists if it does not.

Suggested JSON layout:

- `schema_version`: version the output contract.
- `source`: input path, row count, source checksum if practical, and provenance/limitations.
- `run_metadata`: fluid/model description; `H`, `U`, `nu`; `dt`, `dy`, point and interval counts; spatial and temporal scheme; initial/boundary conditions; units; `Fo_nu` definition; `r` definition and value; and known/missing parameters. Use explicit units in field names or metadata. Store unknown values as `null` with a reason, never invent them.
- `profile_summaries`: one record per time containing `time_s`, `Fo_nu`, maximum absolute analytical discrepancy normalized by `U`, normalized RMSE, and gradient diagnostics. Report the first *stored* `Fo_nu` for which the profile-wide normalized maximum error is strictly below 0.05; report `null` if none meets it.
- `samples`: one record per original `(time, y)` sample, including a stable `profile_id`, dimensional and dimensionless coordinates (`y_m`, `y_over_H`), `time_s`, `Fo_nu`, `r`, numerical and analytical velocity (dimensional and, if useful, divided by `U`), signed error, absolute error normalized by `U`, and the local dimensionless gradient `d(u/U)/d(y/H)`.

The error here is the deterministic discrepancy against the analytical solution, not statistical uncertainty. Name it accordingly. If the desired ML task is prediction of the physical transient solution, make the analytical solution the reference target and keep numerical velocity/error out of the feature set to avoid target leakage. If the task is to emulate this particular numerical solver, state that separately and use numerical velocity as the target. The output must make feature/target roles explicit rather than silently mixing them.

## Interpretation and additional ML context

- Define `Fo_nu = nu*t/H^2`, a channel-scale transient coordinate.
- Define `r = nu*dt/dy^2`, the cell-scale viscous diffusion number. Include it as a numerical-method descriptor/feature for now, and explain that it is not itself an accuracy guarantee. Backward Euler is implicit, but stability does not imply adequate transient accuracy.
- The reference case has 21 profiles and 51 positions per profile (1,071 point samples), with one physical parameter combination and one grid/time-step combination. This is useful for a pipeline test or baseline, not evidence of generalization across fluids, geometries, resolutions, or numerical schemes. Do not claim otherwise.
- For each profile, calculate normalized maximum error as `max_y(abs(u_num-u_analytical))/U`; use the supplied signed error after verifying its sign and units. Also report normalized RMSE, with the spatial quadrature/weighting convention stated. Screen the 5% criterion using the profile-wide maximum, not a pointwise average. Keep the strict condition `< 0.05` explicit.
- Derive spatial gradients consistently from the sampled profile and document the numerical differentiation method and edge treatment. Report the peak dimensionless gradient and the extent of high-gradient regions. If using a relative threshold such as 10% of that profile's peak gradient, record the threshold and define the distance as `y/H` from the nearest wall. These are descriptive diagnostics, not proof that a coarser grid is safe.
- Mesh sensitivity and safe coarsening require additional simulations. Recommend a study that varies spatial resolution at fixed `dt`, then temporal resolution at fixed `dy`, and compares each run to the analytical solution (and, where useful, a fine-grid reference). Record `r` for every run because changing `dy` or `dt` changes `r`; do not attribute coupled changes to one factor. A single run cannot establish how mesh requirements depend on `Fo_nu` or `r`.
- Preserve the impulsive-start corner discontinuity and boundary values. Include the `t=0` profile, but flag the moving-wall corner convention so it is not mistaken for an ordinary smooth initial profile.
- Preserve sufficient run provenance to reproduce the result: source/code revision if available, parameter values, discretization, data-generation date if known, and all assumptions. Avoid claiming a statistical sample size or uncertainty estimate from one deterministic simulation.

## Codespaces implementation prompt

Use the following task prompt in GitHub Codespaces:

> ### Task
> Implement and independently validate the post-processing workflow described in `to_ML/instructor_2_ML.md`. The goal is to convert the existing transient Couette-flow CSV into an auditable ML-ready structured dataset, including profile-wise analytical-error screening, the numerical diffusion number `r`, and descriptive mesh/gradient information.
>
> ### Role specifications
> 1. Read `to_ML/instructor_2_ML.md`, `to_ML/implementer.md`, and `to_ML/validator.md` before changing files. The implementer and validator must be distinct stages: implement first, then conduct a read-only independent validation. Do not let the validator modify code or data.
> 2. If `to_ML/implementer.md` or `to_ML/validator.md` is not present, check for the repository equivalents `agent_specs/python_implementer.md` and `agent_specs/physics_validator.md`. If neither the named file nor its equivalent exists, stop and report the missing role spec instead of inventing its requirements.
>
> ### Strict write boundary
> 1. Never create, edit, delete, format, or overwrite any file outside `to_ML/`. This includes source code, tests, project documentation, original CSV/JSON/PNG outputs, caches, and validation artifacts.
> 2. Treat `results/transient_velocity_profiles.csv`, `src/transient_velocity_profile.py`, project tests, and project guides as read-only inputs. The current solver path is `src/transient_velocity_profile.py`; inspect its `diffusion_number` calculation to derive `r = nu*dt/dy**2`, but do not edit it.
> 3. Put all post-processing implementation, tests, structured outputs, logs, plots, and temporary validation artifacts under `to_ML/`. Do not run a command that writes to the repository's existing `results/` or elsewhere. If an existing script/test writes outside `to_ML`, do not invoke it as-is; use a safe output override or an equivalent read-only/in-memory check. Disable bytecode and pytest cache writes when running checks (for example, `PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider`).
> 4. Do not commit or push.
>
> ### Implementation requirements
> 1. Inspect the raw CSV header, row count, unique times and positions, and source solver parameters before implementation. Validate input columns, numeric values, profile completeness, monotonic/consistent coordinate grids, and finite values; fail clearly rather than silently dropping or fabricating data.
> 2. Keep the post-processing code and its focused tests inside `to_ML/`. Use established project dependencies where practical; do not modify dependency files outside `to_ML/`.
> 3. Generate `to_ML/transient_velocity_ml_dataset.json` with the schema described above: version, provenance, model/run metadata and units, profile summaries, and flat point samples. Include a companion flat CSV only if useful; any outputs must stay inside `to_ML/`.
> 4. Read `nu`, `dt`, `dy`, grid counts, and scheme from the actual solver/run configuration. Calculate `r = nu*dt/dy**2`; do not infer `r` from the CSV or hardcode it without checking the source. Store it at run and sample level. If the CSV and source disagree or a required value cannot be established, report the mismatch or leave it explicitly unknown; do not edit upstream files to resolve it.
> 5. Validate `Fo_nu = nu*time/H**2` against every stored profile and against the CSV values. For the documented reference case, expect 21 values from 0.00 through 1.00 in increments of 0.05. Preserve enough precision in machine-readable values; rounding is for reports only.
> 6. Verify the signed-error convention from the CSV/source. For each profile compute `max_y(abs(u_num-u_analytical))/U`, normalized RMSE with its weighting convention, and the first stored `Fo_nu` satisfying strict `< 5%`. Report the selected time and `Fo_nu`, or explicitly report that no sampled profile qualifies. Call this analytical discrepancy/error, not statistical error.
> 7. Emit dimensionless coordinates and velocities where useful. Document gradient computation, boundary treatment, threshold used for high-gradient extent (if any), and gradient units. Any gradient-distance/coarser-grid field must be clearly labeled descriptive. Do not state a coarser mesh is acceptable based on a single grid.
> 8. Define the intended ML feature/target roles and prevent leakage. For physical-solution prediction, analytical velocity is the reference target; numerical velocity and its error are not input features. If a different task is chosen, label it explicitly. State that the present 1,071 samples cover one parameter/grid configuration and are not a generalization dataset.
> 9. Preserve `t=0`, wall conditions, the impulsive moving-wall corner convention, and all original raw values. Do not modify the numerical solver or weaken any project tests/thresholds.
>
> ### Required validation and report
> 1. Add focused deterministic tests under `to_ML/` for parsing/validation, row and profile counts, `Fo_nu`, `r`, error summaries and the 5% selection, gradient diagnostics, schema fields, finite values, and feature/target leakage rules.
> 2. Independently compare generated sample count and values to the source CSV; expected reference count is 21 x 51 = 1,071. Recompute `Fo_nu`, `r`, and analytical-error summaries independently in the validator stage. Check all 21 profiles and boundary samples.
> 3. Run post-processing and its tests with every generated file/cache redirected or contained under `to_ML/`. Existing project checks may be run only when you have confirmed they will not write outside `to_ML`; otherwise report that restriction and perform equivalent read-only checks without invoking the writing command.
> 4. In the final report list every file created/changed (all must be under `to_ML`), the JSON schema and ML target choice, `Fo_nu`/`r` ranges, first stored `<5%` profile, gradient-threshold definition, validation commands/results, and any limitation. Explicitly state that mesh/refinement sensitivity and a safe coarse-grid recommendation remain unproven until multiple grids/time steps are processed.
