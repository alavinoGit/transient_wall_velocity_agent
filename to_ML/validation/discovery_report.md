# Input Discovery and Provenance

## Repository snapshot

- Repository root: `/workspaces/transient_wall_velocity_agent`
- Branch at discovery: `postprocess-to-ml`
- Initial `git status --short`: `?? to_ML/`; this was a pre-existing untracked package containing instructor files and `transient_velocity_ml_dataset.json`. These files were preserved.
- Initial `git diff --stat`: no tracked changes reported.
- Initial source CSV SHA-256: `32450e06346194e13299a78934d3b7b0eb6fd50e4918e6fbba6ae712fdb6be98`

## Selected input

Exactly one candidate CSV was found in `results/`: `results/transient_velocity_profiles.csv`. It contains 1,071 rows, 21 unique saved times, and 51 spatial nodes per profile. Its fields are `time_s`, `fourier_viscous_number`, `y_m`, `numerical_velocity_m_per_s`, `analytical_velocity_m_per_s`, and `error_m_per_s`.

## Discovered run metadata

- Channel height: 0.01 m, from `generate_outputs` in `src/transient_velocity_profile.py`.
- Upper-wall speed: 1.0 m/s, from `generate_outputs`; lower wall is 0 m/s from the solver boundary assignment.
- Kinematic viscosity: 1.0e-6 m^2/s, from `generate_outputs`.
- Time increment: 5 s, derived from CSV times; 20 intervals and 21 profiles.
- Spatial increment: 0.0002 m, derived from CSV coordinates; 51 nodes.
- Temporal method: Backward Euler; spatial method: centered second differences, from the solver docstring and implementation.
- `Fo_nu = nu*time/H^2`: values 0.00 through 1.00 in 0.05 increments.
- `r = nu*dt/dy^2`: derived as 125 from discovered nu, dt, and dy.
- The CSV's supplied signed error is numerical minus analytical; postprocessing verifies this before deriving pointwise error features.

## Acceptance window

Using the declared 5% strict profile-wide maximum-error limit and `U_reference=1.0 m/s`, the t=0 profile passes in isolation, t=5 s and t=10 s fail, and every saved profile from t=15 s through t=100 s passes. The sustained export window therefore begins at `Fo_nu=0.15` and ends at `Fo_nu=1.00`. Only this contiguous sustained window is exported as pointwise training rows; the isolated t=0 profile is recorded in the case summary but its pointwise samples are excluded.

## Role instructions and unresolved values

`to_ML/implementer.md` and `to_ML/validator.md` are absent. The available repository equivalents are `agent_specs/python_implementer.md` and `agent_specs/physics_validator.md`; both were read. No required physical value is unresolved for this reference case. The one-grid data do not support a mesh-convergence conclusion.
