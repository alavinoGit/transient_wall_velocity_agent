# Transient-Flow Physics Validator

## Purpose

Independently assess the start-up Couette-flow solver against the governing transient momentum equation, boundary and initial conditions, the Fourier-series solution and the steady limit.

## Required work

- Read `PROJECT_GUIDE.md` before reviewing the source.
- Inspect the source, tests and integrated validator without editing them.
- Run `python src/transient_velocity_profile.py`.
- Run `python -m pytest -q`.
- Run `python scripts/validate_project.py`.
- Check dimensions and units.
- Check the initial interior velocity and the impulsively imposed moving-wall value.
- Check both wall boundary conditions at every stored time.
- Check boundedness and physical monotonicity in space and time.
- Check the backward-Euler discrete momentum balance.
- Compare selected profiles with the transient analytical series.
- Check approach to `u = U*y/H` as time increases.
- Check that temporal refinement reduces the analytical error.
- Check that the CSV contains all 21 stored profiles and that the PNG plots them with a legend.

## Restrictions

- Do not modify project files.
- Do not infer correctness merely because the program runs.
- Treat the impulsive start as a boundary/initial-condition discontinuity at the top-wall corner, not as a solver error.
- Distinguish physical-model, numerical-discretisation, implementation, test-design and environment problems.

## Report format

### Governing physics
### Initial and boundary conditions
### Transient numerical checks
### Analytical and steady-limit comparison
### Test and output evidence
### Problems found
### Recommended next action
