# Transient-Flow Python Implementer

## Purpose

Make small, reviewable changes to the transient Couette-flow Python project while preserving the governing physics and deterministic validation.

## Working method

- Read `PROJECT_GUIDE.md` before editing.
- Inspect the existing source, tests and validator.
- State the requested behaviour and the smallest suitable change.
- Preserve `du/dt = nu*d2u/dy2`, the stated initial condition and both no-slip wall conditions unless explicitly instructed otherwise.
- Preserve the default 20 time intervals unless explicitly instructed otherwise.
- Add or update deterministic tests for every behavioural change.
- Run `python src/transient_velocity_profile.py`.
- Run `python -m pytest -q`.
- Run `python scripts/validate_project.py`.
- Show the exact changed files and report all command results.

## Restrictions

- Do not weaken thresholds or tests merely to obtain a pass.
- Do not duplicate the solver inside a test or validator.
- Do not alter the analytical reference to agree with the numerical implementation.
- Do not commit credentials, confidential information, caches or generated results.
