# Independent Validation Report

## Governing physics
PASS: Raw analytical profiles match the existing analytical reference at all 21 stored times.
PASS: Backward-Euler discrete momentum residual max is 1.119e-14.
PASS: Fo_nu was independently recomputed as nu*time/H^2; r was independently recomputed as nu*dt/dy^2.

## Initial and boundary conditions
PASS: Initial interior is at rest and both wall conditions hold at every stored time.
PASS: Velocity remains bounded and is monotone in time and space.

## Transient numerical checks
PASS: Raw case has 21 profiles x 51 nodes = 1071 source rows.
PASS: Export includes only 18 sustained accepted profiles (918 rows) from Fo_nu=0.15 to 1.00.
PASS: Normalized profile maximum error remains below 5%; largest exported fraction is 0.049959944.
PASS: No failing or isolated early-pass profile samples appear in either training table.

## Analytical and steady-limit comparison
PASS: Final absolute distance from the steady profile is 0.00020926271 m/s.
PASS: Halving timestep reduces final analytical error (0.0001763347 to 6.1224288e-05 m/s).

## Test and output evidence
PASS: Recorded package tests passed; Parquet and CSV point rows agree; profile summaries, schema, and manifest were independently checked.
PASS: Source CSV SHA-256 is unchanged: `32450e06346194e13299a78934d3b7b0eb6fd50e4918e6fbba6ae712fdb6be98`.
PASS: Repository status outside to_ML matches the initial package baseline.

## Problems found
No correctness failures found in the generated package.
The isolated t=0 match is documented but excluded from training rows; the 5 s and 10 s failing profiles are excluded from all pointwise/profile-summary training tables.
Gradient/curvature-based fine/coarse labels are heuristics only.

## Recommended next action
Use only the sustained accepted-window rows for the current ML baseline. Generate separate mesh and timestep refinement cases before making any mesh-sensitivity or safe-coarsening claim.

## Acceptance-window record
- First isolated accepted profile: t=0.00 s, Fo_nu=0.00.
- First sustained accepted profile: t=15.00 s, Fo_nu=0.15.
- Exported interval: t=15.00..100.00 s.
