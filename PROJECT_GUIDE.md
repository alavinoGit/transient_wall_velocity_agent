# Transient Couette-Flow Project Guide

## Purpose

This project solves the start-up of plane Couette flow after the upper plate is impulsively set into motion, using an implicit finite-difference method and 20 equal time intervals.

## Physical model

- Two infinite parallel plates are separated by `H`.
- The Newtonian fluid is initially at rest.
- At `t = 0`, the upper plate at `y = H` moves at constant speed `U`.
- The lower plate at `y = 0` remains stationary.
- Density and kinematic viscosity `nu` are constant.
- There is no pressure gradient and no advection in the reduced one-dimensional model.

Governing equation:

```text
du/dt = nu*d2u/dy2
```

Conditions:

```text
u(y,0) = 0 for 0 <= y < H
u(0,t) = 0
u(H,t) = U for t >= 0
```

The value at `(y=H,t=0)` follows the imposed impulsive wall condition, creating the classical initial/boundary corner discontinuity.

Steady limit:

```text
u_steady(y) = U*y/H
```

## Reference case

```text
H = 0.01 m
U = 1.0 m/s
nu = 1.0e-6 m^2/s
t_final = 100 s
n_space = 51
n_time_intervals = 20
Delta t = 5 s
```

Twenty intervals produce 21 stored profiles, including `t = 0` and `t = 100 s`.

## Numerical method

Backward Euler is used in time and centred second differences in space; the resulting tridiagonal linear system is solved at each time interval.

## Acceptance criteria

1. Exactly 20 time intervals and 21 stored profiles are returned.
2. The interior fluid is initially at rest.
3. Both wall velocities are exact at every stored time.
4. Velocity remains between `0` and `U`.
5. Interior velocity does not decrease with time.
6. Every profile increases monotonically from the stationary to moving wall.
7. The backward-Euler discrete PDE residual is below `1e-10`.
8. Selected profiles agree with the analytical transient series within time-specific tolerances (0.035 m/s at 25 s, 0.010 m/s at 50 s and 0.002 m/s at 100 s), recognising the deliberately coarse 20-interval backward-Euler discretisation.
9. The final profile approaches the steady linear Couette solution.
10. Reducing the time step reduces the final analytical error.
11. The unit tests pass.
12. The expected CSV and PNG are generated.

## Commands

```bash
python -m pip install -r requirements.txt
python src/transient_velocity_profile.py
python -m pytest -q
python scripts/validate_project.py
```

## Working rule

AI assistance may propose or review changes, but acceptance is determined by executable checks and human inspection; never change the physics or reference solution merely to make the tests pass.
