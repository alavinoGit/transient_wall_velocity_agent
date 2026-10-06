# transient_wall_velocity_agent
Transient Couette-flow agent. Objective is to calculate the velocity profile and investigate the stability of the time numerical integration

# Transient Couette-Flow Validation Workflow

A GitHub Codespaces tutorial for solving and validating the time evolution of start-up plane Couette flow.

## What changes from the steady project?

The steady equation `d2u/dy2 = 0` is replaced by the transient diffusion equation `du/dt = nu*d2u/dy2`; the solver now returns 21 profiles spanning 20 equal time intervals and validates them against the analytical start-up solution and steady limit.

## Project structure

```text
transient-couette-agent/
├── .devcontainer/
│   └── devcontainer.json
├── agent_specs/
│   ├── physics_validator.md
│   └── python_implementer.md
├── scripts/
│   └── validate_project.py
├── src/
│   └── transient_velocity_profile.py
├── tests/
│   ├── conftest.py
│   └── test_transient_velocity_profile.py
├── results/
│   └── .gitkeep
├── .gitignore
├── PROJECT_GUIDE.md
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Run in Codespaces

```bash
python src/transient_velocity_profile.py
python -m pytest -q
python scripts/validate_project.py
```

The model creates:

```text
results/transient_velocity_profiles.csv
results/transient_velocity_profiles.png
```

The PNG contains all 21 stored profiles on one graph with a time legend; the CSV contains numerical and analytical velocities for each time-position pair.

The outputs also report the channel-scale viscous Fourier number
`Fo_nu = nu*t/H**2`, where `nu` is kinematic viscosity, `t` is physical time
and `H` is channel height. `Fo_nu << 1` indicates an early transient, and
`Fo_nu` of order 1 indicates momentum has diffused across the channel scale.
Steady-state closeness must still be quantified from the velocity-profile error,
not inferred from `Fo_nu` alone.

## Spatial resolution, temporal resolution, stability and accuracy

The transient Couette-flow model solves:

$$
\frac{\partial u}{\partial t}
=
\nu\frac{\partial^2u}{\partial y^2}
$$

Here, $u(y,t)$ is the streamwise velocity, $y$ is the distance between
the plates and $\nu$ is the kinematic viscosity.

The model is one-dimensional in space because the flow is assumed to
be fully developed and homogeneous along the streamwise and spanwise
directions.

### Numerical grid

The solution is represented on a grid in space and time:

$$
y_i=i\Delta y,
\qquad
t_n=n\Delta t
$$

At each grid point, the numerical variable is:

$$
u_i^n \approx u(y_i,t_n)
$$

The two grid resolutions serve different purposes:

- $\Delta y$ controls spatial resolution.
- $\Delta t$ controls temporal resolution.

A smaller $\Delta y$ places more nodes across the channel and improves
the representation of spatial curvature and wall gradients.

A smaller $\Delta t$ introduces more time levels and improves the
representation of rapid transient evolution.

At early times, the velocity changes most rapidly near the moving wall.
A smaller $\Delta y$ therefore helps resolve the early
momentum-penetration layer, while a smaller $\Delta t$ helps resolve
how that layer evolves with time.

### Spatial discretisation

The second spatial derivative is represented using a centred
finite-difference approximation:

$$
\left.
\frac{\partial^2u}{\partial y^2}
\right|_i
\approx
\frac{u_{i-1}-2u_i+u_{i+1}}{\Delta y^2}
$$

This approximation is second-order accurate in space:

$$
E_y=O(\Delta y^2)
$$

In the asymptotic convergence regime, halving $\Delta y$ should reduce
the spatial discretisation error by approximately a factor of four.

### Temporal discretisation

The time derivative is represented using implicit Backward Euler:

$$
\left.
\frac{\partial u}{\partial t}
\right|_i^{n+1}
\approx
\frac{u_i^{n+1}-u_i^n}{\Delta t}
$$

Backward Euler is first-order accurate in time:

$$
E_t=O(\Delta t)
$$

Halving $\Delta t$ should therefore reduce the dominant temporal error
by approximately a factor of two when the spatial error is sufficiently
small.

The combined discretisation error behaves approximately as:

$$
E \sim C_t\Delta t+C_y\Delta y^2
$$

Spatial and temporal refinement should therefore be assessed
separately.

## 1. Algebraic convergence at each timestep

After spatial and temporal discretisation, each timestep requires the
solution of:

$$
A\mathbf{u}^{n+1}=\mathbf{b}
$$

Algebraic convergence asks whether this discrete matrix equation has
been solved accurately.

This tutorial uses `numpy.linalg.solve`, which is a direct linear
solver. There is no nonlinear or iterative correction loop such as
SIMPLE, PISO, Newton iteration or a Krylov iteration. Consequently,
the tutorial does not currently produce an iterative residual history.

If required, the algebraic residual can be calculated after every
timestep as:

$$
\mathbf{R}_{\mathrm{alg}}^{n+1}
=
A\mathbf{u}^{n+1}-\mathbf{b}
$$

and summarised using:

$$
\left\|
\mathbf{R}_{\mathrm{alg}}^{n+1}
\right\|_\infty
$$

A small algebraic residual confirms that the discrete matrix equation
was solved correctly. It does not establish that $\Delta y$ and
$\Delta t$ are sufficiently small.

Effects of grid refinement:

- Smaller $\Delta y$ creates more spatial unknowns and a larger matrix.
- Smaller $\Delta t$ requires more matrix solves over the same physical time.
- Neither refinement automatically guarantees physical accuracy.

## 2. Numerical stability

The cell-scale viscous diffusion number is:

$$
r=\frac{\nu\Delta t}{\Delta y^2}
$$

It compares one timestep with the viscous diffusion time across one
spatial interval:

$$
t_{\Delta y}=\frac{\Delta y^2}{\nu}
$$

Therefore:

$$
r=\frac{\Delta t}{t_{\Delta y}}
$$

Its dependence on resolution is:

$$
\Delta t\downarrow
\quad\Rightarrow\quad
r\downarrow
$$

and:

$$
\Delta y\downarrow
\quad\Rightarrow\quad
r\uparrow
$$

Halving $\Delta y$ at fixed $\Delta t$ increases $r$ by a factor of
four.

For an explicit Forward Euler diffusion solver, stability would
require:

$$
r\leq\frac{1}{2}
$$

The present solver uses implicit Backward Euler, which is
unconditionally stable for this linear diffusion problem. A large
value of $r$ therefore does not cause the solution to diverge.

However, unconditional stability does not imply accuracy. A large
$\Delta t$, and therefore often a large $r$, can cause excessive
numerical damping and smear the early transient.

Thus:

- $r$ characterises numerical time-space coupling.
- $r$ is not an error measurement.
- A stable solution can still be inaccurate.
- Stability should not be inferred from a small analytical error.
- Accuracy should not be inferred from stability.

## 3. Numerical accuracy

Numerical accuracy is measured against the analytical start-up
Couette solution:

$$
E_{\mathrm{exact}}(t)
=
\frac{
\max_y
\left|
u_{\mathrm{num}}(y,t)
-
u_{\mathrm{exact}}(y,t)
\right|
}{U}
$$

The wall velocity $U$ is used as a fixed reference scale because the
local analytical velocity is zero or very small near the stationary
wall.

Effects of refinement:

- Smaller $\Delta y$ improves the spatial resolution of profile
  curvature and wall gradients.
- Smaller $\Delta t$ improves the temporal resolution of profile
  evolution and reduces Backward Euler damping.
- Refining only $\Delta y$ may leave temporal error dominant.
- Refining only $\Delta t$ may leave spatial error dominant.

A spatial-refinement study holds $\Delta t$ sufficiently small and
compares successively smaller values of $\Delta y$.

A temporal-refinement study holds $\Delta y$ fixed and compares
successively smaller values of $\Delta t$.

The numerical solution is considered grid-independent only when
further refinement causes negligible change in the quantities of
interest.

## 4. Physical relaxation and the Fourier viscous number

The channel-scale viscous Fourier number is:

$$
Fo_\nu=\frac{\nu t}{H^2}
$$

It compares physical elapsed time with the channel-scale viscous
diffusion time:

$$
t_H=\frac{H^2}{\nu}
$$

Therefore:

$$
Fo_\nu=\frac{t}{t_H}
$$

Interpretation:

- $Fo_\nu\ll1$: early transient.
- $Fo_\nu=O(1)$: momentum has diffused over the channel scale.
- Increasing $Fo_\nu$: the profile approaches steady Couette flow.

The physical distance from the steady profile is:

$$
E_{\mathrm{steady}}(Fo_\nu)
=
\frac{
\max_y
\left|
u_{\mathrm{num}}(y,t)-Uy/H
\right|
}{U}
$$

This is not the same as numerical error.

The analytical error:

$$
E_{\mathrm{exact}}
=
\frac{
\max_y
\left|
u_{\mathrm{num}}-u_{\mathrm{exact}}
\right|
}{U}
$$

asks whether the numerical transient is accurate.

The distance from steady state:

$$
E_{\mathrm{steady}}
=
\frac{
\max_y
\left|
u_{\mathrm{num}}-u_{\mathrm{steady}}
\right|
}{U}
$$

asks whether the physical flow has relaxed towards steady state.

The correct value of $Fo_\nu$ at a given physical time does not depend
on $\Delta y$ or $\Delta t$. Grid resolution only determines how
accurately the numerical profile represents the physical state
associated with that value of $Fo_\nu$.

## Summary

| Question | Diagnostic | Effect of smaller $\Delta y$ | Effect of smaller $\Delta t$ |
|---|---|---|---|
| Was the matrix equation solved? | Algebraic residual $A\mathbf{u}-\mathbf{b}$ | Larger matrix | More matrix solves |
| Is the method stable? | Backward Euler stability and $r=\nu\Delta t/\Delta y^2$ | Increases $r$ | Decreases $r$ |
| Is the transient accurate? | Error against the analytical solution | Improves spatial accuracy | Improves temporal accuracy |
| Has the flow relaxed? | $Fo_\nu$ and error from $Uy/H$ | Does not change physical $Fo_\nu$ | Does not change physical $Fo_\nu$ |

The key distinction is:

> Algebraic residuals assess whether the discrete equations were
> solved. Stability assesses whether numerical disturbances remain
> bounded. Analytical comparison and grid refinement assess accuracy.
> The Fourier viscous number describes physical progress through the
> transient.

## Optional AI review

In a repository-aware AI chat, ask it to read `agent_specs/physics_validator.md`, inspect the project, run the three standard commands and return the required report without editing files.

## Security

Use synthetic data only and do not commit credentials, confidential files, caches or generated results.
