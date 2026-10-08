# `instructor_2_ML.md` Guide

`instructor_2_ML.md` is the task brief for turning the existing transient Couette-flow output into a structured dataset for ML work. It describes both the intended data design and a ready-to-use prompt for the GitHub Codespaces assistant.

## Document structure

- **Purpose and output design** describes the proposed `transient_velocity_ml_dataset.json`: run-level metadata, one summary per time profile, and flat point samples for each `(time, y)` pair. It also identifies the current CSV input and solver source.
- **Interpretation and additional ML context** defines `Fo_nu` and numerical diffusion number `r`, explains the 5% analytical-error screen, and covers gradient diagnostics, provenance, and feature/target leakage.
- **Codespaces implementation prompt** contains the operational instructions to implement the workflow and validate it independently. It defines the read-only boundary, implementation checks, and final report requirements.

## Important distinctions

The 5% criterion is based on the maximum absolute discrepancy from the analytical solution, normalized by the wall velocity; it is not a statistical uncertainty estimate. Gradient extent is descriptive only. A single mesh and time step cannot prove mesh-refinement sensitivity or justify a coarser grid.

The embedded prompt requires all writes to remain inside `to_ML/`. The raw CSV, solver, tests, and project documentation are read-only inputs. It asks the Codespaces assistant to use the role instructions `implementer.md` and `validator.md` when supplied, with the current repository's `agent_specs/python_implementer.md` and `agent_specs/physics_validator.md` as equivalents if the named files are absent.

## Suggested use

1. Read `instructor_2_ML.md` for the full output contract and scientific caveats.
2. In GitHub Codespaces, give the assistant the prompt under **Codespaces implementation prompt**.
3. Review the generated artifacts and the independent validator's report, especially the error threshold result, `r`, and the stated limits on mesh conclusions.
