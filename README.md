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

## Optional AI review

In a repository-aware AI chat, ask it to read `agent_specs/physics_validator.md`, inspect the project, run the three standard commands and return the required report without editing files.

## Security

Use synthetic data only and do not commit credentials, confidential files, caches or generated results.
