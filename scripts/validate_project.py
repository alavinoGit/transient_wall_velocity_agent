"""Integrated deterministic validation for transient start-up Couette flow."""

from pathlib import Path
import subprocess
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transient_velocity_profile import (  # noqa: E402
    analytical_startup_couette,
    generate_outputs,
    solve_transient_couette,
)


def report(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    return passed


def main() -> int:
    height = 0.01
    wall_velocity = 1.0
    nu = 1.0e-6
    final_time = 100.0
    n_time_intervals = 20
    checks = []

    y, times, velocity = solve_transient_couette(
        height=height,
        wall_velocity=wall_velocity,
        kinematic_viscosity=nu,
        final_time=final_time,
        n_space=51,
        n_time_intervals=n_time_intervals,
    )
    checks.append(report("Exactly 20 time intervals", len(times) - 1 == 20))
    checks.append(report("Initial interior at rest", np.allclose(velocity[0, 1:-1], 0.0)))
    checks.append(report("Stationary-wall condition at every time", np.allclose(velocity[:, 0], 0.0)))
    checks.append(report("Moving-wall condition at every time", np.allclose(velocity[:, -1], wall_velocity)))
    checks.append(report("Velocity bounded between wall speeds", bool(np.all((velocity >= -1e-12) & (velocity <= wall_velocity + 1e-12)))))
    checks.append(report("Interior velocity monotonic in time", bool(np.all(np.diff(velocity[:, 1:-1], axis=0) >= -1e-12))))
    checks.append(report("Every profile monotonic in space", bool(np.all(np.diff(velocity, axis=1) >= -1e-12))))

    analytical_errors = []
    analytical_tolerances = {25.0: 3.5e-2, 50.0: 1.0e-2, 100.0: 2.0e-3}
    for time, tolerance in analytical_tolerances.items():
        index = int(np.where(np.isclose(times, time))[0][0])
        exact = analytical_startup_couette(y, time, height, wall_velocity, nu)
        error = float(np.max(np.abs(velocity[index] - exact)))
        analytical_errors.append(error)
        checks.append(report(f"Analytical agreement at t={time:.0f} s", error < tolerance, f"L_inf error={error:.6f} m/s; tolerance={tolerance:.3f}"))

    dy = y[1] - y[0]
    dt = times[1] - times[0]
    residual = (
        (velocity[1:, 1:-1] - velocity[:-1, 1:-1]) / dt
        - nu
        * (velocity[1:, :-2] - 2.0 * velocity[1:, 1:-1] + velocity[1:, 2:])
        / dy**2
    )
    max_residual = float(np.max(np.abs(residual)))
    checks.append(report("Discrete transient momentum balance", max_residual < 1.0e-10, f"max residual={max_residual:.3e}"))

    steady = wall_velocity * y / height
    steady_error = float(np.max(np.abs(velocity[-1] - steady)))
    checks.append(report("Final approach to steady Couette line", steady_error < 2.0e-3, f"L_inf distance={steady_error:.6f} m/s"))

    y40, t40, u40 = solve_transient_couette(n_time_intervals=40)
    exact40 = analytical_startup_couette(y40, t40[-1], height, wall_velocity, nu)
    refined_error = float(np.max(np.abs(u40[-1] - exact40)))
    checks.append(report("Temporal refinement reduces final error", refined_error < analytical_errors[-1], f"20-step={analytical_errors[-1]:.6f}, 40-step={refined_error:.6f}"))

    test_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    checks.append(report("Unit-test suite", test_run.returncode == 0, test_run.stdout.strip()))

    output_directory = PROJECT_ROOT / "results"
    generate_outputs(output_directory)
    csv_data = np.genfromtxt(
        output_directory / "transient_velocity_profiles.csv",
        delimiter=",",
        names=True,
    )
    fourier_values = np.unique(csv_data["fourier_viscous_number"])
    expected_fourier_values = np.arange(n_time_intervals + 1) * 0.05
    checks.append(
        report(
            "CSV Fourier number for all stored profiles",
            len(fourier_values) == 21
            and np.allclose(fourier_values, expected_fourier_values)
            and len(csv_data) == 21 * len(y),
            f"{len(fourier_values)} profiles, Fo_nu={fourier_values[0]:.2f}..{fourier_values[-1]:.2f}",
        )
    )
    output_pass = (
        (output_directory / "transient_velocity_profiles.csv").is_file()
        and (output_directory / "transient_velocity_profiles.png").is_file()
    )
    checks.append(report("CSV and PNG output generation", output_pass))

    final_pass = all(checks)
    print()
    print(f"VALIDATION RESULT: {'PASS' if final_pass else 'FAIL'}")
    return 0 if final_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
