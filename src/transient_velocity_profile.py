"""Implicit finite-difference solver for start-up plane Couette flow."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def validate_inputs(
    height: float,
    wall_velocity: float,
    kinematic_viscosity: float,
    final_time: float,
    n_space: int,
    n_time_intervals: int,
) -> None:
    """Reject non-physical parameters and grids that are too small."""
    if height <= 0:
        raise ValueError("Channel height must be positive.")
    if wall_velocity < 0:
        raise ValueError("Wall velocity must be non-negative.")
    if kinematic_viscosity <= 0:
        raise ValueError("Kinematic viscosity must be positive.")
    if final_time <= 0:
        raise ValueError("Final time must be positive.")
    if not isinstance(n_space, int) or n_space < 3:
        raise ValueError("n_space must be an integer of at least 3.")
    if not isinstance(n_time_intervals, int) or n_time_intervals < 1:
        raise ValueError("n_time_intervals must be a positive integer.")


def fourier_viscous_number(
    time: float | np.ndarray,
    height: float,
    kinematic_viscosity: float,
) -> float | np.ndarray:
    """Return the channel-scale viscous Fourier number, nu*time/height**2."""
    if height <= 0:
        raise ValueError("Channel height must be positive.")
    if kinematic_viscosity <= 0:
        raise ValueError("Kinematic viscosity must be positive.")

    time_array = np.asarray(time, dtype=float)
    if np.any(time_array < 0):
        raise ValueError("Time must be non-negative.")

    result = kinematic_viscosity * time_array / height**2
    if result.ndim == 0:
        return float(result)
    return result


def analytical_startup_couette(
    y: np.ndarray,
    time: float,
    height: float,
    wall_velocity: float,
    kinematic_viscosity: float,
    n_terms: int = 400,
) -> np.ndarray:
    """Return the Fourier-series solution for impulsively started Couette flow."""
    y_array = np.asarray(y, dtype=float)
    if time < 0:
        raise ValueError("Time must be non-negative.")
    if np.any(y_array < 0) or np.any(y_array > height):
        raise ValueError("Every y coordinate must lie inside the channel.")
    if n_terms < 1:
        raise ValueError("n_terms must be positive.")

    if time == 0:
        velocity = np.zeros_like(y_array)
        velocity[np.isclose(y_array, height)] = wall_velocity
        return velocity

    n = np.arange(1, n_terms + 1, dtype=float)[:, None]
    y_row = y_array[None, :]
    transient = (
        (2.0 * wall_velocity / np.pi)
        * np.sum(
            ((-1.0) ** n / n)
            * np.sin(n * np.pi * y_row / height)
            * np.exp(
                -(n**2)
                * np.pi**2
                * kinematic_viscosity
                * time
                / height**2
            ),
            axis=0,
        )
    )
    velocity = wall_velocity * y_array / height + transient
    velocity[np.isclose(y_array, 0.0)] = 0.0
    velocity[np.isclose(y_array, height)] = wall_velocity
    return velocity


def solve_transient_couette(
    height: float = 0.01,
    wall_velocity: float = 1.0,
    kinematic_viscosity: float = 1.0e-6,
    final_time: float = 100.0,
    n_space: int = 51,
    n_time_intervals: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve start-up Couette flow with backward Euler in time and centred space."""
    validate_inputs(
        height,
        wall_velocity,
        kinematic_viscosity,
        final_time,
        n_space,
        n_time_intervals,
    )

    y = np.linspace(0.0, height, n_space)
    times = np.linspace(0.0, final_time, n_time_intervals + 1)
    dy = y[1] - y[0]
    dt = times[1] - times[0]
    diffusion_number = kinematic_viscosity * dt / dy**2

    velocity = np.zeros((n_time_intervals + 1, n_space))
    # The moving wall starts impulsively at t = 0; the fluid interior is initially at rest.
    velocity[:, 0] = 0.0
    velocity[:, -1] = wall_velocity

    n_interior = n_space - 2
    matrix = np.diag(np.full(n_interior, 1.0 + 2.0 * diffusion_number))
    if n_interior > 1:
        off_diagonal = np.full(n_interior - 1, -diffusion_number)
        matrix += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)

    for step in range(n_time_intervals):
        right_hand_side = velocity[step, 1:-1].copy()
        right_hand_side[-1] += diffusion_number * wall_velocity
        velocity[step + 1, 1:-1] = np.linalg.solve(matrix, right_hand_side)

    return y, times, velocity


def create_transient_profile_plot(
    y: np.ndarray,
    times: np.ndarray,
    velocity: np.ndarray,
    height: float,
    wall_velocity: float,
    kinematic_viscosity: float,
) -> tuple[plt.Figure, plt.Axes]:
    """Build the transient plot with Fourier labels and the t=5 s reference."""
    figure, axes = plt.subplots(figsize=(10, 8))
    colour_map = plt.get_cmap("viridis")
    fourier_numbers = fourier_viscous_number(times, height, kinematic_viscosity)
    label_point_indices = np.rint(
        np.linspace(0.58 * (len(y) - 1), 0.98 * (len(y) - 1), len(times))
    ).astype(int)

    for index, time in enumerate(times):
        colour = colour_map(index / (len(times) - 1))
        axes.plot(
            velocity[index],
            y,
            color=colour,
            linewidth=1.2,
            label=f"t = {time:.0f} s",
        )
        point_index = label_point_indices[index]
        axes.annotate(
            rf"$\mathrm{{Fo}}_{{\nu}} = {fourier_numbers[index]:.2f}$",
            xy=(velocity[index, point_index], y[point_index]),
            xytext=(7, 0),
            textcoords="offset points",
            color=colour,
            fontsize=6,
            va="center",
        )

    analytical_time = 5.0
    analytical = analytical_startup_couette(
        y,
        analytical_time,
        height,
        wall_velocity,
        kinematic_viscosity,
    )
    axes.plot(
        analytical,
        y,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="analyt sol",
    )
    axes.set_xlabel("Velocity, u [m/s]")
    axes.set_ylabel("Distance from stationary wall, y [m]")
    axes.set_title("Start-up Couette flow: 20 time intervals")
    axes.set_xlim(-0.02, 1.27)
    axes.grid(True)
    axes.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, ncol=1)
    figure.tight_layout()
    return figure, axes


def generate_outputs(output_directory: Path | str = "results") -> dict[str, float]:
    """Run the 20-interval reference case and write CSV and PNG outputs."""
    height = 0.01
    wall_velocity = 1.0
    kinematic_viscosity = 1.0e-6
    final_time = 100.0
    n_space = 51
    n_time_intervals = 20

    y, times, velocity = solve_transient_couette(
        height=height,
        wall_velocity=wall_velocity,
        kinematic_viscosity=kinematic_viscosity,
        final_time=final_time,
        n_space=n_space,
        n_time_intervals=n_time_intervals,
    )
    fourier_numbers = fourier_viscous_number(
        times,
        height,
        kinematic_viscosity,
    )

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, time in enumerate(times):
        analytical = analytical_startup_couette(
            y,
            time,
            height,
            wall_velocity,
            kinematic_viscosity,
        )
        for y_value, numerical_value, analytical_value in zip(
            y, velocity[index], analytical
        ):
            rows.append(
                [
                    time,
                    fourier_numbers[index],
                    y_value,
                    numerical_value,
                    analytical_value,
                    numerical_value - analytical_value,
                ]
            )

    np.savetxt(
        output_path / "transient_velocity_profiles.csv",
        np.asarray(rows),
        delimiter=",",
        header=(
            "time_s,fourier_viscous_number,y_m,numerical_velocity_m_per_s,"
            "analytical_velocity_m_per_s,error_m_per_s"
        ),
        comments="",
    )

    figure, _ = create_transient_profile_plot(
        y,
        times,
        velocity,
        height,
        wall_velocity,
        kinematic_viscosity,
    )
    figure.savefig(output_path / "transient_velocity_profiles.png", dpi=180)
    plt.close(figure)

    steady_profile = wall_velocity * y / height
    final_analytical = analytical_startup_couette(
        y,
        final_time,
        height,
        wall_velocity,
        kinematic_viscosity,
    )
    return {
        "height": height,
        "wall_velocity": wall_velocity,
        "kinematic_viscosity": kinematic_viscosity,
        "final_time": final_time,
        "n_space": float(n_space),
        "n_time_intervals": float(n_time_intervals),
        "time_step": final_time / n_time_intervals,
        "diffusion_time": height**2 / kinematic_viscosity,
        "fourier_number_min": float(np.min(fourier_numbers)),
        "fourier_number_max": float(np.max(fourier_numbers)),
        "fourier_number_increment": float(fourier_numbers[1] - fourier_numbers[0]),
        "final_max_error_vs_analytical": float(
            np.max(np.abs(velocity[-1] - final_analytical))
        ),
        "final_max_distance_from_steady": float(
            np.max(np.abs(velocity[-1] - steady_profile))
        ),
    }


def main() -> None:
    """Run the reference transient case and print its key diagnostics."""
    result = generate_outputs()
    print(f"Channel height: {result['height']:.4f} m")
    print(f"Moving-wall velocity: {result['wall_velocity']:.3f} m/s")
    print(f"Kinematic viscosity: {result['kinematic_viscosity']:.2e} m^2/s")
    print(f"Final time: {result['final_time']:.1f} s")
    print(f"Time intervals: {int(result['n_time_intervals'])}")
    print(f"Time step: {result['time_step']:.2f} s")
    print(f"Momentum-diffusion time H^2/nu: {result['diffusion_time']:.1f} s")
    print(
        "Channel-scale viscous Fourier number: "
        f"{result['fourier_number_min']:.2f} to {result['fourier_number_max']:.2f} "
        f"(increment {result['fourier_number_increment']:.2f})"
    )
    print(
        "Final maximum error versus analytical solution: "
        f"{result['final_max_error_vs_analytical']:.6f} m/s"
    )
    print(
        "Final maximum distance from steady profile: "
        f"{result['final_max_distance_from_steady']:.6f} m/s"
    )


if __name__ == "__main__":
    main()
