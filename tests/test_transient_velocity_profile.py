import numpy as np
import pytest

import src.transient_velocity_profile as profile_module
from src.transient_velocity_profile import (
    analytical_startup_couette,
    create_transient_profile_plot,
    fourier_viscous_number,
    generate_outputs,
    solve_transient_couette,
)


@pytest.fixture
def reference_solution():
    return solve_transient_couette()


def test_result_shapes_and_20_intervals(reference_solution):
    y, times, velocity = reference_solution
    assert y.shape == (51,)
    assert times.shape == (21,)
    assert velocity.shape == (21, 51)
    assert len(times) - 1 == 20


def test_initial_interior_is_at_rest(reference_solution):
    _, _, velocity = reference_solution
    assert np.allclose(velocity[0, 1:-1], 0.0)


def test_stationary_wall_is_zero_at_all_times(reference_solution):
    _, _, velocity = reference_solution
    assert np.allclose(velocity[:, 0], 0.0)


def test_moving_wall_is_imposed_at_all_times(reference_solution):
    _, _, velocity = reference_solution
    assert np.allclose(velocity[:, -1], 1.0)


def test_velocity_remains_bounded(reference_solution):
    _, _, velocity = reference_solution
    assert np.all(velocity >= -1.0e-12)
    assert np.all(velocity <= 1.0 + 1.0e-12)


def test_interior_velocity_increases_monotonically(reference_solution):
    _, _, velocity = reference_solution
    assert np.all(np.diff(velocity[:, 1:-1], axis=0) >= -1.0e-12)


def test_profiles_increase_monotonically_towards_moving_wall(reference_solution):
    _, _, velocity = reference_solution
    assert np.all(np.diff(velocity, axis=1) >= -1.0e-12)


def test_final_profile_is_close_to_analytical(reference_solution):
    y, times, velocity = reference_solution
    analytical = analytical_startup_couette(y, times[-1], 0.01, 1.0, 1.0e-6)
    assert np.max(np.abs(velocity[-1] - analytical)) < 2.0e-3


def test_selected_times_match_analytical_solution(reference_solution):
    y, times, velocity = reference_solution
    tolerances = {25.0: 3.5e-2, 50.0: 1.0e-2, 100.0: 2.0e-3}
    for time, tolerance in tolerances.items():
        index = int(np.where(np.isclose(times, time))[0][0])
        analytical = analytical_startup_couette(y, time, 0.01, 1.0, 1.0e-6)
        assert np.max(np.abs(velocity[index] - analytical)) < tolerance


def test_final_profile_approaches_steady_line(reference_solution):
    y, _, velocity = reference_solution
    steady = y / 0.01
    assert np.max(np.abs(velocity[-1] - steady)) < 2.0e-3


def test_discrete_pde_residual_is_small(reference_solution):
    y, times, velocity = reference_solution
    dy = y[1] - y[0]
    dt = times[1] - times[0]
    nu = 1.0e-6
    temporal = (velocity[1:, 1:-1] - velocity[:-1, 1:-1]) / dt
    laplacian = (
        velocity[1:, :-2]
        - 2.0 * velocity[1:, 1:-1]
        + velocity[1:, 2:]
    ) / dy**2
    residual = temporal - nu * laplacian
    assert np.max(np.abs(residual)) < 1.0e-10


def test_smaller_time_step_reduces_final_analytical_error():
    y20, t20, u20 = solve_transient_couette(n_time_intervals=20)
    y40, t40, u40 = solve_transient_couette(n_time_intervals=40)
    exact20 = analytical_startup_couette(y20, t20[-1], 0.01, 1.0, 1.0e-6)
    exact40 = analytical_startup_couette(y40, t40[-1], 0.01, 1.0, 1.0e-6)
    error20 = np.max(np.abs(u20[-1] - exact20))
    error40 = np.max(np.abs(u40[-1] - exact40))
    assert error40 < error20


@pytest.mark.parametrize(
    ("time", "expected"),
    [(0.0, 0.0), (5.0, 0.05), (25.0, 0.25), (50.0, 0.50), (100.0, 1.00)],
)
def test_fourier_viscous_number_reference_values(time, expected):
    assert fourier_viscous_number(time, 0.01, 1.0e-6) == pytest.approx(expected)


def test_fourier_viscous_number_accepts_array_input():
    times = np.array([0.0, 5.0, 25.0, 50.0, 100.0])
    result = fourier_viscous_number(times, 0.01, 1.0e-6)
    assert np.allclose(result, [0.0, 0.05, 0.25, 0.50, 1.00])


@pytest.mark.parametrize(
    ("time", "height", "viscosity"),
    [(-1.0, 0.01, 1.0e-6), (0.0, 0.0, 1.0e-6), (0.0, 0.01, 0.0)],
)
def test_fourier_viscous_number_rejects_invalid_inputs(time, height, viscosity):
    with pytest.raises(ValueError):
        fourier_viscous_number(time, height, viscosity)


def test_generated_csv_contains_fourier_number_for_every_profile(tmp_path):
    generate_outputs(tmp_path)
    data = np.genfromtxt(
        tmp_path / "transient_velocity_profiles.csv",
        delimiter=",",
        names=True,
    )
    assert "fourier_viscous_number" in data.dtype.names
    fourier_numbers = np.unique(data["fourier_viscous_number"])
    assert len(fourier_numbers) == 21
    assert np.allclose(fourier_numbers, np.arange(21) * 0.05)
    assert len(data) == 21 * 51


def test_plot_contains_all_profiles_and_fourier_labels(monkeypatch):
    y, times, velocity = solve_transient_couette()
    analytical_times = []
    original_analytical = profile_module.analytical_startup_couette

    def recording_analytical(y_values, time, height, wall_velocity, viscosity):
        analytical_times.append(time)
        return original_analytical(y_values, time, height, wall_velocity, viscosity)

    monkeypatch.setattr(profile_module, "analytical_startup_couette", recording_analytical)
    figure, axes = create_transient_profile_plot(y, times, velocity, 0.01, 1.0, 1.0e-6)
    try:
        numerical_lines = [line for line in axes.lines if line.get_label().startswith("t = ")]
        analytical_lines = [line for line in axes.lines if line.get_label() == "analyt sol"]
        assert len(numerical_lines) == 21
        assert len(analytical_lines) == 1
        analytical_line = analytical_lines[0]
        assert analytical_line.get_color() == "black"
        assert analytical_line.get_linestyle() == "--"
        assert analytical_line.get_linewidth() > numerical_lines[0].get_linewidth()
        assert analytical_times == [5.0]
        expected = original_analytical(y, 5.0, 0.01, 1.0, 1.0e-6)
        assert np.allclose(analytical_line.get_xdata(), expected)
        assert np.allclose(analytical_line.get_ydata(), y)

        labels = [text.get_text() for text in axes.texts]
        label_y_positions = [text.xy[1] for text in axes.texts]
        expected_labels = [
            rf"$\mathrm{{Fo}}_{{\nu}} = {index * 0.05:.2f}$" for index in range(21)
        ]
        assert labels == expected_labels
        assert len(set(label_y_positions)) == 21
    finally:
        profile_module.plt.close(figure)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"height": 0.0},
        {"wall_velocity": -1.0},
        {"kinematic_viscosity": 0.0},
        {"final_time": 0.0},
        {"n_space": 2},
        {"n_time_intervals": 0},
    ],
)
def test_invalid_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        solve_transient_couette(**kwargs)
