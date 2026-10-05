import numpy as np
import pytest

from src.transient_velocity_profile import (
    analytical_startup_couette,
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
