#!/usr/bin/env python3
"""Validate and package only the sustained accepted transient profile window."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "postprocess_config.yaml"
DEFAULT_INPUT = REPOSITORY_ROOT / "results" / "transient_velocity_profiles.csv"
REQUIRED_COLUMNS = (
    "time_s",
    "fourier_viscous_number",
    "y_m",
    "numerical_velocity_m_per_s",
    "analytical_velocity_m_per_s",
)
OPTIONAL_COLUMNS = ("error_m_per_s",)
POLICY_DEFAULTS = {
    "maximum_acceptable_normalized_profile_error": 0.05,
    "sustained_acceptance_required": True,
    "numerical_epsilon": 1.0e-12,
    "parquet_output": True,
    "csv_mirror": True,
    "gradient_threshold": 0.10,
    "curvature_threshold": 0.10,
    "wall_exclusion_fraction": 0.05,
    "export_only_sustained_accepted_window": True,
}
OUTPUT_NAMES = (
    "structured_data_2ML.parquet",
    "structured_data_2ML.csv",
    "structured_data_2ML_profile_summary.csv",
    "structured_data_2ML_case_summary.json",
    "structured_data_2ML_metadata.json",
    "structured_data_2ML_schema.json",
    "structured_data_2ML_manifest.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_within(root: Path, path: Path) -> Path:
    root = Path(root).resolve()
    resolved = Path(path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Refusing to write outside {root}: {resolved}")
    return resolved


def repository_relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def load_configuration(config_path: Path) -> dict:
    config_path = Path(config_path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Postprocessing configuration must be a YAML mapping.")
    policy = config.setdefault("policy", {})
    for name, default in POLICY_DEFAULTS.items():
        policy.setdefault(name, default)
    required = ("case_id", "input_csv", "simulation_source", "policy", "case_metadata", "provenance")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Configuration is missing keys: {missing}")
    for key, value in POLICY_DEFAULTS.items():
        if config["policy"][key] != value:
            raise ValueError(f"Configured policy {key} must remain {value!r} for this workflow.")
    return config


def read_simulation_metadata(source_path: Path) -> dict:
    source_path = Path(source_path).resolve()
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    output_function = functions.get("generate_outputs")
    solver_function = functions.get("solve_transient_couette")
    if output_function is None or solver_function is None:
        raise ValueError("Simulation source must define generate_outputs and solve_transient_couette.")

    values = {}
    value_lines = {}
    for node in output_function.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
                value_lines[target.id] = node.lineno

    required_names = {
        "height": "channel_gap_m",
        "wall_velocity": "upper_wall_velocity_m_per_s",
        "kinematic_viscosity": "kinematic_viscosity_m2_per_s",
        "final_time": "final_time_s",
        "n_space": "spatial_points",
        "n_time_intervals": "time_intervals",
    }
    missing = [name for name in required_names if name not in values]
    if missing:
        raise ValueError(f"Simulation reference configuration is unresolved: {missing}")

    lower_wall_velocity = None
    lower_wall_line = None
    solver_lines = source_text.splitlines()
    for node in ast.walk(solver_function):
        if not isinstance(node, ast.Assign):
            continue
        if any(ast.unparse(target) == "velocity[:, 0]" for target in node.targets):
            try:
                lower_wall_velocity = ast.literal_eval(node.value)
                lower_wall_line = node.lineno
            except (ValueError, TypeError):
                pass
    if lower_wall_velocity is None:
        raise ValueError("Lower-wall velocity could not be established from the simulation source.")

    solver_doc = ast.get_docstring(solver_function) or ""
    if "backward euler" not in solver_doc.lower() or "centred" not in solver_doc.lower():
        raise ValueError("Spatial/temporal schemes could not be established from the solver docstring.")
    scheme_start = next(
        node.lineno for node in solver_function.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    )
    source_provenance = {
        output_name: f"{repository_relative(source_path)}:generate_outputs line {value_lines[name]}"
        for name, output_name in required_names.items()
    }
    source_provenance["lower_wall_velocity_m_per_s"] = (
        f"{repository_relative(source_path)}:solve_transient_couette line {lower_wall_line}"
    )
    source_provenance["temporal_scheme"] = (
        f"{repository_relative(source_path)}:solve_transient_couette docstring line {scheme_start}"
    )
    source_provenance["spatial_scheme"] = source_provenance["temporal_scheme"]
    return {
        "channel_gap_m": float(values["height"]),
        "upper_wall_velocity_m_per_s": float(values["wall_velocity"]),
        "lower_wall_velocity_m_per_s": float(lower_wall_velocity),
        "kinematic_viscosity_m2_per_s": float(values["kinematic_viscosity"]),
        "final_time_s": float(values["final_time"]),
        "spatial_points": int(values["n_space"]),
        "time_intervals": int(values["n_time_intervals"]),
        "temporal_scheme": "Backward Euler",
        "spatial_scheme": "Centered second-order finite differences",
        "source_provenance": source_provenance,
    }


def read_raw_csv(input_path: Path) -> tuple[list[dict], list[str]]:
    input_path = Path(input_path).resolve()
    with input_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Raw CSV is missing a header.")
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Raw CSV is missing required columns: {missing}")
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            row = {}
            for field in reader.fieldnames:
                value = raw.get(field)
                if value is None or value.strip() == "":
                    row[field] = None
                    continue
                try:
                    number = float(value)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Non-numeric {field} on CSV line {line_number}.") from error
                if field in REQUIRED_COLUMNS and not math.isfinite(number):
                    raise ValueError(f"Non-finite {field} on CSV line {line_number}.")
                row[field] = number
            rows.append(row)
    if not rows:
        raise ValueError("Raw CSV contains no data rows.")
    return rows, list(reader.fieldnames)


def validate_grid(rows: list[dict], config: dict, simulation: dict) -> tuple[list[float], list[float], dict]:
    policy = config["policy"]
    case = config["case_metadata"]
    for field in (
        "channel_gap_m",
        "kinematic_viscosity_m2_per_s",
        "lower_wall_velocity_m_per_s",
        "upper_wall_velocity_m_per_s",
        "spatial_points",
        "time_intervals",
        "temporal_scheme",
        "spatial_scheme",
    ):
        configured = case.get(field)
        discovered = simulation.get(field)
        if configured is None or discovered is None or configured != discovered:
            raise ValueError(f"Configured {field} does not match simulation source: {configured!r} vs {discovered!r}.")

    times = sorted({float(row["time_s"]) for row in rows})
    positions = sorted({float(row["y_m"]) for row in rows})
    if len(times) != simulation["time_intervals"] + 1:
        raise ValueError("Raw CSV time count does not match simulation time intervals.")
    if len(positions) != simulation["spatial_points"]:
        raise ValueError("Raw CSV spatial-node count does not match simulation source.")
    if any(b <= a for a, b in zip(times, times[1:])) or any(
        b <= a for a, b in zip(positions, positions[1:])
    ):
        raise ValueError("Time and spatial coordinates must be strictly increasing.")

    gap = simulation["channel_gap_m"]
    nu = simulation["kinematic_viscosity_m2_per_s"]
    if gap <= 0 or nu <= 0:
        raise ValueError("Gap and kinematic viscosity must be positive.")
    expected_y = np.linspace(0.0, gap, len(positions))
    if not np.allclose(positions, expected_y, rtol=1e-9, atol=1e-12):
        raise ValueError("CSV coordinates are not the solver's uniform channel grid.")
    if not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-9, atol=1e-12):
        raise ValueError("CSV time grid is not uniform.")
    if not np.allclose(np.diff(positions), np.diff(positions)[0], rtol=1e-9, atol=1e-12):
        raise ValueError("CSV spatial grid is not uniform.")

    pairs = set()
    grouped = defaultdict(list)
    for row in rows:
        time = float(row["time_s"])
        fourier = float(row["fourier_viscous_number"])
        y = float(row["y_m"])
        pair = (time, fourier, y)
        if pair in pairs:
            raise ValueError(f"Duplicate profile-coordinate row: {pair}")
        pairs.add(pair)
        expected_fourier = nu * time / gap**2
        if not math.isclose(fourier, expected_fourier, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"CSV Fo_nu mismatch at t={time}: {fourier} vs {expected_fourier}.")
        grouped[time].append(row)

    expected_rows = len(times) * len(positions)
    if len(rows) != expected_rows:
        raise ValueError(f"Incomplete profiles: expected {expected_rows} rows, found {len(rows)}.")
    for time, profile in grouped.items():
        profile_y = sorted(float(row["y_m"]) for row in profile)
        if not np.allclose(profile_y, positions, rtol=0.0, atol=1e-12):
            raise ValueError(f"Profile at t={time} does not share the full spatial grid.")
        profile_fourier = [float(row["fourier_viscous_number"]) for row in profile]
        if not np.allclose(profile_fourier, profile_fourier[0], rtol=0.0, atol=1e-12):
            raise ValueError(f"Profile at t={time} has inconsistent Fourier numbers.")

    derived = {
        "delta_t_s": float(times[1] - times[0]),
        "delta_y_m": float(positions[1] - positions[0]),
        "numerical_diffusion_number_r": None,
    }
    derived["numerical_diffusion_number_r"] = nu * derived["delta_t_s"] / derived["delta_y_m"] ** 2
    configured_dt = case.get("timestep_s")
    configured_dy = case.get("spatial_step_m")
    configured_r = case.get("numerical_diffusion_number_r")
    if configured_dt is None or not math.isclose(configured_dt, derived["delta_t_s"], rel_tol=1e-9):
        raise ValueError("Configured timestep does not match CSV coordinates.")
    if configured_dy is None or not math.isclose(configured_dy, derived["delta_y_m"], rel_tol=1e-9):
        raise ValueError("Configured spatial step does not match CSV coordinates.")
    if configured_r is None or not math.isclose(configured_r, derived["numerical_diffusion_number_r"], rel_tol=1e-9):
        raise ValueError("Configured r does not match nu*dt/dy**2.")
    return times, positions, {**derived, "grouped": grouped}


def find_acceptance_windows(times: list[float], accepted: list[bool]) -> dict:
    """Find first individual pass and the earliest suffix of sustained passes."""
    first_isolated_index = next((i for i, value in enumerate(accepted) if value), None)
    sustained_index = next(
        (i for i in range(len(accepted)) if all(accepted[i:])), None
    )
    return {
        "first_isolated_index": first_isolated_index,
        "first_sustained_index": sustained_index,
        "sustained_end_index": len(times) - 1 if sustained_index is not None else None,
    }


def _largest_contiguous_region(mask: np.ndarray, positions: np.ndarray) -> dict | None:
    best_start = best_end = None
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(mask) and mask[index + 1]:
            index += 1
        end = index
        if best_start is None or end - start > best_end - best_start:
            best_start, best_end = start, end
        index += 1
    if best_start is None:
        return None
    return {
        "start_y_m": float(positions[best_start]),
        "end_y_m": float(positions[best_end]),
        "start_y_over_gap": float(positions[best_start] / positions[-1]),
        "end_y_over_gap": float(positions[best_end] / positions[-1]),
        "node_count": int(best_end - best_start + 1),
        "span_m": float(positions[best_end] - positions[best_start]),
    }


def _relative_l2(error: np.ndarray, reference: np.ndarray, y: np.ndarray, epsilon: float) -> float:
    numerator = math.sqrt(float(np.trapezoid(error**2, y)))
    denominator = max(math.sqrt(float(np.trapezoid(reference**2, y))), epsilon)
    return numerator / denominator


def analyze_profiles(rows: list[dict], config: dict, simulation: dict, times: list[float], positions: list[float], grid_data: dict) -> tuple[list[dict], list[dict], dict]:
    policy = config["policy"]
    grouped = grid_data["grouped"]
    gap = simulation["channel_gap_m"]
    nu = simulation["kinematic_viscosity_m2_per_s"]
    lower = simulation["lower_wall_velocity_m_per_s"]
    upper = simulation["upper_wall_velocity_m_per_s"]
    configured_reference = config.get("reference_velocity_m_per_s")
    if upper != 0.0:
        reference_velocity = abs(upper)
    else:
        reference_velocity = max(abs(lower), abs(upper))
    if reference_velocity is None or not math.isfinite(reference_velocity) or reference_velocity <= 0.0:
        raise ValueError("Cannot compute percentage acceptance without a non-zero declared wall reference velocity.")
    if configured_reference is not None and not math.isclose(
        abs(float(configured_reference)), reference_velocity, rel_tol=1e-12, abs_tol=1e-15
    ):
        raise ValueError("Configured reference velocity disagrees with the discovered wall speed.")

    max_error_limit = float(policy["maximum_acceptable_normalized_profile_error"])
    epsilon = float(policy["numerical_epsilon"])
    gradient_limit = float(policy["gradient_threshold"])
    curvature_limit = float(policy["curvature_threshold"])
    wall_exclusion = float(policy["wall_exclusion_fraction"])
    if not 0.0 < max_error_limit < 1.0 or epsilon <= 0.0:
        raise ValueError("Error threshold or numerical epsilon is outside its valid range.")
    if not 0.0 <= wall_exclusion < 0.5 or gradient_limit < 0 or curvature_limit < 0:
        raise ValueError("Mesh heuristic policy values are invalid.")

    y = np.asarray(positions, dtype=float)
    y_over_gap = y / gap
    accepted_profiles = []
    all_profile_results = []
    accepted_flags = []
    for profile_index, time in enumerate(times):
        fourier = nu * float(time) / gap**2
        profile_rows = sorted(grouped[float(time)], key=lambda row: float(row["y_m"]))
        numerical = np.asarray([row["numerical_velocity_m_per_s"] for row in profile_rows], dtype=float)
        analytical = np.asarray([row["analytical_velocity_m_per_s"] for row in profile_rows], dtype=float)
        if not np.all(np.isfinite(numerical)) or not np.all(np.isfinite(analytical)):
            raise ValueError(f"Non-finite velocity in profile at t={time}.")
        supplied_errors = [row.get("error_m_per_s") for row in profile_rows]
        signed_error = numerical - analytical
        if any(value is not None for value in supplied_errors):
            if any(value is None for value in supplied_errors) or not np.allclose(
                np.asarray(supplied_errors, dtype=float), signed_error, rtol=1e-8, atol=1e-12
            ):
                raise ValueError(f"Supplied error_m_per_s has inconsistent sign/value at t={time}.")

        normalized_max = float(np.max(np.abs(signed_error)) / max(reference_velocity, epsilon))
        accepted = normalized_max < max_error_limit
        accepted_flags.append(accepted)
        first_derivative = np.gradient(numerical, y, edge_order=2)
        second_derivative = np.gradient(first_derivative, y, edge_order=2)
        normalized_gradient = np.abs(first_derivative) / max(reference_velocity / gap, epsilon)
        normalized_curvature = np.abs(second_derivative) / max(reference_velocity / gap**2, epsilon)
        distance_to_wall = np.minimum(y, gap - y)
        distance_fraction = distance_to_wall / gap
        near_wall = distance_fraction <= wall_exclusion
        fine_candidate = near_wall | (normalized_gradient >= gradient_limit) | (normalized_curvature >= curvature_limit)
        coarse_candidate = ~fine_candidate
        coarse_region = _largest_contiguous_region(coarse_candidate, y)
        absolute_error = np.abs(signed_error)
        error_l2 = _relative_l2(signed_error, analytical, y, epsilon)
        max_index = int(np.argmax(absolute_error))
        profile_id = f"{config['case_id']}_profile_{profile_index:03d}"
        profile_summary = {
            "case_id": config["case_id"],
            "trajectory_id": config["case_id"],
            "profile_id": profile_id,
            "time_s": float(time),
            "fourier_viscous_number": float(fourier),
            "numerical_diffusion_number_r": float(grid_data["numerical_diffusion_number_r"]),
            "profile_max_absolute_error_m_per_s": float(np.max(absolute_error)),
            "profile_max_normalized_error": normalized_max,
            "profile_mae_m_per_s": float(np.mean(absolute_error)),
            "profile_rmse_m_per_s": float(np.sqrt(np.mean(signed_error**2))),
            "profile_relative_l2_error": float(error_l2),
            "maximum_error_location_y_m": float(y[max_index]),
            "maximum_error_location_y_over_gap": float(y_over_gap[max_index]),
            "maximum_absolute_normalized_gradient": float(np.max(normalized_gradient)),
            "maximum_absolute_normalized_curvature": float(np.max(normalized_curvature)),
            "fine_grid_heuristic_fraction": float(np.mean(fine_candidate)),
            "largest_contiguous_coarse_candidate_region": coarse_region,
            "profile_accepted": accepted,
            "accepted_window_member": False,
        }
        all_profile_results.append(profile_summary)
        point_rows = []
        local_delta_y = np.empty_like(y)
        local_delta_y[0] = y[1] - y[0]
        local_delta_y[-1] = y[-1] - y[-2]
        if len(y) > 2:
            local_delta_y[1:-1] = 0.5 * (y[2:] - y[:-2])
        for i, source_row in enumerate(profile_rows):
            point = {
                "case_id": config["case_id"],
                "trajectory_id": config["case_id"],
                "profile_id": profile_id,
                "time_s": float(time),
                "fourier_viscous_number": float(fourier),
                "numerical_diffusion_number_r": float(grid_data["numerical_diffusion_number_r"]),
                "y_m": float(y[i]),
                "y_over_gap": float(y_over_gap[i]),
                "distance_to_nearest_wall_m": float(distance_to_wall[i]),
                "distance_to_nearest_wall_over_gap": float(distance_fraction[i]),
                "numerical_velocity_m_per_s": float(numerical[i]),
                "analytical_velocity_m_per_s": float(analytical[i]),
                "source_error_m_per_s": source_row.get("error_m_per_s"),
                "recomputed_signed_error_m_per_s": float(signed_error[i]),
                "recomputed_absolute_error_m_per_s": float(absolute_error[i]),
                "normalized_absolute_error": float(absolute_error[i] / max(reference_velocity, epsilon)),
                "squared_error_m2_per_s2": float(signed_error[i] ** 2),
                "local_delta_y_m": float(local_delta_y[i]),
                "du_dy_m_per_s_per_m": float(first_derivative[i]),
                "d2u_dy2_m_per_s_per_m2": float(second_derivative[i]),
                "normalized_gradient": float(normalized_gradient[i]),
                "normalized_curvature": float(normalized_curvature[i]),
                "fine_grid_heuristic": bool(fine_candidate[i]),
                "coarse_grid_candidate_heuristic": bool(coarse_candidate[i]),
                "within_wall_exclusion_zone": bool(near_wall[i]),
                "profile_accepted": accepted,
                "accepted_window_member": False,
            }
            point_rows.append(point)
        all_profile_results[-1]["_point_rows"] = point_rows

    window = find_acceptance_windows(times, accepted_flags)
    sustained_start = window["first_sustained_index"]
    if policy["sustained_acceptance_required"] and sustained_start is None:
        export_indices = []
    elif sustained_start is not None:
        export_indices = list(range(sustained_start, len(times)))
    else:
        export_indices = [i for i, accepted in enumerate(accepted_flags) if accepted]

    exported_summaries = []
    exported_rows = []
    for index in export_indices:
        summary = dict(all_profile_results[index])
        summary.pop("_point_rows")
        summary["accepted_window_member"] = True
        exported_summaries.append(summary)
        for point in all_profile_results[index]["_point_rows"]:
            point["accepted_window_member"] = True
            exported_rows.append(point)

    first_isolated = (
        all_profile_results[window["first_isolated_index"]]
        if window["first_isolated_index"] is not None
        else None
    )
    first_sustained = (
        all_profile_results[sustained_start] if sustained_start is not None else None
    )
    case_summary = {
        "case_id": config["case_id"],
        "training_eligible_under_threshold": bool(exported_rows),
        "generalization_ready": False,
        "training_eligibility_warning": "A single physical parameter, mesh, and timestep configuration is not a generalization dataset.",
        "acceptance_metric": "max_y(abs(u_numerical-u_analytical))/max(abs(reference_velocity), epsilon)",
        "maximum_acceptable_normalized_profile_error": max_error_limit,
        "comparison": "strictly less than threshold",
        "reference_velocity_m_per_s": float(reference_velocity),
        "sustained_acceptance_required": bool(policy["sustained_acceptance_required"]),
        "first_isolated_accepted_profile": _summary_pointer(first_isolated),
        "first_sustained_accepted_profile": _summary_pointer(first_sustained),
        "accepted_window_start_time_s": float(times[sustained_start]) if sustained_start is not None else None,
        "accepted_window_start_fourier_viscous_number": float(nu * times[sustained_start] / gap**2) if sustained_start is not None else None,
        "accepted_window_end_time_s": float(times[-1]) if sustained_start is not None else None,
        "accepted_window_end_fourier_viscous_number": float(nu * times[-1] / gap**2) if sustained_start is not None else None,
        "exported_profile_count": len(exported_summaries),
        "exported_point_count": len(exported_rows),
        "raw_profile_count": len(times),
        "raw_point_count": len(rows),
        "excluded_profile_count": len(times) - len(exported_summaries),
        "excluded_profile_policy": "Only the first sustained contiguous accepted suffix is exported. Isolated early passes and all failing profiles are excluded from pointwise and profile-summary training tables.",
        "isolated_t0_excluded_from_training_rows": bool(first_isolated and first_isolated["time_s"] == 0.0 and (sustained_start or 0) > 0),
        "mesh_refinement_conclusion": "Not established: only one spatial grid and one timestep are present.",
    }
    case_summary["first_isolated_accepted_profile"] = _summary_pointer(first_isolated)
    case_summary["first_sustained_accepted_profile"] = _summary_pointer(first_sustained)
    return exported_rows, exported_summaries, case_summary, all_profile_results


def _summary_pointer(summary: dict | None) -> dict | None:
    if summary is None:
        return None
    return {
        "profile_id": summary["profile_id"],
        "time_s": summary["time_s"],
        "fourier_viscous_number": summary["fourier_viscous_number"],
        "profile_max_normalized_error": summary["profile_max_normalized_error"],
    }


def schema_fields() -> list[dict]:
    specs = [
        ("case_id", "string", None, False, "Simulation case identifier.", "config", "case configuration", "feature"),
        ("trajectory_id", "string", None, False, "Grouping key for a trajectory/case.", "config", "equals case_id for this single run", "grouping_key"),
        ("profile_id", "string", None, False, "Stable identifier for one saved time profile.", "derived", "case_id plus original time-profile index", "grouping_key"),
        ("time_s", "float64", "s", True, "Physical time; only the sustained accepted window is exported.", "raw_csv", "preserved source field", "feature"),
        ("fourier_viscous_number", "float64", "dimensionless", False, "Channel-scale viscous Fourier number.", "raw_csv+simulation_source", "validated nu*time/H^2", "feature"),
        ("numerical_diffusion_number_r", "float64", "dimensionless", False, "Cell-scale viscous diffusion number.", "simulation_source+raw_csv", "nu*dt/dy^2", "feature"),
        ("y_m", "float64", "m", True, "Wall-normal coordinate.", "raw_csv", "preserved and sorted within profile", "feature"),
        ("y_over_gap", "float64", "dimensionless", False, "Wall-normal coordinate normalized by channel gap.", "raw_csv+simulation_source", "y_m/H", "feature"),
        ("distance_to_nearest_wall_m", "float64", "m", True, "Distance from the nearest wall.", "derived", "min(y,H-y)", "feature"),
        ("distance_to_nearest_wall_over_gap", "float64", "dimensionless", False, "Nearest-wall distance normalized by channel gap.", "derived", "distance_to_nearest_wall_m/H", "feature"),
        ("numerical_velocity_m_per_s", "float64", "m/s", True, "Simulated numerical velocity; quality/diagnostic field, not a physical-target feature.", "raw_csv", "preserved source field", "quality_label"),
        ("analytical_velocity_m_per_s", "float64", "m/s", True, "Analytical reference velocity.", "raw_csv", "preserved source field", "target"),
        ("source_error_m_per_s", "float64", "m/s", True, "Source CSV error field, if provided.", "raw_csv", "preserved and independently checked as numerical minus analytical", "provenance_only"),
        ("recomputed_signed_error_m_per_s", "float64", "m/s", True, "Numerical velocity minus analytical velocity.", "derived", "u_numerical-u_analytical", "quality_label"),
        ("recomputed_absolute_error_m_per_s", "float64", "m/s", True, "Absolute analytical discrepancy.", "derived", "abs(recomputed signed error)", "quality_label"),
        ("normalized_absolute_error", "float64", "dimensionless", False, "Absolute discrepancy divided by reference wall speed magnitude with epsilon guard.", "derived+config", "abs(error)/max(abs(reference_velocity),epsilon)", "quality_label"),
        ("squared_error_m2_per_s2", "float64", "m^2/s^2", True, "Squared signed analytical discrepancy.", "derived", "signed_error^2", "quality_label"),
        ("local_delta_y_m", "float64", "m", True, "Local nearest-neighbor spacing; centered mean internally and one-sided at endpoints.", "raw_csv", "neighbor coordinate differences within profile", "provenance_only"),
        ("du_dy_m_per_s_per_m", "float64", "(m/s)/m", True, "First wall-normal velocity derivative.", "raw_csv", "numpy.gradient per profile, edge_order=2", "provenance_only"),
        ("d2u_dy2_m_per_s_per_m2", "float64", "(m/s)/m^2", True, "Second wall-normal velocity derivative.", "raw_csv", "numpy.gradient of first derivative per profile, edge_order=2", "provenance_only"),
        ("normalized_gradient", "float64", "dimensionless", False, "Absolute gradient relative to reference_velocity/H.", "derived+config", "abs(du/dy)/(reference_velocity/H)", "quality_label"),
        ("normalized_curvature", "float64", "dimensionless", False, "Absolute curvature relative to reference_velocity/H^2.", "derived+config", "abs(d2u/dy2)/(reference_velocity/H^2)", "quality_label"),
        ("fine_grid_heuristic", "bool", None, False, "Heuristic flag for wall, gradient, or curvature regions needing finer resolution.", "derived+config", "wall exclusion OR normalized gradient/curvature threshold", "quality_label"),
        ("coarse_grid_candidate_heuristic", "bool", None, False, "Heuristic candidate only; not evidence a coarse grid is accurate.", "derived+config", "logical inverse of fine_grid_heuristic", "quality_label"),
        ("within_wall_exclusion_zone", "bool", None, False, "True within the configured fraction of the gap from either wall.", "derived+config", "distance_to_nearest_wall_over_gap <= wall_exclusion_fraction", "quality_label"),
        ("profile_accepted", "bool", None, False, "Profile-wide maximum normalized error is strictly below the configured threshold.", "derived+config", "max error over profile < configured limit", "quality_label"),
        ("accepted_window_member", "bool", None, False, "Point belongs to the sustained contiguous accepted export window.", "derived", "profile at/after earliest sustained acceptance", "quality_label"),
    ]
    fields = []
    for name, dtype, units, dimensional, description, source, derivation, role in specs:
        fields.append({
            "field_name": name,
            "type": dtype,
            "units": units,
            "dimensional": dimensional,
            "description": description,
            "source": source,
            "derivation": derivation,
            "null_policy": "not null for this validated reference case; source_error_m_per_s may be null if absent in a future input",
            "valid_range": None,
            "ml_role": role,
        })
    return fields


def _capture_external_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return [f"git status unavailable: {result.stderr.strip()}"]
    external = []
    for line in result.stdout.splitlines():
        path = line[3:].strip()
        if path == "to_ML" or path.startswith("to_ML/"):
            continue
        external.append(line)
    return external


def _backup_existing(paths: list[Path]) -> list[dict]:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return []
    backup_root = ensure_within(PACKAGE_ROOT, PACKAGE_ROOT / "backups")
    backup_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_backup = backup_root / timestamp
    run_backup.mkdir()
    results = []
    for path in existing:
        destination = run_backup / path.relative_to(PACKAGE_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        results.append({"original": path.relative_to(PACKAGE_ROOT).as_posix(), "backup": destination.relative_to(PACKAGE_ROOT).as_posix()})
    return results


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def build_package(input_path: Path, config_path: Path, output_root: Path) -> dict:
    output_root = ensure_within(PACKAGE_ROOT, Path(output_root))
    data_root = ensure_within(PACKAGE_ROOT, output_root / "data")
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise ValueError(f"Configuration file does not exist: {config_path}")
    input_path = Path(input_path).resolve()
    source_path = (REPOSITORY_ROOT / load_configuration(config_path)["simulation_source"]).resolve()
    config = load_configuration(config_path)
    initial_external = _capture_external_status()
    status_path = ensure_within(PACKAGE_ROOT, PACKAGE_ROOT / "validation" / "repository_status_before.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps({"external_status": initial_external}, indent=2) + "\n", encoding="utf-8")
    before_source_hash = sha256_file(input_path)
    source_metadata = read_simulation_metadata(source_path)
    rows, source_columns = read_raw_csv(input_path)
    times, positions, grid_data = validate_grid(rows, config, source_metadata)
    export_rows, profile_summaries, case_summary, all_profile_results = analyze_profiles(
        rows, config, source_metadata, times, positions, grid_data
    )
    if config["policy"]["sustained_acceptance_required"] and not export_rows:
        case_summary["training_eligible_under_threshold"] = False

    output_paths = [data_root / name for name in OUTPUT_NAMES]
    backups = _backup_existing(output_paths)
    data_root.mkdir(parents=True, exist_ok=True)
    if config["policy"]["parquet_output"]:
        arrow_types = {"string": pa.string(), "float64": pa.float64(), "bool": pa.bool_()}
        arrow_schema = pa.schema([
            pa.field(field["field_name"], arrow_types[field["type"]], nullable=True)
            for field in schema_fields()
        ])
        table = pa.Table.from_pylist(export_rows, schema=arrow_schema)
        pq.write_table(table, data_root / "structured_data_2ML.parquet", compression="zstd")
    if config["policy"]["csv_mirror"]:
        point_fields = list(export_rows[0]) if export_rows else [field["field_name"] for field in schema_fields()]
        _write_csv(data_root / "structured_data_2ML.csv", export_rows, point_fields)
    summary_fields = list(profile_summaries[0]) if profile_summaries else [
        "case_id", "trajectory_id", "profile_id", "time_s", "fourier_viscous_number",
        "numerical_diffusion_number_r", "profile_max_absolute_error_m_per_s",
        "profile_max_normalized_error", "profile_mae_m_per_s", "profile_rmse_m_per_s",
        "profile_relative_l2_error", "maximum_error_location_y_m",
        "maximum_error_location_y_over_gap", "maximum_absolute_normalized_gradient",
        "maximum_absolute_normalized_curvature", "fine_grid_heuristic_fraction",
        "largest_contiguous_coarse_candidate_region", "profile_accepted", "accepted_window_member",
    ]
    _write_csv(data_root / "structured_data_2ML_profile_summary.csv", profile_summaries, summary_fields)

    gap = source_metadata["channel_gap_m"]
    nu = source_metadata["kinematic_viscosity_m2_per_s"]
    reference = abs(source_metadata["upper_wall_velocity_m_per_s"])
    if reference == 0.0:
        reference = max(abs(source_metadata["upper_wall_velocity_m_per_s"]), abs(source_metadata["lower_wall_velocity_m_per_s"]))
    provenance = dict(config["provenance"])
    provenance.update(source_metadata["source_provenance"])
    provenance.update({
        "reference_velocity_m_per_s": "upper_wall_speed_magnitude_from_simulation_source",
        "timestep_s": "derived_from_raw_csv_time_coordinates",
        "spatial_step_m": "derived_from_raw_csv_spatial_coordinates",
        "numerical_diffusion_number_r": "derived_from_simulation_viscosity_and_csv_dt_dy",
        "fourier_viscous_number": "raw_csv_verified_against_simulation_parameters",
    })
    selected_times = [row["time_s"] for row in profile_summaries]
    selected_fo = [row["fourier_viscous_number"] for row in profile_summaries]
    metadata = {
        "schema_version": "2.0.0",
        "case_id": config["case_id"],
        "source": {
            "input_csv": repository_relative(input_path),
            "input_csv_sha256": before_source_hash,
            "source_columns": source_columns,
            "simulation_source": repository_relative(source_path),
            "simulation_source_sha256": sha256_file(source_path),
            "raw_rows": len(rows),
            "raw_profile_count": len(times),
            "raw_spatial_node_count": len(positions),
            "source_row_sha256_after_read": sha256_file(input_path),
        },
        "physical_case": source_metadata,
        "derived_grid": {
            "delta_t_s": grid_data["delta_t_s"],
            "delta_y_m": grid_data["delta_y_m"],
            "fourier_viscous_number_definition": "Fo_nu = nu*time/H^2",
            "fourier_viscous_number_range_raw": [float(nu * times[0] / gap**2), float(nu * times[-1] / gap**2)],
            "numerical_diffusion_number_r_definition": "r = nu*dt/dy^2",
            "numerical_diffusion_number_r": grid_data["numerical_diffusion_number_r"],
            "reference_velocity_m_per_s": reference,
        },
        "postprocessing_policy": config["policy"],
        "provenance": provenance,
        "accepted_training_window": {
            "start_time_s": case_summary["accepted_window_start_time_s"],
            "end_time_s": case_summary["accepted_window_end_time_s"],
            "start_fourier_viscous_number": case_summary["accepted_window_start_fourier_viscous_number"],
            "end_fourier_viscous_number": case_summary["accepted_window_end_fourier_viscous_number"],
            "criterion": case_summary["acceptance_metric"],
            "threshold": case_summary["maximum_acceptable_normalized_profile_error"],
            "strict_comparison": "<",
            "only_window_exported": True,
            "exported_profile_count": case_summary["exported_profile_count"],
            "exported_point_count": case_summary["exported_point_count"],
        },
        "ml_roles": {
            "features": ["time_s", "fourier_viscous_number", "numerical_diffusion_number_r", "y_over_gap", "distance_to_nearest_wall_over_gap"],
            "target": "analytical_velocity_m_per_s",
            "quality_fields_not_to_use_as_inputs": ["numerical_velocity_m_per_s", "source_error_m_per_s", "recomputed_signed_error_m_per_s", "recomputed_absolute_error_m_per_s", "normalized_absolute_error"],
            "sample_weight": None,
            "grouping_keys": ["case_id", "trajectory_id", "profile_id"],
            "warning": "Only one physical parameter, mesh, and timestep configuration is present; data are not suitable as proof of cross-case generalization.",
        },
        "gradient_curvature_heuristics": {
            "gradient_threshold": config["policy"]["gradient_threshold"],
            "curvature_threshold": config["policy"]["curvature_threshold"],
            "wall_exclusion_fraction": config["policy"]["wall_exclusion_fraction"],
            "fine_label_rule": "wall-excluded OR normalized_gradient >= threshold OR normalized_curvature >= threshold",
            "coarse_label_caveat": "Heuristic only; grid convergence requires independent mesh/time-step refinement runs.",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (data_root / "structured_data_2ML_case_summary.json").write_text(
        json.dumps(case_summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (data_root / "structured_data_2ML_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (data_root / "structured_data_2ML_schema.json").write_text(
        json.dumps({"schema_version": "2.0.0", "fields": schema_fields()}, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    manifest_entries = []
    current_outputs = [
        data_root / name for name in OUTPUT_NAMES if name != "structured_data_2ML_manifest.csv"
    ]
    for path in [input_path, source_path, *current_outputs]:
        if not path.is_file():
            continue
        manifest_entries.append({
            "path": repository_relative(path),
            "kind": "source_input" if path in (input_path, source_path) else "generated_output",
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    _write_csv(data_root / "structured_data_2ML_manifest.csv", manifest_entries, ["path", "kind", "sha256", "size_bytes"])

    final_source_hash = sha256_file(input_path)
    if final_source_hash != before_source_hash:
        raise RuntimeError("Raw source CSV checksum changed during processing.")
    return {
        "case_summary": case_summary,
        "metadata": metadata,
        "backups": backups,
        "all_profile_results": all_profile_results,
        "source_sha256": before_source_hash,
        "output_root": output_root,
        "selected_time_range_s": [min(selected_times), max(selected_times)] if selected_times else None,
        "selected_fourier_range": [min(selected_fo), max(selected_fo)] if selected_fo else None,
    }


def _write_backup_info(result: dict, output_root: Path) -> None:
    if result["backups"]:
        path = ensure_within(PACKAGE_ROOT, output_root / "validation" / "backups_created.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result["backups"], indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT)
    args = parser.parse_args(argv)
    try:
        output_root = ensure_within(PACKAGE_ROOT, args.output_root)
        result = build_package(args.input, args.config, output_root)
        _write_backup_info(result, output_root)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    summary = result["case_summary"]
    print(f"Selected source: {repository_relative(args.input)}")
    print(f"Source rows/profiles/nodes: {summary['raw_point_count']}/{summary['raw_profile_count']}/{result['metadata']['source']['raw_spatial_node_count']}")
    print(f"Exported sustained window: {summary['accepted_window_start_fourier_viscous_number']}..{summary['accepted_window_end_fourier_viscous_number']} Fo_nu, {summary['accepted_window_start_time_s']}..{summary['accepted_window_end_time_s']} s")
    print(f"Exported profiles/point rows: {summary['exported_profile_count']}/{summary['exported_point_count']}")
    isolated = summary["first_isolated_accepted_profile"]
    sustained = summary["first_sustained_accepted_profile"]
    print(f"First isolated accepted Fo_nu/time: {isolated['fourier_viscous_number']}/{isolated['time_s']}" if isolated else "First isolated accepted profile: none")
    print(f"First sustained accepted Fo_nu/time: {sustained['fourier_viscous_number']}/{sustained['time_s']}" if sustained else "First sustained accepted profile: none")
    print(f"r: {result['metadata']['derived_grid']['numerical_diffusion_number_r']:.12g}")
    print(f"Source CSV SHA-256: {result['source_sha256']}")
    print(f"Primary Parquet: {result['output_root'] / 'data' / 'structured_data_2ML.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
