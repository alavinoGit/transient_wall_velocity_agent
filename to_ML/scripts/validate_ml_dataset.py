#!/usr/bin/env python3
"""Independently validate ML tables without modifying simulation inputs."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DATA_ROOT = PACKAGE_ROOT / "data"
VALIDATION_ROOT = PACKAGE_ROOT / "validation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _repo_status_outside_package() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git status failed: {result.stderr.strip()}")
    return [
        line for line in result.stdout.splitlines()
        if line[3:].strip() != "to_ML" and not line[3:].strip().startswith("to_ML/")
    ]


def _source_configuration(source_path: Path) -> dict:
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    generate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate_outputs")
    values = {}
    for node in generate.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    return values


def _csv_scalar(value: str, sample_value):
    if isinstance(sample_value, bool):
        return value.strip().lower() == "true"
    if isinstance(sample_value, (int, float)):
        return float(value)
    return value


def _parse_initial_checksum() -> str:
    report = (VALIDATION_ROOT / "discovery_report.md").read_text(encoding="utf-8")
    match = re.search(r"Initial source CSV SHA-256:\s*`?([0-9a-f]{64})`?", report)
    if not match:
        raise ValueError("Could not read initial raw CSV checksum from discovery report.")
    return match.group(1)


def validate(input_path: Path, config_path: Path, package_root: Path = PACKAGE_ROOT) -> dict:
    input_path = Path(input_path).resolve()
    package_root = Path(package_root).resolve()
    if package_root != PACKAGE_ROOT:
        if PACKAGE_ROOT not in package_root.parents and package_root != PACKAGE_ROOT:
            raise ValueError("Validation package root must remain inside to_ML/.")
    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    case = config["case_metadata"]
    policy = config["policy"]
    test_result_path = VALIDATION_ROOT / "test_result.json"
    if not test_result_path.is_file() or _read_json(test_result_path).get("status") != "PASS":
        raise AssertionError("Package test suite has no recorded PASS result.")
    source_path = (REPOSITORY_ROOT / config["simulation_source"]).resolve()
    simulation = _source_configuration(source_path)
    height = float(simulation["height"])
    wall_speed = float(simulation["wall_velocity"])
    viscosity = float(simulation["kinematic_viscosity"])
    final_time = float(simulation["final_time"])
    n_space = int(simulation["n_space"])
    n_intervals = int(simulation["n_time_intervals"])
    reference_velocity = abs(wall_speed) if wall_speed != 0 else max(abs(case["lower_wall_velocity_m_per_s"]), abs(wall_speed))
    epsilon = float(policy["numerical_epsilon"])
    if reference_velocity <= 0:
        raise ValueError("Reference velocity is unavailable; acceptance cannot be independently verified.")

    with input_path.open(newline="", encoding="utf-8") as stream:
        raw_reader = csv.DictReader(stream)
        raw_fields = raw_reader.fieldnames
        raw_rows = list(raw_reader)
    required = {"time_s", "fourier_viscous_number", "y_m", "numerical_velocity_m_per_s", "analytical_velocity_m_per_s"}
    if raw_fields is None or not required.issubset(raw_fields):
        raise ValueError("Raw CSV does not satisfy the required input schema.")
    times = np.array(sorted({float(row["time_s"]) for row in raw_rows}))
    positions = np.array(sorted({float(row["y_m"]) for row in raw_rows}))
    if len(times) != n_intervals + 1 or len(positions) != n_space:
        raise AssertionError("Raw CSV profile/node counts disagree with the simulation source.")
    dt = float(times[1] - times[0])
    dy = float(positions[1] - positions[0])
    r_number = viscosity * dt / dy**2
    fourier_values = viscosity * times / height**2

    grouped = {}
    raw_numerical = np.empty((len(times), len(positions)))
    raw_analytical = np.empty_like(raw_numerical)
    raw_supplied_error = np.empty_like(raw_numerical)
    for ti, time in enumerate(times):
        profile = sorted((row for row in raw_rows if float(row["time_s"]) == time), key=lambda row: float(row["y_m"]))
        if len(profile) != len(positions) or not np.allclose([float(row["y_m"]) for row in profile], positions):
            raise AssertionError(f"Raw profile at t={time} has missing or inconsistent coordinates.")
        expected_fo = viscosity * float(time) / height**2
        if not np.allclose([float(row["fourier_viscous_number"]) for row in profile], expected_fo):
            raise AssertionError(f"Raw Fo_nu does not match nu*t/H^2 at t={time}.")
        raw_numerical[ti] = [float(row["numerical_velocity_m_per_s"]) for row in profile]
        raw_analytical[ti] = [float(row["analytical_velocity_m_per_s"]) for row in profile]
        raw_supplied_error[ti] = [float(row["error_m_per_s"]) for row in profile] if "error_m_per_s" in raw_fields else raw_numerical[ti] - raw_analytical[ti]
        grouped[float(time)] = profile
    recomputed_error = raw_numerical - raw_analytical
    if not np.allclose(raw_supplied_error, recomputed_error, rtol=1e-8, atol=1e-12):
        raise AssertionError("Raw signed-error field is not numerical minus analytical.")
    profile_max_normalized = np.max(np.abs(recomputed_error), axis=1) / max(reference_velocity, epsilon)
    profile_accepted = profile_max_normalized < float(policy["maximum_acceptable_normalized_profile_error"])
    first_isolated_index = next((i for i, accepted in enumerate(profile_accepted) if accepted), None)
    first_sustained_index = next((i for i in range(len(profile_accepted)) if np.all(profile_accepted[i:])), None)
    if policy["sustained_acceptance_required"]:
        selected_indices = list(range(first_sustained_index, len(times))) if first_sustained_index is not None else []
    else:
        selected_indices = [i for i, accepted in enumerate(profile_accepted) if accepted]
    selected_times = times[selected_indices]
    accepted_pairs = {(float(times[i]), float(y)) for i in selected_indices for y in positions}

    point_csv_path = DATA_ROOT / "structured_data_2ML.csv"
    parquet_path = DATA_ROOT / "structured_data_2ML.parquet"
    summary_path = DATA_ROOT / "structured_data_2ML_profile_summary.csv"
    case_path = DATA_ROOT / "structured_data_2ML_case_summary.json"
    metadata_path = DATA_ROOT / "structured_data_2ML_metadata.json"
    schema_path = DATA_ROOT / "structured_data_2ML_schema.json"
    manifest_path = DATA_ROOT / "structured_data_2ML_manifest.csv"
    point_csv = _read_csv(point_csv_path)
    parquet_table = pq.read_table(parquet_path)
    parquet_rows = parquet_table.to_pylist()
    profile_summary_rows = _read_csv(summary_path)
    case_summary = _read_json(case_path)
    metadata = _read_json(metadata_path)
    schema = _read_json(schema_path)

    expected_samples = len(selected_indices) * len(positions)
    if len(point_csv) != expected_samples or len(parquet_rows) != expected_samples:
        raise AssertionError(f"Expected {expected_samples} accepted rows in both data formats.")
    if len(profile_summary_rows) != len(selected_indices):
        raise AssertionError("Profile-summary row count differs from accepted window count.")
    for row in point_csv:
        pair = (float(row["time_s"]), float(row["y_m"]))
        if pair not in accepted_pairs:
            raise AssertionError(f"Out-of-window row exported: {pair}.")
        if row["profile_accepted"].lower() != "true" or row["accepted_window_member"].lower() != "true":
            raise AssertionError("An exported point row is not labelled as accepted.")
    if len({(row["profile_id"], float(row["y_m"])) for row in point_csv}) != expected_samples:
        raise AssertionError("Output profile_id × y_m keys are not unique.")

    for csv_row, parquet_row in zip(point_csv, parquet_rows):
        for field, value in parquet_row.items():
            parsed = _csv_scalar(csv_row[field], value)
            if isinstance(value, float):
                if not math.isclose(parsed, value, rel_tol=1e-12, abs_tol=1e-14):
                    raise AssertionError(f"CSV/Parquet mismatch in {field}.")
            elif parsed != value:
                raise AssertionError(f"CSV/Parquet mismatch in {field}.")
    for profile_index in selected_indices:
        time = float(times[profile_index])
        summary = next(row for row in profile_summary_rows if float(row["time_s"]) == time)
        error = recomputed_error[profile_index]
        expected_max = float(np.max(np.abs(error)) / max(reference_velocity, epsilon))
        if not math.isclose(float(summary["profile_max_normalized_error"]), expected_max, rel_tol=1e-10, abs_tol=1e-12):
            raise AssertionError(f"Profile maximum error summary mismatch at t={time}.")
        if expected_max >= float(policy["maximum_acceptable_normalized_profile_error"]):
            raise AssertionError(f"Rejected profile leaked into exported summary at t={time}.")

    if not selected_indices:
        raise AssertionError("No sustained accepted window exists; output cannot train under required policy.")
    window_start = float(times[first_sustained_index])
    window_end = float(times[-1])
    if case_summary["accepted_window_start_time_s"] != window_start or case_summary["accepted_window_end_time_s"] != window_end:
        raise AssertionError("Case summary accepted-window time bounds are incorrect.")
    if case_summary["first_isolated_accepted_profile"]["time_s"] != float(times[first_isolated_index]):
        raise AssertionError("First isolated acceptance metadata is incorrect.")
    if case_summary["first_sustained_accepted_profile"]["time_s"] != window_start:
        raise AssertionError("First sustained acceptance metadata is incorrect.")
    if not metadata["accepted_training_window"]["only_window_exported"]:
        raise AssertionError("Metadata does not declare accepted-window-only export.")
    if metadata["derived_grid"]["numerical_diffusion_number_r"] is None or not math.isclose(
        metadata["derived_grid"]["numerical_diffusion_number_r"], r_number, rel_tol=1e-12
    ):
        raise AssertionError("Numerical diffusion number r is missing or incorrect.")
    if metadata["source"]["raw_rows"] != len(raw_rows):
        raise AssertionError("Metadata raw row count is incorrect.")
    schema_names = {entry["field_name"] for entry in schema["fields"]}
    if schema_names != set(point_csv[0]):
        raise AssertionError("Schema fields differ from point-table columns.")
    if "numerical_velocity_m_per_s" in metadata["ml_roles"]["features"] or "recomputed_absolute_error_m_per_s" in metadata["ml_roles"]["features"]:
        raise AssertionError("Numerical/error target leakage is present in declared ML features.")

    source_initial_hash = _parse_initial_checksum()
    source_current_hash = sha256_file(input_path)
    if source_current_hash != source_initial_hash:
        raise AssertionError("Raw source CSV checksum differs from the initial snapshot.")
    if metadata["source"]["input_csv_sha256"] != source_current_hash:
        raise AssertionError("Metadata raw input checksum does not match source.")

    # Read-only physical checks reuse the project's analytical reference and solver.
    os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_ROOT / ".mplconfig"))
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from src.transient_velocity_profile import analytical_startup_couette, solve_transient_couette

    if not np.allclose(raw_analytical[0], analytical_startup_couette(positions, 0.0, height, wall_speed, viscosity)):
        raise AssertionError("Initial analytical profile differs from the existing reference implementation.")
    for i, time in enumerate(times):
        if not np.allclose(raw_analytical[i], analytical_startup_couette(positions, float(time), height, wall_speed, viscosity), atol=2e-10):
            raise AssertionError(f"Analytical reference mismatch at t={time}.")
    if not np.allclose(raw_numerical[0, 1:-1], 0.0):
        raise AssertionError("Initial interior is not at rest.")
    if not np.allclose(raw_numerical[:, 0], case["lower_wall_velocity_m_per_s"]):
        raise AssertionError("Lower-wall condition is violated.")
    if not np.allclose(raw_numerical[:, -1], wall_speed):
        raise AssertionError("Upper-wall condition is violated.")
    bounded = bool(np.all((raw_numerical >= min(case["lower_wall_velocity_m_per_s"], wall_speed) - 1e-12) & (raw_numerical <= max(case["lower_wall_velocity_m_per_s"], wall_speed) + 1e-12)))
    time_monotone = bool(np.all(np.diff(raw_numerical[:, 1:-1], axis=0) >= -1e-12))
    space_monotone = bool(np.all(np.diff(raw_numerical, axis=1) >= -1e-12))
    if not bounded or not time_monotone or not space_monotone:
        raise AssertionError("Boundedness or physical monotonicity check failed.")
    residual = (
        (raw_numerical[1:, 1:-1] - raw_numerical[:-1, 1:-1]) / dt
        - viscosity * (raw_numerical[1:, :-2] - 2.0 * raw_numerical[1:, 1:-1] + raw_numerical[1:, 2:]) / dy**2
    )
    max_residual = float(np.max(np.abs(residual)))
    if max_residual >= 1e-10:
        raise AssertionError(f"Backward-Euler discrete residual is too large: {max_residual}.")
    steady_profile = case["lower_wall_velocity_m_per_s"] + (
        case["upper_wall_velocity_m_per_s"] - case["lower_wall_velocity_m_per_s"]
    ) * positions / height
    steady_error = float(np.max(np.abs(raw_numerical[-1] - steady_profile)))
    if steady_error >= 0.002:
        raise AssertionError("Final profile does not meet the existing steady-limit tolerance.")
    y20, t20, u20 = solve_transient_couette()
    y40, t40, u40 = solve_transient_couette(n_time_intervals=40)
    err20 = float(np.max(np.abs(u20[-1] - analytical_startup_couette(y20, float(t20[-1]), height, wall_speed, viscosity))))
    err40 = float(np.max(np.abs(u40[-1] - analytical_startup_couette(y40, float(t40[-1]), height, wall_speed, viscosity))))
    if err40 >= err20:
        raise AssertionError("Temporal refinement did not reduce final analytical error.")

    initial_status_path = VALIDATION_ROOT / "repository_status_before.json"
    initial_external = _read_json(initial_status_path)["external_status"]
    final_external = _repo_status_outside_package()
    if final_external != initial_external:
        raise AssertionError(f"Repository changes outside to_ML differ from initial snapshot: {final_external}")

    manifest = _read_csv(manifest_path)
    for entry in manifest:
        path = REPOSITORY_ROOT / entry["path"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise AssertionError(f"Manifest checksum mismatch for {entry['path']}.")
    if not any(entry["path"] == input_path.relative_to(REPOSITORY_ROOT).as_posix() for entry in manifest):
        raise AssertionError("Manifest omits the raw input CSV.")

    metrics = {
        "status": "PASS",
        "selected_input": input_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "initial_source_sha256": source_initial_hash,
        "final_source_sha256": source_current_hash,
        "source_checksum_unchanged": True,
        "package_tests_passed": True,
        "raw_rows": len(raw_rows),
        "raw_profiles": len(times),
        "spatial_nodes": len(positions),
        "exported_profiles": len(selected_indices),
        "exported_rows_each_format": expected_samples,
        "excluded_profiles": len(times) - len(selected_indices),
        "first_isolated_accepted_time_s": float(times[first_isolated_index]),
        "first_isolated_accepted_Fo_nu": float(fourier_values[first_isolated_index]),
        "first_sustained_accepted_time_s": window_start,
        "first_sustained_accepted_Fo_nu": float(fourier_values[first_sustained_index]),
        "accepted_window_end_time_s": window_end,
        "accepted_window_end_Fo_nu": float(fourier_values[-1]),
        "maximum_accepted_profile_error_fraction": float(np.max(profile_max_normalized[selected_indices])),
        "Fo_nu_min_max_increment_raw": [float(fourier_values[0]), float(fourier_values[-1]), float(fourier_values[1] - fourier_values[0])],
        "r": float(r_number),
        "backward_euler_max_discrete_residual": max_residual,
        "final_steady_profile_error_m_per_s": steady_error,
        "temporal_refinement_final_errors_m_per_s": {"20_intervals": err20, "40_intervals": err40},
        "repository_external_status_unchanged": True,
        "checks": {
            "point_csv_parquet_equivalence": True,
            "accepted_window_only_export": True,
            "profile_summary_consistency": True,
            "Fo_nu_and_r": True,
            "boundary_initial_conditions": True,
            "boundedness_and_monotonicity": True,
            "analytical_reference_all_profiles": True,
            "discrete_momentum_balance": True,
            "steady_limit": True,
            "temporal_refinement": True,
            "schema_and_target_leakage": True,
            "manifest_checksums": True,
            "repository_isolation": True,
        },
        "limitation": "Only one physical case, mesh, and timestep are available; mesh sensitivity and coarsening safety are unproven.",
    }
    return metrics


def write_validation_artifacts(metrics: dict, input_path: Path, config_path: Path) -> None:
    input_path = Path(input_path).resolve()
    config_path = Path(config_path).resolve()
    VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = VALIDATION_ROOT / "validation_report.md"
    case = _read_json(DATA_ROOT / "structured_data_2ML_case_summary.json")
    lines = [
        "# Independent Validation Report",
        "",
        "## Governing physics",
        "PASS: Raw analytical profiles match the existing analytical reference at all 21 stored times.",
        f"PASS: Backward-Euler discrete momentum residual max is {metrics['backward_euler_max_discrete_residual']:.3e}.",
        "PASS: Fo_nu was independently recomputed as nu*time/H^2; r was independently recomputed as nu*dt/dy^2.",
        "",
        "## Initial and boundary conditions",
        "PASS: Initial interior is at rest and both wall conditions hold at every stored time.",
        "PASS: Velocity remains bounded and is monotone in time and space.",
        "",
        "## Transient numerical checks",
        f"PASS: Raw case has {metrics['raw_profiles']} profiles x {metrics['spatial_nodes']} nodes = {metrics['raw_rows']} source rows.",
        f"PASS: Export includes only {metrics['exported_profiles']} sustained accepted profiles ({metrics['exported_rows_each_format']} rows) from Fo_nu={metrics['first_sustained_accepted_Fo_nu']:.2f} to {metrics['accepted_window_end_Fo_nu']:.2f}.",
        f"PASS: Normalized profile maximum error remains below 5%; largest exported fraction is {metrics['maximum_accepted_profile_error_fraction']:.8g}.",
        "PASS: No failing or isolated early-pass profile samples appear in either training table.",
        "",
        "## Analytical and steady-limit comparison",
        f"PASS: Final absolute distance from the steady profile is {metrics['final_steady_profile_error_m_per_s']:.8g} m/s.",
        f"PASS: Halving timestep reduces final analytical error ({metrics['temporal_refinement_final_errors_m_per_s']['20_intervals']:.8g} to {metrics['temporal_refinement_final_errors_m_per_s']['40_intervals']:.8g} m/s).",
        "",
        "## Test and output evidence",
        "PASS: Recorded package tests passed; Parquet and CSV point rows agree; profile summaries, schema, and manifest were independently checked.",
        f"PASS: Source CSV SHA-256 is unchanged: `{metrics['final_source_sha256']}`.",
        "PASS: Repository status outside to_ML matches the initial package baseline.",
        "",
        "## Problems found",
        "No correctness failures found in the generated package.",
        "The isolated t=0 match is documented but excluded from training rows; the 5 s and 10 s failing profiles are excluded from all pointwise/profile-summary training tables.",
        "Gradient/curvature-based fine/coarse labels are heuristics only.",
        "",
        "## Recommended next action",
        "Use only the sustained accepted-window rows for the current ML baseline. Generate separate mesh and timestep refinement cases before making any mesh-sensitivity or safe-coarsening claim.",
        "",
        "## Acceptance-window record",
        f"- First isolated accepted profile: t={case['first_isolated_accepted_profile']['time_s']:.2f} s, Fo_nu={case['first_isolated_accepted_profile']['fourier_viscous_number']:.2f}.",
        f"- First sustained accepted profile: t={case['first_sustained_accepted_profile']['time_s']:.2f} s, Fo_nu={case['first_sustained_accepted_profile']['fourier_viscous_number']:.2f}.",
        f"- Exported interval: t={case['accepted_window_start_time_s']:.2f}..{case['accepted_window_end_time_s']:.2f} s.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    checksummed_paths = [
        Path(input_path), Path(REPOSITORY_ROOT / yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["simulation_source"]),
        *sorted(DATA_ROOT.glob("structured_data_2ML_*")),
        PACKAGE_ROOT / "README.md", PACKAGE_ROOT / "requirements.txt", PACKAGE_ROOT / "run_to_ML.sh",
        PACKAGE_ROOT / "config" / "postprocess_config.yaml", PACKAGE_ROOT / "scripts" / "build_ml_dataset.py",
        PACKAGE_ROOT / "scripts" / "validate_ml_dataset.py", PACKAGE_ROOT / "tests" / "test_build_ml_dataset.py",
        PACKAGE_ROOT / "validation" / "discovery_report.md", PACKAGE_ROOT / "validation" / "validation_report.md",
        PACKAGE_ROOT / "validation" / "validation_metrics.json", PACKAGE_ROOT / "validation" / "test_result.json",
    ]
    checksum_lines = []
    for path in checksummed_paths:
        if path.is_file():
            checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(REPOSITORY_ROOT).as_posix()}")
    (VALIDATION_ROOT / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPOSITORY_ROOT / "results" / "transient_velocity_profiles.csv")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config" / "postprocess_config.yaml")
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    args = parser.parse_args(argv)
    try:
        metrics = validate(args.input, args.config, args.package_root)
        (VALIDATION_ROOT / "validation_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        write_validation_artifacts(metrics, args.input, args.config)
    except Exception as error:
        failure_metrics = {"status": "FAIL", "error": str(error)}
        VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)
        (VALIDATION_ROOT / "validation_metrics.json").write_text(
            json.dumps(failure_metrics, indent=2) + "\n", encoding="utf-8"
        )
        (VALIDATION_ROOT / "validation_report.md").write_text(
            f"# Independent Validation Report\n\n## Problems found\nFAIL: {error}\n", encoding="utf-8"
        )
        print(f"VALIDATION RESULT: FAIL - {error}", file=sys.stderr)
        return 1
    print("VALIDATION RESULT: PASS")
    print(f"Export window: Fo_nu {metrics['first_sustained_accepted_Fo_nu']:.2f}..{metrics['accepted_window_end_Fo_nu']:.2f}")
    print(f"Parquet/CSV rows: {metrics['exported_rows_each_format']}")
    print(f"Source checksum unchanged: {metrics['source_checksum_unchanged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
