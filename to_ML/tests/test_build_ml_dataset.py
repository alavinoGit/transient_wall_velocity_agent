import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from to_ML.scripts import build_ml_dataset as build


@pytest.fixture(scope="module")
def built_package(tmp_path_factory):
    output_root = tmp_path_factory.mktemp("generated-package")
    input_path = build.DEFAULT_INPUT
    original_hash = build.sha256_file(input_path)
    result = build.build_package(input_path, build.DEFAULT_CONFIG, output_root)
    assert build.sha256_file(input_path) == original_hash
    return output_root, result


def test_output_paths_must_remain_inside_to_ml():
    with pytest.raises(ValueError, match="Refusing to write outside"):
        build.ensure_within(build.PACKAGE_ROOT, Path("/tmp/outside-to-ml"))
    inside = build.PACKAGE_ROOT / "data" / "safe-output"
    assert build.ensure_within(build.PACKAGE_ROOT, inside) == inside.resolve()


def test_required_columns_and_finite_values_are_validated(tmp_path):
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("time_s,y_m\n0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        build.read_raw_csv(invalid_csv)


def test_only_sustained_accepted_window_is_exported(built_package):
    output_root, result = built_package
    summary = result["case_summary"]
    with (output_root / "data/structured_data_2ML.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert summary["raw_profile_count"] == 21
    assert summary["raw_point_count"] == 1071
    assert summary["first_isolated_accepted_profile"]["time_s"] == 0.0
    assert summary["first_sustained_accepted_profile"]["time_s"] == 15.0
    assert summary["accepted_window_start_fourier_viscous_number"] == pytest.approx(0.15)
    assert summary["accepted_window_end_fourier_viscous_number"] == pytest.approx(1.0)
    assert summary["exported_profile_count"] == 18
    assert summary["exported_point_count"] == 18 * 51
    assert len(rows) == 18 * 51
    assert {float(row["time_s"]) for row in rows} == set(np.arange(15.0, 100.1, 5.0))
    assert all(float(row["normalized_absolute_error"]) < 0.05 for row in rows)
    assert all(row["profile_accepted"] == "True" and row["accepted_window_member"] == "True" for row in rows)
    assert not any(float(row["time_s"]) < 15.0 for row in rows)


def test_isolated_and_sustained_acceptance_are_distinguished():
    windows = build.find_acceptance_windows([0.0, 5.0, 10.0, 15.0], [True, False, True, True])
    assert windows["first_isolated_index"] == 0
    assert windows["first_sustained_index"] == 2
    assert windows["sustained_end_index"] == 3


def test_raw_values_and_pointwise_errors_are_preserved_for_exported_window(built_package):
    output_root, _ = built_package
    source_rows, _ = build.read_raw_csv(build.DEFAULT_INPUT)
    source_lookup = {
        (row["time_s"], row["y_m"]): row
        for row in source_rows
        if row["time_s"] >= 15.0
    }
    output_rows = build._read_csv if hasattr(build, "_read_csv") else None
    with (output_root / "data/structured_data_2ML.csv").open(newline="", encoding="utf-8") as stream:
        output_rows = list(csv.DictReader(stream))
    for row in output_rows:
        source = source_lookup[(float(row["time_s"]), float(row["y_m"]))]
        numerical = float(row["numerical_velocity_m_per_s"])
        analytical = float(row["analytical_velocity_m_per_s"])
        signed_error = numerical - analytical
        assert numerical == source["numerical_velocity_m_per_s"]
        assert analytical == source["analytical_velocity_m_per_s"]
        assert float(row["source_error_m_per_s"]) == source["error_m_per_s"]
        assert float(row["recomputed_signed_error_m_per_s"]) == pytest.approx(signed_error)
        assert float(row["recomputed_absolute_error_m_per_s"]) == pytest.approx(abs(signed_error))
        assert float(row["normalized_absolute_error"]) == pytest.approx(abs(signed_error))
        assert float(row["squared_error_m2_per_s2"]) == pytest.approx(signed_error**2)


def test_profiles_have_unique_ordered_coordinates_and_local_derivatives(built_package):
    output_root, _ = built_package
    rows = []
    with (output_root / "data/structured_data_2ML.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["profile_id"], []).append(row)
    assert len(grouped) == 18
    assert len({(row["profile_id"], row["y_m"]) for row in rows}) == len(rows)
    for profile in grouped.values():
        y = np.array([float(row["y_m"]) for row in profile])
        u = np.array([float(row["numerical_velocity_m_per_s"]) for row in profile])
        assert len(profile) == 51
        assert np.all(np.diff(y) > 0.0)
        first = np.gradient(u, y, edge_order=2)
        second = np.gradient(first, y, edge_order=2)
        assert np.allclose(first, [float(row["du_dy_m_per_s_per_m"]) for row in profile])
        assert np.allclose(second, [float(row["d2u_dy2_m_per_s_per_m2"]) for row in profile])
        assert all(float(row["local_delta_y_m"]) > 0.0 for row in profile)


def test_profile_summary_error_statistics_match_source(built_package):
    output_root, _ = built_package
    with (output_root / "data/structured_data_2ML_profile_summary.csv").open(newline="", encoding="utf-8") as stream:
        summaries = list(csv.DictReader(stream))
    raw, _ = build.read_raw_csv(build.DEFAULT_INPUT)
    grouped = {}
    for row in raw:
        if row["time_s"] >= 15.0:
            grouped.setdefault(row["time_s"], []).append(row)
    assert len(summaries) == 18
    for summary in summaries:
        rows = grouped[float(summary["time_s"])]
        errors = np.array([row["numerical_velocity_m_per_s"] - row["analytical_velocity_m_per_s"] for row in rows])
        assert float(summary["profile_max_absolute_error_m_per_s"]) == pytest.approx(np.max(np.abs(errors)))
        assert float(summary["profile_max_normalized_error"]) == pytest.approx(np.max(np.abs(errors)))
        assert float(summary["profile_mae_m_per_s"]) == pytest.approx(np.mean(np.abs(errors)))
        assert float(summary["profile_rmse_m_per_s"]) == pytest.approx(np.sqrt(np.mean(errors**2)))
        assert float(summary["profile_relative_l2_error"]) >= 0.0


def test_missing_raw_r_is_derived_with_provenance(built_package):
    output_root, result = built_package
    _, raw_columns = build.read_raw_csv(build.DEFAULT_INPUT)
    assert "numerical_diffusion_number_r" not in raw_columns
    metadata = json.loads((output_root / "data/structured_data_2ML_metadata.json").read_text())
    assert metadata["derived_grid"]["numerical_diffusion_number_r"] == pytest.approx(125.0)
    assert metadata["provenance"]["numerical_diffusion_number_r"] == "derived_from_simulation_viscosity_and_csv_dt_dy"
    assert result["case_summary"]["training_eligible_under_threshold"] is True


def test_parquet_csv_equivalence_schema_and_manifest_checksums(built_package):
    output_root, _ = built_package
    data_root = output_root / "data"
    csv_rows = []
    with (data_root / "structured_data_2ML.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    parquet_rows = pq.read_table(data_root / "structured_data_2ML.parquet").to_pylist()
    assert len(csv_rows) == len(parquet_rows) == 918
    for csv_row, parquet_row in zip(csv_rows, parquet_rows):
        assert csv_row["profile_id"] == parquet_row["profile_id"]
        assert float(csv_row["time_s"]) == pytest.approx(parquet_row["time_s"])
        assert float(csv_row["normalized_absolute_error"]) == pytest.approx(parquet_row["normalized_absolute_error"])
    schema = json.loads((data_root / "structured_data_2ML_schema.json").read_text())
    assert {field["field_name"] for field in schema["fields"]} == set(csv_rows[0])
    assert all({"type", "units", "dimensional", "description", "source", "derivation", "null_policy", "valid_range", "ml_role"} <= field.keys() for field in schema["fields"])
    with (data_root / "structured_data_2ML_manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    for item in manifest:
        path = build.REPOSITORY_ROOT / item["path"]
        assert build.sha256_file(path) == item["sha256"]


def test_source_checksum_and_repository_scope_are_preserved(built_package):
    _, result = built_package
    discovery = (build.PACKAGE_ROOT / "validation/discovery_report.md").read_text()
    expected_hash = "32450e06346194e13299a78934d3b7b0eb6fd50e4918e6fbba6ae712fdb6be98"
    assert expected_hash in discovery
    assert build.sha256_file(build.DEFAULT_INPUT) == expected_hash
    assert result["metadata"]["source"]["input_csv_sha256"] == expected_hash


def test_fine_coarse_heuristics_and_wall_exclusion_are_explicit(built_package):
    output_root, _ = built_package
    with (output_root / "data/structured_data_2ML.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["fine_grid_heuristic"] in {"True", "False"} for row in rows)
    assert all(row["coarse_grid_candidate_heuristic"] in {"True", "False"} for row in rows)
    assert all(row["within_wall_exclusion_zone"] in {"True", "False"} for row in rows)
    metadata = json.loads((output_root / "data/structured_data_2ML_metadata.json").read_text())
    assert metadata["gradient_curvature_heuristics"]["wall_exclusion_fraction"] == pytest.approx(0.05)
    assert "grid convergence requires" in metadata["gradient_curvature_heuristics"]["coarse_label_caveat"]
