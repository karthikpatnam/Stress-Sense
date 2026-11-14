"""Comprehensive unit and integration tests for StressSense Interactive Dashboard (Phase 4 Stage 3)."""

import json
from pathlib import Path
import pytest

from src.dashboard.data_builder import DashboardDataBuilder
from src.dashboard.generator import DashboardGenerator
from src.dashboard.server import DashboardServer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestDashboardDataBuilder:
    """Tests for the dashboard data extraction and payload building."""

    @pytest.fixture(scope="class")
    def builder(self):
        return DashboardDataBuilder()

    @pytest.fixture(scope="class")
    def payload(self, builder):
        return builder.build_payload()

    def test_load_kpis(self, builder):
        kpis = builder.load_kpis()
        assert isinstance(kpis, dict)
        assert kpis["total_patches"] == 640
        assert kpis["annotated_patches"] == 480
        assert kpis["anomalous_patches"] == 130
        assert kpis["healthy_annotated_patches"] == 350
        assert kpis["healthy_patches_total"] == 510
        assert kpis["unlabeled_test_patches"] == 160
        assert kpis["valid_canopy_regions"] == 8920
        assert kpis["valid_reduction_pct_mean"] == 60.06
        assert kpis["high_stress_adaptive_mean"] == 100.0
        assert kpis["high_stress_equal_mean"] == 39.02
        assert kpis["composite_recall_adaptive_mean"] == 48.37
        assert kpis["energy_reduction_pct_mean"] == 60.06
        assert kpis["spark_throughput_pps"] == 43.13

    def test_load_patch_table(self, builder):
        table = builder.load_patch_table()
        assert isinstance(table, list)
        assert len(table) == 640
        sample = table[0]
        assert "patch_id" in sample
        assert "split" in sample
        assert "field_id" in sample
        assert "valid_regions" in sample
        assert "k_selected" in sample
        assert "valid_reduction_pct" in sample

    def test_load_category_sensitivity(self, builder):
        sens = builder.load_category_sensitivity()
        assert "thresholds" in sens
        assert sens["thresholds"] == ["0%", "5%", "10%", "20%"]
        assert "series" in sens
        assert "standing_water" in sens["series"]
        sw_macro = sens["series"]["standing_water"]["adaptive_macro"]
        assert len(sw_macro) == 4
        assert sw_macro[-1] == 100.0  # 100% recall at 20% overlap

    def test_load_spark_scaling(self, builder):
        scaling = builder.load_spark_scaling()
        assert "empirical" in scaling
        assert "projections" in scaling
        assert len(scaling["projections"]) == 4
        assert scaling["empirical"]["single_machine"]["throughput_patches_per_sec"] == 52.26

    def test_extract_representative_patches(self, builder):
        patches = builder.extract_representative_patches()
        assert isinstance(patches, list)
        assert len(patches) == 8
        categories = {p["category"] for p in patches}
        assert "standing_water" in categories
        assert "waterway" in categories
        assert "planter_skip" in categories
        assert "cloud_shadow" in categories
        assert "double_plant" in categories
        assert "weed_cluster" in categories
        assert "healthy" in categories

        # Verify each patch has 16 regions and base64 imagery
        for p in patches:
            assert len(p["regions"]) == 16
            assert p["rgb_b64"].startswith("data:image/png;base64,")
            assert p["nir_b64"].startswith("data:image/png;base64,")
            assert p["rsdi_b64"].startswith("data:image/png;base64,")

    def test_build_payload_schema(self, payload):
        assert "metadata" in payload
        assert "kpis" in payload
        assert "patch_table" in payload
        assert "category_sensitivity" in payload
        assert "spark_scaling" in payload
        assert "representative_patches" in payload


class TestDashboardGenerator:
    """Tests for the HTML compilation and output file."""

    def test_generate_html_file(self, tmp_path):
        target = tmp_path / "test_dashboard.html"
        generator = DashboardGenerator()
        out = generator.generate_html(target)
        assert out.is_file()
        assert out.stat().st_size > 500_000  # Over 500 KB

        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "StressSense" in content
        assert "id=\"dashboard-payload\"" in content
        assert "tab-overview" in content
        assert "tab-inspector" in content
        assert "tab-sensitivity" in content
        assert "tab-spark" in content
        assert "tab-catalog" in content
        assert "tab-audit" in content


class TestDashboardServer:
    """Tests for the server configuration and port finding."""

    def test_find_available_port(self):
        server = DashboardServer(port=8050)
        port = server.find_available_port()
        assert isinstance(port, int)
        assert 8050 <= port <= 8070

    def test_ensure_dashboard_built(self):
        server = DashboardServer()
        p = server.ensure_dashboard_built()
        assert p.is_file()
