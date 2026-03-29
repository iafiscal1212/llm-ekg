"""
Tests for LLM EKG Security Layer.

Uses the synthetic data generator from demo.py — zero hardcoded data.

(c) 2026 Carmen Esteban — IAFISCAL & PARTNERS
"""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

# Import from the project
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_ekg.engine import LLMAnalyzer, FEATURE_NAMES, N_FEATURES
from llm_ekg.security import SecurityBaseline, SecurityReport
from llm_ekg.report import EKGReport

# Use demo's synthetic generator for real (non-hardcoded) data
from demo import generate_synthetic_conversation


def _build_analyzer(responses):
    """Helper: ingest responses into a fresh analyzer."""
    analyzer = LLMAnalyzer(n_scales=6)
    for r in responses:
        analyzer.ingest(
            response=r["response"],
            timestamp=r.get("timestamp", 0.0),
            response_time_s=r.get("response_time_s", 0.0),
        )
    return analyzer


def _get_normal_responses(seed=42):
    """Get only the normal-phase responses (first 50)."""
    all_responses = generate_synthetic_conversation(seed=seed)
    return all_responses[:50]


def _get_critical_responses(seed=42):
    """Get only the critical-phase responses (last 20)."""
    all_responses = generate_synthetic_conversation(seed=seed)
    return all_responses[80:]


class TestBaselineSaveLoad:
    def test_roundtrip(self):
        """Baseline save → load preserves all data."""
        responses = _get_normal_responses()
        analyzer = _build_analyzer(responses)
        bl = SecurityBaseline.from_analyzer(analyzer, model="test-model")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            bl.save(path)
            loaded = SecurityBaseline.load(path)

            assert loaded.model == "test-model"
            assert loaded.n_responses == len(responses)
            assert len(loaded.feature_stats) == N_FEATURES

            for orig, ld in zip(bl.feature_stats, loaded.feature_stats):
                assert orig["name"] == ld["name"]
                assert abs(orig["mean"] - ld["mean"]) < 1e-10
                assert abs(orig["std"] - ld["std"]) < 1e-10
        finally:
            Path(path).unlink(missing_ok=True)

    def test_json_structure(self):
        """Saved JSON has expected keys."""
        responses = _get_normal_responses()
        analyzer = _build_analyzer(responses)
        bl = SecurityBaseline.from_analyzer(analyzer)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            bl.save(path)
            data = json.loads(Path(path).read_text())
            assert "version" in data
            assert "feature_stats" in data
            assert "hurst_stats" in data
            assert "state_stats" in data
            assert "n_responses" in data
            assert len(data["feature_stats"]) == N_FEATURES
            for fs in data["feature_stats"]:
                assert all(k in fs for k in ["name", "mean", "std", "min", "max", "median", "p5", "p95"])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_minimum_responses(self):
        """Baseline creation fails with < 5 responses."""
        responses = _get_normal_responses()[:3]
        analyzer = _build_analyzer(responses)
        with pytest.raises(ValueError, match="at least 5"):
            SecurityBaseline.from_analyzer(analyzer)


class TestCleanDetection:
    def test_same_distribution_is_clean(self):
        """Same data as baseline → CLEAN."""
        responses = _get_normal_responses(seed=42)
        analyzer_bl = _build_analyzer(responses)
        bl = SecurityBaseline.from_analyzer(analyzer_bl)

        # Use same distribution (different seed for slight variation)
        responses2 = _get_normal_responses(seed=43)
        analyzer_check = _build_analyzer(responses2)
        report = bl.check(analyzer_check, sigma=3.0)

        assert report.status == "CLEAN"
        assert report.n_deviated == 0


class TestCompromisedDetection:
    def test_critical_vs_normal_baseline(self):
        """Critical responses against normal baseline → WARNING or COMPROMISED."""
        normal = _get_normal_responses(seed=42)
        analyzer_bl = _build_analyzer(normal)
        bl = SecurityBaseline.from_analyzer(analyzer_bl)

        critical = _get_critical_responses(seed=42)
        analyzer_check = _build_analyzer(critical)
        report = bl.check(analyzer_check, sigma=3.0)

        assert report.status in ("WARNING", "COMPROMISED")
        assert report.n_deviated >= 1

    def test_deviated_features_have_high_z(self):
        """Deviated features should have weighted z > sigma."""
        normal = _get_normal_responses(seed=42)
        analyzer_bl = _build_analyzer(normal)
        bl = SecurityBaseline.from_analyzer(analyzer_bl)

        critical = _get_critical_responses(seed=42)
        analyzer_check = _build_analyzer(critical)
        report = bl.check(analyzer_check, sigma=3.0)

        for d in report.deviations:
            if d["deviated"] and "_hurst_triggered" not in d:
                assert d["weighted_z"] > 3.0


class TestSigmaThreshold:
    def test_lower_sigma_detects_more(self):
        """Lower sigma should detect >= as many deviations as higher sigma."""
        normal = _get_normal_responses(seed=42)
        analyzer_bl = _build_analyzer(normal)
        bl = SecurityBaseline.from_analyzer(analyzer_bl)

        all_responses = generate_synthetic_conversation(seed=42)
        mixed = all_responses[40:90]  # Mix of normal + degraded
        analyzer_check = _build_analyzer(mixed)

        report_low = bl.check(analyzer_check, sigma=1.5)
        report_high = bl.check(analyzer_check, sigma=5.0)

        assert report_low.n_deviated >= report_high.n_deviated


class TestSecurityWeighting:
    def test_weighted_features_detected_earlier(self):
        """Security-weighted features should have higher weighted_z than raw z."""
        normal = _get_normal_responses(seed=42)
        analyzer_bl = _build_analyzer(normal)
        bl = SecurityBaseline.from_analyzer(analyzer_bl)

        critical = _get_critical_responses(seed=42)
        analyzer_check = _build_analyzer(critical)
        report = bl.check(analyzer_check, sigma=3.0)

        for d in report.deviations:
            if d["weight"] > 1.0:
                assert d["weighted_z"] >= d["z_score"]


class TestReportIntegration:
    def test_report_with_security_section(self):
        """Report with security section contains expected HTML."""
        normal = _get_normal_responses(seed=42)
        analyzer_bl = _build_analyzer(normal)
        bl = SecurityBaseline.from_analyzer(analyzer_bl)

        critical = _get_critical_responses(seed=42)
        analyzer_check = _build_analyzer(critical)
        sec_report = bl.check(analyzer_check, sigma=3.0)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            report = EKGReport(analyzer_check, security_report=sec_report)
            report.generate(path)
            html = Path(path).read_text()
            assert "Security Status" in html
            assert "Security Verdict" in html
            assert sec_report.status in html
            assert "Features Deviated" in html
        finally:
            Path(path).unlink(missing_ok=True)

    def test_report_without_security_backward_compat(self):
        """Report without security section is identical to v1.0 behavior."""
        normal = _get_normal_responses(seed=42)
        analyzer = _build_analyzer(normal)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            report = EKGReport(analyzer)
            report.generate(path)
            html = Path(path).read_text()
            assert "Executive Summary" in html
            assert "Hallucination Monitor" in html
            assert "Security Status" not in html
        finally:
            Path(path).unlink(missing_ok=True)
